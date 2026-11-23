"""Preprocessing and augmentation specified by MadgwickFall-Net."""
import numpy as np
from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi
from scipy.interpolate import CubicSpline

FS = 100
WINDOW_MS = 875
OVERLAP = 0.60
PRE_IMPACT_MARGIN_MS = 250
BUTTER_ORDER = 4
BUTTER_CUTOFF_HZ = 8.0

JITTER_SIGMA = 0.04
SCALING_MEAN = 1.0
SCALING_STD = 0.10
TIMEWARP_SIGMA = 0.15
TIMEWARP_KNOTS = 4


def samples_from_ms(ms, fs=FS):
    return int(np.floor(ms * fs / 1000.0))


def butterworth_lowpass(x, cutoff=BUTTER_CUTOFF_HZ, fs=FS, order=BUTTER_ORDER):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, x, axis=0)


def causal_butterworth_lowpass(x, cutoff=BUTTER_CUTOFF_HZ, fs=FS, order=BUTTER_ORDER):
    """Streaming validation variant; phase/initial state are not specified by paper."""
    sos = butter(order, cutoff, fs=fs, btype="low", output="sos")
    zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    return sosfilt(sos, x, axis=0, zi=zi)[0]


def truncate_fall_trial(data, t_impact_idx, margin_ms=PRE_IMPACT_MARGIN_MS, fs=FS):
    cutoff_idx = max(int(t_impact_idx) - samples_from_ms(margin_ms, fs), 0)
    return data[:cutoff_idx]


def reverse_sliding_windows(data, window_ms=WINDOW_MS, overlap=OVERLAP, fs=FS):
    nwin = samples_from_ms(window_ms, fs)
    step = int(np.floor(nwin * (1.0 - overlap)))
    pos = data.shape[0]
    windows = []
    while pos - nwin >= 0:
        windows.append((pos - nwin, pos))
        pos -= step
    return windows


def forward_sliding_windows(data, window_ms=WINDOW_MS, overlap=OVERLAP, fs=FS):
    nwin = samples_from_ms(window_ms, fs)
    step = int(np.floor(nwin * (1.0 - overlap)))
    start = 0
    windows = []
    while start + nwin <= data.shape[0]:
        windows.append((start, start + nwin))
        start += step
    return windows


def assign_window_label(win_start, win_end, fall_start_idx, fall_impact_idx,
                        overlap_thresh=0.5, label_rule="window_fraction", observable_end=None):
    """The paper omits its numerical labelling rule; keep interpretations explicit.

    window_fraction is the original reproduction's unverified 50% rule.
    any_overlap is a documented alternative (overlap ratio > 0), not an
    author-provided threshold. observable_fraction normalizes by the shorter
    of the window and retained fall phase, with an explicit cutoff argument.
    All rules keep pre-onset windows negative.
    """
    inter_start = max(win_start, fall_start_idx)
    phase_end = observable_end if label_rule == "observable_fraction" else fall_impact_idx
    if phase_end is None:
        raise ValueError("observable_fraction requires the pre-impact cutoff")
    inter_end = min(win_end, phase_end)
    overlap_len = max(0, inter_end - inter_start)
    if label_rule == "any_overlap":
        return int(overlap_len > 0)
    if label_rule == "observable_fraction":
        # Normalize by the shorter of the window and observable fall phase.
        # This avoids erasing short falls while rejecting tiny boundary overlap.
        denominator = min(win_end - win_start, phase_end - fall_start_idx)
        return int(denominator > 0 and overlap_len / denominator >= overlap_thresh)
    if label_rule != "window_fraction":
        raise ValueError(f"Unknown label rule: {label_rule}")
    return int((overlap_len / (win_end - win_start)) >= overlap_thresh)


def augment_jitter(x, rng):
    return x + rng.normal(0.0, JITTER_SIGMA, size=x.shape)


def augment_scaling(x, rng):
    return x * rng.normal(SCALING_MEAN, SCALING_STD)


def augment_time_warp(x, rng):
    t = np.arange(x.shape[0], dtype=float)
    # Literal four-point interpretation, including the two fixed endpoints.
    knot_x = np.linspace(0, x.shape[0] - 1, TIMEWARP_KNOTS)
    knot_y = knot_x + rng.normal(
        0.0, TIMEWARP_SIGMA * x.shape[0] / (TIMEWARP_KNOTS - 1),
        size=knot_x.shape,
    )
    knot_y[0], knot_y[-1] = 0.0, x.shape[0] - 1.0
    knot_y = np.sort(knot_y)
    warp = CubicSpline(knot_x, knot_y)(t)
    warp = np.clip(warp, 0, x.shape[0] - 1)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(warp, t, x[:, c])
    return out


def _apply_variant(streams, kind, rng, synchronized=False):
    if synchronized and kind == "scaling":
        scale = rng.normal(SCALING_MEAN, SCALING_STD)
        return {name: x * scale for name, x in streams.items()}
    if synchronized and kind == "time_warp":
        # One temporal map shared by all axes and both coordinate frames.
        names = list(streams)
        warped = augment_time_warp(np.concatenate([streams[k] for k in names], axis=1), rng)
        return {name: part for name, part in zip(names, np.split(warped, len(names), axis=1))}
    out = {}
    for name, x in streams.items():
        if kind == "jitter":
            out[name] = augment_jitter(x, rng)
        elif kind == "scaling":
            out[name] = augment_scaling(x, rng)
        elif kind == "time_warp":
            out[name] = augment_time_warp(x, rng)
        else:
            raise ValueError(kind)
    return out


def augment_four_streams(record, rng=None, synchronized=False):
    """Create the six augmented variants needed for 7-fold total data.

    The paper specifies the three augmentation operations and the total
    7-fold expansion, but does not enumerate the seven variants.  The
    implementation uses the same seven-way construction already used by the
    reproduction package: original + three single operations + three pairwise
    combinations.  No additional augmentation method is introduced.
    """
    rng = rng or np.random.default_rng(42)
    base = {
        "body_accel": np.asarray(record["body_accel"]),
        "body_gyro": np.asarray(record["body_gyro"]),
        "global_accel": np.asarray(record["global_accel"]),
        "global_gyro": np.asarray(record["global_gyro"]),
    }

    # Seven total samples: original + 6 augmented variants.
    j = _apply_variant(base, "jitter", rng, synchronized)
    s = _apply_variant(base, "scaling", rng, synchronized)
    w = _apply_variant(base, "time_warp", rng, synchronized)
    js = _apply_variant(j, "scaling", rng, synchronized)
    jw = _apply_variant(j, "time_warp", rng, synchronized)
    sw = _apply_variant(s, "time_warp", rng, synchronized)
    return [j, s, w, js, jw, sw]
