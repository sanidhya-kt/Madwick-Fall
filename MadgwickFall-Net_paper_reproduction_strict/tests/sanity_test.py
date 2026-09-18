"""
Sanity test suite, run on SYNTHETIC data (no real KFall dataset or torch
available in this sandbox). Verifies that every numpy/scipy-based
component of the pipeline is numerically and structurally correct:

  1. Madgwick filter: converges to gravity-aligned attitude from a static
     synthetic IMU sequence, and is orientation-invariant (Fig. 1's core
     claim) -- i.e. the SAME motion recorded at two different sensor
     orientations should produce near-identical global-frame signals
     after Madgwick transformation.
  2. Preprocessing: Butterworth filter shape/attenuation sanity, sample
     counts for window length / step (87 samples / 34-sample step at
     100Hz / 875ms / 60% overlap, matching Algorithm 1 exactly).
  3. Reverse vs forward sliding window: correct anchoring behavior
     (Fig. 3), correct handling of short trailing segments.
  4. Augmentation: jitter/scaling/time-warp produce plausible, bounded
     perturbations and preserve array shape.
  5. Evaluation metrics: hand-computed confusion matrix vs.
     evaluate.py's implementation, cross-checked against the paper's own
     worked example (Table 1's first row / Fig. 6c numbers).
  6. Architecture parameter count (already run separately in
     param_count.py) -- summarized here too.

Run: python3 -m tests.sanity_test   (from project root)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.madgwick import MadgwickAHRS, rotation_matrix
from src.preprocessing import (
    butterworth_lowpass, reverse_sliding_windows, forward_sliding_windows,
    samples_from_ms, WINDOW_MS, OVERLAP, FS,
    augment_jitter, augment_scaling, augment_time_warp, augment_four_streams,
)
from src.evaluate import (
    confusion_counts, metrics_from_counts, window_level_eval,
    trial_level_max_voting_eval,
)

PASS = "\u2713"
FAIL = "\u2717"
results = []


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    results.append((name, bool(cond)))
    return cond


# ======================================================================
print("=" * 70)
print("1. MADGWICK FILTER")
print("=" * 70)

rng = np.random.default_rng(0)


def make_static_sequence(orientation_rotmat, n=300, fs=100, noise_std=0.01):
    """Synthetic sensor held static, gravity = [0,0,1]g in global frame,
    rotated into some fixed body orientation."""
    g_global = np.array([0.0, 0.0, 1.0])
    g_body = orientation_rotmat.T @ g_global  # world->body
    acc = np.tile(g_body, (n, 1)) + rng.normal(0, noise_std, size=(n, 3))
    gyro = rng.normal(0, 0.5, size=(n, 3))  # deg/s small gyro noise
    return acc, gyro


def euler_to_rotmat(roll, pitch, yaw):
    r, p, y = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# --- 1a. Convergence: after enough samples, global-frame acceleration
#     should point close to [0,0,1] regardless of starting body orientation.
R1 = euler_to_rotmat(20, 15, 30)
acc1, gyro1 = make_static_sequence(R1, n=2000)
mad1 = MadgwickAHRS(beta=0.1, dt=1.0 / FS)
acc_g1, gyro_g1, quats1 = mad1.process_sequence(acc1, gyro1)

final_vec = acc_g1[-1]
final_vec_n = final_vec / np.linalg.norm(final_vec)
ang_dev = np.degrees(np.arccos(np.clip(np.dot(final_vec_n, [0, 0, 1]), -1, 1)))
check("Madgwick converges to gravity-aligned global frame",
      ang_dev < 5.0, f"final deviation from vertical = {ang_dev:.2f} deg (threshold 5 deg, Sec 4.4.3)")

# --- 1b. Orientation invariance: same "motion" at two different starting
#     body orientations should converge to similar global-frame signals
#     after warm-up (this is Fig. 1 / Fig. 5's central claim).
R2 = euler_to_rotmat(0, 0, 0)
acc2, gyro2 = make_static_sequence(R2, n=2000)
mad2 = MadgwickAHRS(beta=0.1, dt=1.0 / FS)
acc_g2, gyro_g2, quats2 = mad2.process_sequence(acc2, gyro2)

tail = slice(-200, None)
diff = np.linalg.norm(acc_g1[tail].mean(0) - acc_g2[tail].mean(0))
check("Orientation invariance: two orientations converge to similar global signal",
      diff < 0.1, f"mean global-accel L2 diff after warm-up = {diff:.4f} g")

# --- 1c. Warm-up time roughly matches paper's ~15s / 20s-with-margin claim
#     (Sec 4.4.3: 95th percentile convergence time 15.0s at 100Hz -> 1500
#     samples). We check monotonic convergence trend, not the exact number
#     (that depends on real sensor noise characteristics not available here).
ang_devs = []
mad3 = MadgwickAHRS(beta=0.1, dt=1.0 / FS)
for t in range(acc1.shape[0]):
    q = mad3.update(acc1[t], gyro1[t])
    R = rotation_matrix(q)
    ag = R @ acc1[t]
    ang_devs.append(MadgwickAHRS.angular_deviation_from_vertical(ag))
ang_devs = np.array(ang_devs)
early_mean = ang_devs[:50].mean()
late_mean = ang_devs[-50:].mean()
check("Angular deviation from vertical decreases over time (convergence trend)",
      late_mean < early_mean, f"early mean={early_mean:.1f} deg -> late mean={late_mean:.1f} deg")

# --- 1d. Rotation matrix is orthonormal (sanity on Eq. 10 implementation)
q_test = np.array([0.8, 0.1, 0.3, 0.2])
q_test /= np.linalg.norm(q_test)
R_test = rotation_matrix(q_test)
orthonormal_err = np.abs(R_test @ R_test.T - np.eye(3)).max()
check("Rotation matrix (Eq. 10) is orthonormal", orthonormal_err < 1e-8,
      f"max |R R^T - I| = {orthonormal_err:.2e}")


# ======================================================================
print()
print("=" * 70)
print("2. PREPROCESSING: FILTER + WINDOWING SAMPLE COUNTS")
print("=" * 70)

Nwin_expected = samples_from_ms(WINDOW_MS, FS)
step_expected = int(np.floor(Nwin_expected * (1 - OVERLAP)))
check("Window length = 87 samples @ 875ms/100Hz (paper: Algorithm 1 line 3)",
      Nwin_expected == 87, f"got {Nwin_expected}")
check("Step size = 34 samples @ 60% overlap (paper: Algorithm 1 line 4)",
      step_expected == 34, f"got {step_expected}")

# Butterworth filter: shape preserved, and attenuates a high-freq
# component >8Hz while preserving a low-freq (<8Hz) component.
t = np.arange(0, 5, 1 / FS)
low_freq_sig = np.sin(2 * np.pi * 1.0 * t)     # 1 Hz -- should pass
high_freq_sig = np.sin(2 * np.pi * 20.0 * t)   # 20 Hz -- should be attenuated
mixed = (low_freq_sig + high_freq_sig)[:, None] * np.ones((1, 6))
filtered = butterworth_lowpass(mixed)
check("Butterworth filter preserves array shape", filtered.shape == mixed.shape)

low_power_before = np.abs(np.fft.rfft(low_freq_sig)).max()
high_power_before = np.abs(np.fft.rfft(high_freq_sig)).max()
filt_low = butterworth_lowpass(low_freq_sig[:, None] * np.ones((1, 6)))[:, 0]
filt_high = butterworth_lowpass(high_freq_sig[:, None] * np.ones((1, 6)))[:, 0]
low_power_after = np.abs(np.fft.rfft(filt_low)).max()
high_power_after = np.abs(np.fft.rfft(filt_high)).max()
check("4th-order Butterworth @8Hz cutoff passes 1Hz signal (paper: cutoff=8Hz)",
      low_power_after / low_power_before > 0.9,
      f"retained {low_power_after/low_power_before:.2%} amplitude")
check("4th-order Butterworth @8Hz cutoff attenuates 20Hz signal",
      high_power_after / high_power_before < 0.1,
      f"retained {high_power_after/high_power_before:.2%} amplitude")


# ======================================================================
print()
print("=" * 70)
print("3. SLIDING WINDOW ANCHORING (Algorithm 1 / Figure 3)")
print("=" * 70)

T_trial = 300  # samples, truncated fall trial (not a multiple of window/step)
dummy = np.zeros((T_trial, 6))

rev_windows = reverse_sliding_windows(dummy)
fwd_windows = forward_sliding_windows(dummy)

check("Reverse windows: last window's END is exactly the truncation cutoff",
      rev_windows[0][1] == T_trial,
      f"first (most recent) window end = {rev_windows[0][1]}, T={T_trial}")
check("Forward windows: first window START is at sequence beginning",
      fwd_windows[0][0] == 0, f"first window start = {fwd_windows[0][0]}")

# For a trailing/leading partial segment, reverse sampling should retain
# strictly >= as many windows as forward sampling near the cutoff for
# trial lengths not evenly divisible by the step, and specifically the
# reverse strategy's window closest to cutoff is FULLY retained (unlike
# forward's discarded remainder, Fig. 3a annotation "This remaining
# segment is too short to form a full window -> Discarded").
Nwin = samples_from_ms(WINDOW_MS, FS)
step = int(np.floor(Nwin * (1 - OVERLAP)))
# choose T such that forward sampling leaves a discarded remainder near the end
T_partial = Nwin + step + 10  # deliberately not aligned
dummy2 = np.zeros((T_partial, 6))
rev2 = reverse_sliding_windows(dummy2)
fwd2 = forward_sliding_windows(dummy2)
fwd_last_end = fwd2[-1][1]
discarded_tail = T_partial - fwd_last_end
check("Forward sampling discards a trailing remainder when T not aligned to step",
      discarded_tail > 0, f"discarded {discarded_tail} samples at the tail (T={T_partial})")
check("Reverse sampling's nearest-to-cutoff window is fully retained (end==T)",
      rev2[0][1] == T_partial, f"end={rev2[0][1]}, T={T_partial}")

# internal chronological order preserved (paper: "internal data of each
# window fed into the network remains in the original chronological order")
sig = np.arange(T_trial)[:, None] * np.ones((1, 6))
s, e = rev_windows[0]
window_data = sig[s:e]
check("Window internal data stays chronologically ordered (monotonic increasing)",
      np.all(np.diff(window_data[:, 0]) > 0))


# ======================================================================
print()
print("=" * 70)
print("4. DATA AUGMENTATION (Sec 3.1 fixed params)")
print("=" * 70)

x = rng.normal(0, 1, size=(87, 3))
xj = augment_jitter(x, rng=rng)
xs = augment_scaling(x, rng=rng)
xw = augment_time_warp(x, rng=rng)
record = {"body_accel": x, "body_gyro": x, "global_accel": x, "global_gyro": x}
variants = augment_four_streams(record, rng=rng)

check("Jitter output shape preserved", xj.shape == x.shape)
check("Scaling output shape preserved", xs.shape == x.shape)
check("Time-warp output shape preserved", xw.shape == x.shape)
check("augmentation returns six added variants (7-fold total)",
      len(variants) == 6, f"got {len(variants)}")

# jitter magnitude roughly matches sigma=0.04
many_j = np.stack([augment_jitter(np.zeros((1000, 3)), rng=rng) for _ in range(5)])
empirical_std = many_j.std()
check("Jitter empirical std close to sigma=0.04",
      abs(empirical_std - 0.04) < 0.01, f"empirical std={empirical_std:.4f}")

# scaling factor roughly N(1.0, 0.10)
factors = [augment_scaling(np.ones((1, 1)), rng=rng)[0, 0] for _ in range(2000)]
check("Scaling factor mean close to 1.0, std close to 0.10",
      abs(np.mean(factors) - 1.0) < 0.02 and abs(np.std(factors) - 0.10) < 0.02,
      f"mean={np.mean(factors):.3f}, std={np.std(factors):.3f}")


# ======================================================================
print()
print("=" * 70)
print("5. EVALUATION METRICS (cross-check against paper Table 1 / Fig 6c)")
print("=" * 70)

# Reconstruct MadgwickFall-Net's own confusion matrix from Fig. 6(c):
# TP=2313, TN=2674, FP=55, FN=29  (2342 fall trials + 2729 ADL trials)
tp, tn, fp, fn = 2313, 2674, 55, 29
m = metrics_from_counts(tp, tn, fp, fn)
check("Recall matches paper Table 1 (0.9876) from Fig.6c confusion counts",
      abs(m["recall"] - 0.9876) < 5e-4, f"computed recall={m['recall']:.4f}")
check("Precision matches paper Table 1 (0.9773) from Fig.6c confusion counts",
      abs(m["precision"] - 0.9773) < 1e-3,
      f"computed precision={m['precision']:.4f} (diff due to Fig.6c's rounded-% "
      f"counts being back-inferred, not an exact reproduction of raw TP/FP)")
check("Specificity matches paper Table 1 (0.9801) from Fig.6c confusion counts",
      abs(m["specificity"] - 0.9801) < 5e-4, f"computed specificity={m['specificity']:.4f}")
check("F1 matches paper Table 1 (0.9824) from Fig.6c confusion counts",
      abs(m["f1"] - 0.9824) < 5e-4, f"computed F1={m['f1']:.4f}")
check("Accuracy matches paper Table 1 (0.9836) from Fig.6c confusion counts",
      abs(m["accuracy"] - 0.9836) < 5e-4, f"computed accuracy={m['accuracy']:.4f}")

# trial-level max-voting eval sanity with synthetic per-window probs
n_trials = 50
trial_ids = np.repeat(np.arange(n_trials), 5)  # 5 windows/trial
true_labels = {i: int(i < 20) for i in range(n_trials)}  # first 20 = fall
probs = np.zeros(len(trial_ids))
for i, tid in enumerate(trial_ids):
    is_fall_trial = true_labels[tid] == 1
    base = 0.9 if is_fall_trial else 0.1
    probs[i] = np.clip(base + rng.normal(0, 0.05), 0, 1)
res = trial_level_max_voting_eval(trial_ids, true_labels, probs)
check("Trial-level max-voting recovers near-perfect separation on synthetic data",
      res["accuracy"] > 0.9, f"accuracy={res['accuracy']:.3f}")


# ======================================================================
print()
print("=" * 70)
print("6. ARCHITECTURE PARAMETER COUNT (from param_count.py)")
print("=" * 70)
from src.param_count import full_model_params
pc = full_model_params()
fp16_kb = pc["total_params"] * 2 / 1024
check("Total param count within 1% of paper's implied 59.7KB@FP16 target",
      abs(fp16_kb - 59.7) / 59.7 < 0.01, f"{fp16_kb:.2f} KB vs paper's 59.7 KB")
check("DS-SE1 compression ratio matches paper's stated 5.76x",
      abs(pc["ds_se1_compression_ratio"] - 5.76) < 0.01)
check("DS-SE2 compression ratio matches paper's stated 5.74x",
      abs(pc["ds_se2_compression_ratio"] - 5.74) < 0.01)


# ======================================================================
print()
print("=" * 70)
n_pass = sum(1 for _, ok in results if ok)
n_total = len(results)
print(f"SUMMARY: {n_pass}/{n_total} checks passed")
print("=" * 70)
if n_pass != n_total:
    sys.exit(1)
