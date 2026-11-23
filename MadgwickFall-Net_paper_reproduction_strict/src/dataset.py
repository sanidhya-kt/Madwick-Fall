"""KFall loading and the DualFrame-PFD preprocessing pipeline."""
import os
import glob
import re
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor
import copy

import numpy as np
import pandas as pd

from .madgwick import MadgwickAHRS
from .preprocessing import (
    FS, PRE_IMPACT_MARGIN_MS,
    butterworth_lowpass, truncate_fall_trial,
    reverse_sliding_windows, forward_sliding_windows,
    assign_window_label, augment_four_streams,
)


@dataclass
class TrialMeta:
    subject_id: str
    task_id: str
    trial_id: str
    is_fall: bool
    filepath: str
    fall_start_idx: Optional[int] = None
    fall_impact_idx: Optional[int] = None


class KFallDataset:
    """Load the public KFall release and implement the paper pipeline.

    The paper uses 6 raw channels only: triaxial acceleration and triaxial
    angular velocity. Euler angles supplied by KFall are not used.

    This loader accepts the actual KFall release layout used on the server:

        KFall Dataset/
          sensor_data/SA06/*.csv ...
          label_data/SA06_label.xlsx ...

    It also accepts a root that already points at ``sensor_data`` when labels
    are available in its sibling ``label_data`` directory.
    """

    def __init__(self, kfall_root: str, beta: float = 0.1, fs: int = FS):
        self.root = os.path.abspath(os.path.expanduser(kfall_root))
        self.beta = beta
        self.fs = fs
        if not os.path.isdir(self.root):
            raise FileNotFoundError(self.root)
        self.sensor_root, self.label_root = self._resolve_roots()
        self.labels = self._load_labels()
        self.trials: List[TrialMeta] = self._index_trials()

        falls = sum(t.is_fall for t in self.trials)
        adls = len(self.trials) - falls
        print(f"Loaded {len(self.labels)} label records from {self._label_file_count} subject label files")
        print(f"Indexed {len(self.trials)} trials from {len(set(t.subject_id for t in self.trials))} subjects")
        print(f"KFall distribution: {falls} fall / {adls} ADL")

    def _resolve_roots(self):
        if os.path.isdir(os.path.join(self.root, "sensor_data")):
            sensor_root = os.path.join(self.root, "sensor_data")
            label_root = os.path.join(self.root, "label_data")
            return sensor_root, label_root
        if os.path.basename(self.root).lower() == "sensor_data":
            return self.root, os.path.join(os.path.dirname(self.root), "label_data")
        return self.root, os.path.join(self.root, "label_data")

    @staticmethod
    def _subject_num(subject_id: str) -> int:
        m = re.search(r"(\d+)$", subject_id)
        if not m:
            raise ValueError(f"Invalid KFall subject id: {subject_id}")
        return int(m.group(1))

    def _load_labels(self):
        """Parse the actual per-subject KFall XLSX annotation files.

        KFall stores task code only on the first row of a task block; the
        following trial rows contain NaN in that column.  Forward filling is
        therefore required before mapping task number + trial number to the
        sensor filename SxxTyyRzz.csv.
        """
        files = sorted(glob.glob(os.path.join(self.label_root, "SA*_label.xlsx")))
        if not files:
            raise FileNotFoundError(
                f"No SA*_label.xlsx files found under {self.label_root}"
            )

        rows = []
        for fp in files:
            subject_id = os.path.basename(fp).split("_", 1)[0]
            df = pd.read_excel(fp)
            if df.shape[1] < 5:
                raise ValueError(f"Unexpected KFall label format: {fp}")

            task_raw = df.iloc[:, 0].ffill().astype(str).str.strip().str.upper()
            trial_num = pd.to_numeric(df.iloc[:, 2], errors="coerce")
            onset = pd.to_numeric(df.iloc[:, 3], errors="coerce")
            impact = pd.to_numeric(df.iloc[:, 4], errors="coerce")
            task_num = pd.to_numeric(
                task_raw.str.extract(r"\((\d+)\)", expand=False),
                errors="coerce",
            )

            valid = task_num.notna() & trial_num.notna()
            for task_code, tnum, rnum, on, imp in zip(
                task_raw[valid], task_num[valid], trial_num[valid], onset[valid], impact[valid]
            ):
                task = int(tnum)
                trial = int(rnum)
                trial_id = f"S{self._subject_num(subject_id):02d}T{task:02d}R{trial:02d}"
                is_fall = str(task_code).startswith("F")
                rows.append({
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "is_fall": is_fall,
                    "fall_start_frame": int(on) if is_fall and pd.notna(on) else None,
                    "fall_impact_frame": int(imp) if is_fall and pd.notna(imp) else None,
                })

        self._label_file_count = len(files)
        labels = pd.DataFrame(rows)
        if labels.duplicated(["subject_id", "trial_id"]).any():
            raise ValueError("Duplicate KFall annotations; refusing to discard ambiguity")
        return labels

    def _index_trials(self) -> List[TrialMeta]:
        trials = []
        subject_dirs = sorted(glob.glob(os.path.join(self.sensor_root, "SA*")))
        if not subject_dirs:
            raise FileNotFoundError(f"No SA* subject folders found under {self.sensor_root}")

        for sdir in subject_dirs:
            subject_id = os.path.basename(sdir)
            for fp in sorted(glob.glob(os.path.join(sdir, "*.csv"))):
                trial_id = os.path.splitext(os.path.basename(fp))[0]
                parsed = re.fullmatch(r"S(\d{2})T(\d{2})R(\d{2})", trial_id)
                if parsed is None or int(parsed[1]) != self._subject_num(subject_id):
                    raise ValueError(f"Invalid trial filename: {fp}")
                task = int(parsed[2])
                # Release task IDs: ADL 1..19,35,36; falls 20..34.
                # Missing FALL annotations remain fatal.
                if task not in range(1, 37):
                    raise ValueError(f"Unknown KFall task: {trial_id}")
                is_fall = 20 <= task <= 34
                match = self.labels[
                    (self.labels.subject_id == subject_id) &
                    (self.labels.trial_id == trial_id)
                ]
                if not is_fall and len(match) == 0:
                    trials.append(TrialMeta(subject_id, f"T{task:02d}", trial_id, False, fp))
                    continue
                if len(match) != 1:
                    raise ValueError(f"Missing/ambiguous label for {subject_id}/{trial_id}")
                row = match.iloc[0]
                if bool(row.is_fall) != is_fall:
                    raise ValueError(f"Task/annotation mismatch: {trial_id}")
                if is_fall and (pd.isna(row.fall_start_frame) or pd.isna(row.fall_impact_frame)
                                or not 1 <= row.fall_start_frame < row.fall_impact_frame):
                    raise ValueError(f"Invalid fall annotation: {trial_id}")
                trials.append(TrialMeta(
                    subject_id=subject_id,
                    task_id=trial_id,
                    trial_id=trial_id,
                    is_fall=bool(row.is_fall),
                    filepath=fp,
                    # FrameCounter starts at 1. Onset is an inclusive index;
                    # impact frame count is an exclusive pre-impact boundary.
                    fall_start_idx=int(row.fall_start_frame) - 1 if bool(row.is_fall) else None,
                    fall_impact_idx=int(row.fall_impact_frame) if bool(row.is_fall) else None,
                ))

        if len(trials) != 5075:
            raise ValueError(f"Expected 5075 KFall trials, found {len(trials)}")
        if sum(t.is_fall for t in trials) != 2346 or len({t.subject_id for t in trials}) != 32:
            raise ValueError("Unexpected KFall subject/class distribution")
        indexed_falls = {(t.subject_id, t.trial_id) for t in trials if t.is_fall}
        if set(zip(self.labels.subject_id, self.labels.trial_id)) != indexed_falls:
            raise ValueError("Annotation files and fall recordings do not match")
        return trials

    def _read_trial_signal(self, trial: TrialMeta) -> np.ndarray:
        df = pd.read_csv(trial.filepath)
        normalized = {str(c).strip().lower(): c for c in df.columns}
        names = ["accx", "accy", "accz", "gyrx", "gyry", "gyrz"]
        if not all(n in normalized for n in names):
            raise ValueError(f"Required 6 IMU channels not found in {trial.filepath}")
        cols = [normalized[n] for n in names]
        return df[cols].to_numpy(dtype=np.float64)

    def build_windows(self, augment_fall_train: bool = False, rng=None, window_ms=875, fall_sampling="reverse", overlap_thresh=0.5, jobs=1, label_rule="window_fraction"):
        if jobs > 1:
            if augment_fall_train:
                raise ValueError("Augment only after splitting; parallel preprocessing does not augment")
            chunks = []
            for part in np.array_split(np.arange(len(self.trials)), jobs * 4):
                ds = copy.copy(self)
                ds.trials = [self.trials[i] for i in part]
                chunks.append((ds, window_ms, fall_sampling, overlap_thresh, label_rule))
            records = []
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                for i, recs in enumerate(pool.map(_build_chunk, chunks)):
                    records.extend(recs)
                    print(f"Preprocessing chunk {i+1}/{len(chunks)}: {len(records)} windows", flush=True)
            return records
        rng = rng or np.random.default_rng(42)
        records = []

        for trial in self.trials:
            raw = self._read_trial_signal(trial)
            if not np.isfinite(raw).all():
                raise ValueError(f"Nonfinite IMU data: {trial.trial_id}")
            if trial.is_fall and trial.fall_impact_idx > len(raw):
                raise ValueError(f"Impact outside recording: {trial.trial_id}")

            if trial.is_fall:
                if trial.fall_impact_idx is None:
                    continue
                raw = truncate_fall_trial(
                    raw, trial.fall_impact_idx,
                    margin_ms=PRE_IMPACT_MARGIN_MS, fs=self.fs,
                )
                if raw.shape[0] < int(window_ms * self.fs / 1000):
                    continue

            filtered = butterworth_lowpass(raw, fs=self.fs)
            acc_body = filtered[:, :3]
            gyro_body = filtered[:, 3:]

            mad = MadgwickAHRS(beta=self.beta, dt=1.0 / self.fs)
            acc_global, gyro_global, _ = mad.process_sequence(acc_body, gyro_body)

            windows = (
                reverse_sliding_windows(filtered, window_ms=window_ms)
                if trial.is_fall and fall_sampling == "reverse" else
                forward_sliding_windows(filtered, window_ms=window_ms)
            )

            for s, e in windows:
                label = (
                    assign_window_label(s, e, trial.fall_start_idx or 0, trial.fall_impact_idx,
                                        overlap_thresh, label_rule, observable_end=len(raw))
                    if trial.is_fall else 0
                )
                rec = {
                    "subject_id": trial.subject_id,
                    "trial_id": trial.trial_id,
                    "is_fall": trial.is_fall,
                    "label": label,
                    "window_start": s,
                    "window_end": e,
                    "body_accel": acc_body[s:e],
                    "body_gyro": gyro_body[s:e],
                    "global_accel": acc_global[s:e],
                    "global_gyro": gyro_global[s:e],
                }
                records.append(rec)

                if augment_fall_train and trial.is_fall and label == 1:
                    for variant_id, streams in enumerate(
                        augment_four_streams(rec, rng=rng), start=1
                    ):
                        aug = dict(rec)
                        aug.update(streams)
                        aug["aug_variant"] = variant_id
                        records.append(aug)

        return records

    def process_trial_for_rolling(self, trial: TrialMeta, causal=False):
        """Paper Sec. 4.4.1: process the complete trial before impact."""
        raw = self._read_trial_signal(trial)
        if not causal:
            raise ValueError("Rolling inference requires causal=True to exclude future samples")
        from .preprocessing import causal_butterworth_lowpass
        filtered = causal_butterworth_lowpass(raw, fs=self.fs)
        acc_body = filtered[:, :3]
        gyro_body = filtered[:, 3:]
        mad = MadgwickAHRS(beta=self.beta, dt=1.0 / self.fs)
        acc_global, gyro_global, _ = mad.process_sequence(acc_body, gyro_body)
        return acc_body, gyro_body, acc_global, gyro_global


def _build_chunk(args):
    ds, window_ms, fall_sampling, overlap_thresh, label_rule = args
    return ds.build_windows(window_ms=window_ms, fall_sampling=fall_sampling, overlap_thresh=overlap_thresh, label_rule=label_rule)
