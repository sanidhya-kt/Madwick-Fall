# MadgwickFall-Net reproduction

Reproduction code for:

**Q. Zhong, J. Wang, G. Sun, “MadgwickFall-Net: A Lightweight Dual-Frame Feature Fusion Network for Pre-Impact Fall Detection Using Wearable IMUs,” Bioengineering 2026, 13, 568.**

## Paper protocol implemented

- KFall: 5075 trials, 32 subjects, 2729 ADL + 2346 fall.
- 6 raw channels only: acceleration (x,y,z) + angular velocity (x,y,z).
- Fall truncation: 250 ms before annotated impact.
- 4th-order Butterworth low-pass, 8 Hz.
- Madgwick coordinate transformation, beta=0.1, dt=0.01 s.
- Fall: reverse sliding window, 875 ms, 60% overlap, 87 samples, 34-sample step.
- ADL: forward sliding window with the same window/step.
- Subject-wise 5-fold GroupKFold.
- Fall training data: 7-fold total augmentation using jitter, scaling and time warping with the paper's stated parameters.
- AdamW: lr=0.001, weight_decay=8e-5.
- CosineAnnealingWarmRestarts: T0=15, Tmult=2, eta_min=1e-6.
- Class-weighted cross-entropy + label smoothing 0.01.
- Batch size 64, 105 epochs, gradient clipping 0.8, seed 42.
- Trial-level primary evaluation: maximum fall probability per trial, threshold 0.5.
- Window-level evaluation as the auxiliary metric.
- Optional rolling inference and the three ablation families described in the paper.

## Actual KFall layout used on the server

The loader accepts the KFall release layout:

```text
~/skt/Madgwick_fall/kFall/KFall Dataset/KFall Dataset/
├── label_data/
│   ├── SA06_label.xlsx
│   └── ...
└── sensor_data/
    ├── SA06/
    │   ├── S06T01R01.csv
    │   └── ...
    └── ...
```

The label parser uses the task code + trial number from each subject's XLSX and maps them to the sensor filename. It does not hard-code a fall-task list.

## Server path

The project directory remains:

```bash
~/skt/Madgwick_fall/madgwickfall_net_reproduction
```

After replacing the files from this archive:

```bash
cd ~/skt/Madgwick_fall/madgwickfall_net_reproduction
source .venv/bin/activate
pip install -r requirements.txt
```

Set the same dataset root you already use:

```bash
export KFALL_ROOT=~/skt/Madgwick_fall/kFall/KFall\ Dataset/KFall\ Dataset
```

## Main reproduction

```bash
cd ~/skt/Madgwick_fall/madgwickfall_net_reproduction
source .venv/bin/activate
export KFALL_ROOT=~/skt/Madgwick_fall/kFall/KFall\ Dataset/KFall\ Dataset

python3 run_reproduction.py \
  --kfall_root "$KFALL_ROOT" \
  --out_dir outputs/full_run \
  --device cuda \
  --epochs 105 \
  --batch_size 64 \
  --num_workers 16
```

The run first prints the KFall distribution. It must show:

```text
5075 trials
2346 fall
2729 ADL
```

The model has 30,522 trainable parameters, matching the paper's architecture/size claim.

## Rolling inference

To additionally reproduce Section 4.4.1:

```bash
python3 run_reproduction.py \
  --kfall_root "$KFALL_ROOT" \
  --out_dir outputs/full_run \
  --device cuda \
  --epochs 105 \
  --batch_size 64 \
  --num_workers 16 \
  --rolling_inference
```

## Ablations

To additionally run the paper's coordinate, window-length, beta, and forward-vs-reverse experiments:

```bash
python3 run_reproduction.py \
  --kfall_root "$KFALL_ROOT" \
  --out_dir outputs/full_with_ablations \
  --device cuda \
  --epochs 105 \
  --batch_size 64 \
  --num_workers 16 \
  --ablations
```

## Important paper-specified limitation

The paper states **7-fold total augmentation** and gives the exact parameters for jitter, scaling and time warping, but it does not enumerate the seven generated variants. This implementation therefore keeps the existing seven-way construction (original + three single operations + three pairwise combinations) rather than introducing a new augmentation method.

Likewise, the paper says fall-window labels are assigned from the overlap ratio with the annotated fall phase but does not state a numerical overlap threshold. The code retains the existing 0.5 overlap threshold in `assign_window_label`; this is an implementation detail that is **not explicitly specified by the paper** and must not be reported as an author-provided value.

No additional model block, loss term, sensor modality, or preprocessing method is added.
