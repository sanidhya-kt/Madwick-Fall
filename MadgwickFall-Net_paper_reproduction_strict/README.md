# MadgwickFall-Net

This directory contains the retained KFall implementation and latest completed five-fold run.

Retained output: `outputs/kfall_observable_quarter_20261123/`

Latest trial-level mean:

| Metric | Result |
|---|---:|
| F1 | 98.15% |
| Accuracy | 98.27% |
| Recall | 99.25% |
| Precision | 97.08% |
| Specificity | 97.43% |

Published paper F1 is 98.24%; this run is 0.09 percentage points lower.

The paper’s stated settings are retained: KFall 5,075 trials and 32 subjects; subject-wise 5-fold GroupKFold; six IMU channels at 100 Hz; 250 ms pre-impact margin; fourth-order Butterworth low-pass at 8 Hz; Madgwick beta 0.1, dt 0.01 s and 0.1 g threshold; 875 ms windows with 60% overlap; sevenfold fall augmentation; AdamW 0.001 / 8e-5; cosine restarts 15 / 2 / 1e-6; weighted cross entropy with smoothing 0.01; batch size 64; 105 epochs; gradient clipping 0.8; seed 42; four-branch model with 30,522 parameters.

The retained run’s unspecified choices are recorded in its manifests: observable-phase window-label overlap >= 0.25 and four total spline points including endpoints. The trial decision threshold is 0.5. Checkpoints retain model, optimizer, scheduler, scaler, RNG, sampler, history and metadata state.
