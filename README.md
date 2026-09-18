# Pre-Impact Fall Detection

Reproduction code for a Madgwick-filter-based fall detection pipeline using wearable sensor data.

## Project

The implementation is in [`MadgwickFall-Net_paper_reproduction_strict/`](./MadgwickFall-Net_paper_reproduction_strict/), including:

- Madgwick orientation filtering and preprocessing
- Model training and evaluation scripts
- Ablation and parameter-count utilities
- Sanity tests

## Setup

```bash
cd MadgwickFall-Net_paper_reproduction_strict
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the available sanity test with:

```bash
python -m unittest discover -s tests
```

The kFall dataset is intentionally excluded from this repository. Place a local copy in
`kFall/KFall Dataset/` when running experiments.
