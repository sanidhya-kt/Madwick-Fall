#!/usr/bin/env python3
"""End-to-end reproduction driver for MadgwickFall-Net."""
import argparse
import json
import os
import time

import numpy as np


def _augment_indices(base_records, train_idx):
    from src.preprocessing import augment_four_streams
    rng = np.random.default_rng(42)
    out = []
    for i in train_idx:
        r = base_records[i]
        out.append(r)
        if r["is_fall"] and r["label"] == 1:
            for variant_id, streams in enumerate(augment_four_streams(r, rng=rng), start=1):
                aug = dict(r)
                aug.update(streams)
                aug["aug_variant"] = variant_id
                out.append(aug)
    return out


def _make_splits(records):
    from sklearn.model_selection import GroupKFold
    subjects = np.array([r["subject_id"] for r in records])
    labels = np.array([r["label"] for r in records])
    idx = np.arange(len(records))
    return list(GroupKFold(n_splits=5).split(idx, labels, groups=subjects))


def _run_cv(records, args, model_config="four_branch", save_prefix=None):
    import torch
    from src.model import build_model, count_parameters
    from src.torch_dataset import make_loader
    from src.train import set_seed, train_one_fold, run_inference, compute_class_weights, SEED
    from src.evaluate import trial_level_max_voting_eval, window_level_eval

    set_seed(SEED)
    splits = _make_splits(records)
    fold_trial, fold_window, fold_conf = [], [], []
    fold_models = []

    subjects = np.array([r["subject_id"] for r in records])
    trial_ids = np.array([r["trial_id"] for r in records])

    for fold, (train_idx, val_idx) in enumerate(splits):
        assert set(subjects[train_idx]).isdisjoint(set(subjects[val_idx]))
        train_records = _augment_indices(records, train_idx)
        val_records = [records[i] for i in val_idx]

        train_loader = make_loader(train_records, list(range(len(train_records))),
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
        val_loader = make_loader(val_records, list(range(len(val_records))),
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        weights = compute_class_weights(np.array([r["label"] for r in train_records]))
        model = build_model(model_config)
        if fold == 0:
            print(f"    model params: {count_parameters(model)}")

        model, _ = train_one_fold(
            model, train_loader, val_loader, device=args.device,
            epochs=args.epochs, grad_clip_norm=0.8,
            class_weights=weights, label_smoothing=0.01,
            lr=1e-3, weight_decay=8e-5, use_amp=False,
        )

        result = run_inference(model, val_loader, device=args.device)
        true_by_trial = {}
        for i in val_idx:
            true_by_trial[trial_ids[i]] = int(records[i]["is_fall"])
        trial_m = trial_level_max_voting_eval(
            result["trial_ids"], true_by_trial, result["probs"]
        )
        window_m = window_level_eval(result["window_labels"], result["probs"])
        print(
            f"    fold {fold+1}: trial F1={trial_m['f1']:.4f} "
            f"Acc={trial_m['accuracy']:.4f} Rec={trial_m['recall']:.4f} "
            f"Prec={trial_m['precision']:.4f} Spe={trial_m['specificity']:.4f}"
        )
        fold_trial.append(trial_m)
        fold_window.append(window_m)
        fold_conf.append(trial_m["confusion"])
        fold_models.append((model, val_idx.copy()))

        if save_prefix:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"{save_prefix}_fold{fold}.pt"))

    keys = ("f1", "accuracy", "recall", "precision", "specificity")
    return {
        "trial_level_mean": {k: float(np.mean([m[k] for m in fold_trial])) for k in keys},
        "window_level_mean": {k: float(np.mean([m[k] for m in fold_window])) for k in keys},
        "per_fold_trial": fold_trial,
        "per_fold_window": fold_window,
        "confusions": fold_conf,
        "models": fold_models,
    }


def _rolling_inference(ds, cv_result, records, device):
    """Paper Sec. 4.4.1: forward 875-ms buffer, 340-ms step, first p>=0.5."""
    import torch
    from src.torch_dataset import make_loader
    from src.train import run_inference
    from src.preprocessing import forward_sliding_windows, WINDOW_MS, FS
    from src.evaluate import pre_impact_lead_time

    alarms = {}
    impacts = {}
    for model, val_idx in cv_result["models"]:
        val_subjects = {records[i]["subject_id"] for i in val_idx}
        val_trials = [t for t in ds.trials if t.subject_id in val_subjects and t.is_fall]
        for trial in val_trials:
            acc_b, gyro_b, acc_g, gyro_g = ds.process_trial_for_rolling(trial)
            nwin = int(WINDOW_MS * FS / 1000)
            windows = forward_sliding_windows(acc_b, WINDOW_MS, fs=FS)
            if not windows:
                continue
            recs = []
            for s, e in windows:
                if e > trial.fall_impact_idx:
                    break
                recs.append({
                    "subject_id": trial.subject_id, "trial_id": trial.trial_id,
                    "is_fall": True, "label": 1,
                    "body_accel": acc_b[s:e], "body_gyro": gyro_b[s:e],
                    "global_accel": acc_g[s:e], "global_gyro": gyro_g[s:e],
                })
            if not recs:
                continue
            loader = make_loader(recs, list(range(len(recs))), 64, False, 0)
            out = run_inference(model, loader, device=device)
            hits = np.where(out["probs"] >= 0.5)[0]
            impacts[trial.trial_id] = trial.fall_impact_idx * 1000.0 / FS
            if len(hits):
                first = int(hits[0])
                alarms[trial.trial_id] = windows[first][1] * 1000.0 / FS

    return pre_impact_lead_time(alarms, impacts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfall_root", required=True)
    ap.add_argument("--out_dir", default="outputs/full_run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=105)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--rolling_inference", action="store_true")
    args = ap.parse_args()

    import torch
    from src.dataset import KFallDataset
    from src.train import set_seed, SEED
    from src.model import count_parameters, build_model

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(SEED)
    print(f"Device: {args.device} | CUDA available: {torch.cuda.is_available()}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    ds = KFallDataset(args.kfall_root, beta=0.1)
    base_records = ds.build_windows(augment_fall_train=False, window_ms=875)
    print(f"Preprocessed {len(base_records)} windows in {time.time()-t0:.1f}s")
    print(f"Fall trials indexed: {sum(t.is_fall for t in ds.trials)}")
    print(f"ADL trials indexed: {sum(not t.is_fall for t in ds.trials)}")

    print("\n[MAIN] 5-fold subject-wise GroupKFold")
    result = _run_cv(base_records, args, "four_branch", "model")
    summary = {
        "paper_reference": {
            "f1": 0.9824, "precision": 0.9773, "recall": 0.9876,
            "specificity": 0.9801, "accuracy": 0.9836,
        },
        "trial_level_mean": result["trial_level_mean"],
        "window_level_mean": result["window_level_mean"],
        "per_fold_trial": result["per_fold_trial"],
        "per_fold_window": result["per_fold_window"],
        "confusions": result["confusions"],
        "n_params": count_parameters(build_model("four_branch")),
        "n_trials": len(ds.trials),
        "n_fall_trials": sum(t.is_fall for t in ds.trials),
        "n_adl_trials": sum(not t.is_fall for t in ds.trials),
    }
    with open(os.path.join(args.out_dir, "results_table1.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["trial_level_mean"], indent=2))

    if args.rolling_inference:
        rolling = _rolling_inference(ds, result, base_records, args.device)
        with open(os.path.join(args.out_dir, "rolling_inference.json"), "w") as f:
            json.dump(rolling, f, indent=2)
        print("Rolling inference:", json.dumps(rolling, indent=2))

    if not args.ablations:
        return

    print("\n[ABLATION 4.3.1] Coordinate system")
    coord = {}
    for cfg in ("global_only", "body_only"):
        r = _run_cv(base_records, args, cfg, f"ablation_{cfg}")
        coord[cfg] = {"trial_level_mean": r["trial_level_mean"]}
    with open(os.path.join(args.out_dir, "ablation_coordinate.json"), "w") as f:
        json.dump(coord, f, indent=2)

    print("\n[ABLATION 4.3.2] Window length")
    win = {}
    for w in (500, 625, 750, 875, 1000):
        recs = ds.build_windows(window_ms=w)
        r = _run_cv(recs, args, "four_branch", f"ablation_window_{w}")
        win[str(w)] = {"trial_level_mean": r["trial_level_mean"]}
    with open(os.path.join(args.out_dir, "ablation_window_length.json"), "w") as f:
        json.dump(win, f, indent=2)

    print("\n[ABLATION 4.3.3] Madgwick beta")
    beta = {}
    for b in (0.01, 0.05, 0.1, 0.2, 0.5):
        bds = KFallDataset(args.kfall_root, beta=b)
        recs = bds.build_windows(window_ms=875)
        r = _run_cv(recs, args, "four_branch", f"ablation_beta_{b}")
        beta[str(b)] = {"trial_level_mean": r["trial_level_mean"]}
    with open(os.path.join(args.out_dir, "ablation_beta.json"), "w") as f:
        json.dump(beta, f, indent=2)

    print("\n[ABLATION 4.4.2] Forward vs reverse")
    fv = {}
    for sampling in ("reverse", "forward"):
        recs = ds.build_windows(window_ms=875, fall_sampling=sampling)
        r = _run_cv(recs, args, "four_branch", f"ablation_{sampling}")
        fv[sampling] = {"trial_level_mean": r["trial_level_mean"]}
    with open(os.path.join(args.out_dir, "ablation_forward_reverse.json"), "w") as f:
        json.dump(fv, f, indent=2)


if __name__ == "__main__":
    main()
