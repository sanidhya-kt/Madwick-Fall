#!/usr/bin/env python3
"""End-to-end reproduction driver for MadgwickFall-Net."""
import argparse
import json
import os
import time
import sys
import platform
import hashlib
import pickle
from pathlib import Path

import numpy as np


def _augment_indices(base_records, train_idx, synchronized=False):
    from src.preprocessing import augment_four_streams
    rng = np.random.default_rng(42)
    out = []
    for i in train_idx:
        r = base_records[i]
        out.append(r)
        if r["is_fall"] and r["label"] == 1:
            for variant_id, streams in enumerate(augment_four_streams(r, rng=rng, synchronized=synchronized), start=1):
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
        if args.fold is not None and fold != args.fold:
            continue
        assert set(subjects[train_idx]).isdisjoint(set(subjects[val_idx]))
        train_records = _augment_indices(records, train_idx, args.synchronized_augmentation)
        val_records = [records[i] for i in val_idx]

        train_loader = make_loader(train_records, list(range(len(train_records))),
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, training=True)
        val_loader = make_loader(val_records, list(range(len(val_records))),
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        weights = compute_class_weights(np.array([r["label"] for r in train_records]))
        model = build_model(model_config)
        if fold == 0:
            print(f"    model params: {count_parameters(model)}")

        model, history = train_one_fold(
            model, train_loader, val_loader, device=args.device,
            epochs=args.epochs, grad_clip_norm=0.8,
            class_weights=weights, label_smoothing=0.01,
            lr=1e-3, weight_decay=8e-5, use_amp=False,
            checkpoint_path=os.path.join(args.out_dir, f"{save_prefix}_fold{fold}_last.pt") if save_prefix else None,
            resume=args.resume,
            checkpoint_metadata={"fold": fold, "model_config": model_config,
                                 "source_sha256": args.source_hashes,
                                 "epochs": args.epochs, "batch_size": args.batch_size,
                                 "overlap_thresh": args.overlap_thresh,
                                 "label_rule": args.label_rule,
                                 "synchronized_augmentation": args.synchronized_augmentation,
                                 "train_subjects": sorted(set(subjects[train_idx])),
                                 "test_subjects": sorted(set(subjects[val_idx])),
                                 "dataset_root": args.kfall_root},
        )

        result = run_inference(model, val_loader, device=args.device)
        if save_prefix:
            np.savez_compressed(os.path.join(args.out_dir, f"{save_prefix}_fold{fold}_predictions.npz"), **result)
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
            metrics_path = os.path.join(args.out_dir, f"{save_prefix}_fold{fold}_metrics.json")
            with open(metrics_path + ".tmp", "w") as f:
                json.dump({"fold": fold, "trial": trial_m, "window": window_m,
                           "train_loss": history,
                           "train_subjects": sorted(set(subjects[train_idx])),
                           "test_subjects": sorted(set(subjects[val_idx])),
                           "train_class_counts": np.bincount([r["label"] for r in train_records], minlength=2).tolist(),
                           "training_singleton_dropped": len(train_records) % args.batch_size == 1}, f, indent=2)
            os.replace(metrics_path + ".tmp", metrics_path)

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
        val_ids = {records[i]["trial_id"] for i in val_idx}
        val_trials = [t for t in ds.trials if t.trial_id in val_ids and t.is_fall]
        for trial in val_trials:
            impacts[trial.trial_id] = (trial.fall_impact_idx - 1) * 1000.0 / FS
            acc_b, gyro_b, acc_g, gyro_g = ds.process_trial_for_rolling(trial, causal=True)
            nwin = int(WINDOW_MS * FS / 1000)
            windows = forward_sliding_windows(acc_b, WINDOW_MS, fs=FS)
            if not windows:
                continue
            recs = []
            for s, e in windows:
                if e >= trial.fall_impact_idx:
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
            if len(hits):
                first = int(hits[0])
                alarms[trial.trial_id] = (windows[first][1] - 1) * 1000.0 / FS

    return pre_impact_lead_time(alarms, impacts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfall_root", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=105)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--rolling_inference", action="store_true")
    ap.add_argument("--fold", type=int, choices=range(5), help="Train only this zero-based fold")
    ap.add_argument("--preprocess_workers", type=int, default=8)
    ap.add_argument("--cache", help="Preprocessed cache created by this checkout (trusted local pickle)")
    ap.add_argument("--prepare_only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume each fold from its latest complete epoch")
    ap.add_argument("--label_rule", choices=("window_fraction", "any_overlap", "observable_fraction"), default="observable_fraction",
                    help="Explicit interpretation of unspecified paper labelling; any_overlap uses ratio > 0")
    ap.add_argument("--overlap_thresh", type=float, default=0.25,
                    help="Unspecified by paper; retained implementation assumption, never tune on held-out subjects")
    ap.add_argument("--synchronized_augmentation", action=argparse.BooleanOptionalAction, default=True,
                    help="Share scaling and time warp across the four streams")
    args = ap.parse_args()

    import torch
    from src.dataset import KFallDataset
    from src.train import set_seed, SEED
    from src.model import count_parameters, build_model

    os.makedirs(args.out_dir, exist_ok=True)
    if not 0 < args.overlap_thresh <= 1:
        raise ValueError("overlap_thresh must be in (0,1]")
    torch.set_num_threads(2)
    source_hashes = {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("src/*.py"))}
    source_hashes["run_reproduction.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.source_hashes = source_hashes
    manifest = {"arguments": vars(args), "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "source_sha256": source_hashes,
                "status": "independent_implementation_with_unresolved_paper_details",
                "assumptions": {"window_label_rule": args.label_rule,
                                "window_overlap_threshold": 0.0 if args.label_rule == "any_overlap" else args.overlap_thresh,
                                "augmentation_variants": "original + three singles + three pairs",
                                "timewarp_knots": "4 total points including endpoints; endpoint convention unspecified by paper",
                                "observable_fraction_denominator": "min(window length, cutoff minus onset)" if args.label_rule == "observable_fraction" else None,
                                "synchronized_augmentation": args.synchronized_augmentation,
                                "offline_filter": "filtfilt (phase unspecified by paper)",
                                "rolling_filter": "causal SOS, separate streaming validation",
                                "batching": "drop final singleton training batch only",
                                "onset_index": "one-based frame converted to zero-based inclusive onset"}}
    set_seed(SEED)
    print(f"Device: {args.device} | CUDA available: {torch.cuda.is_available()}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    ds = KFallDataset(args.kfall_root, beta=0.1)
    cache_key = {"source_sha256": source_hashes, "root": ds.root, "overlap_thresh": args.overlap_thresh, "label_rule": args.label_rule}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            cached = pickle.load(f)
        if cached["key"] != cache_key:
            raise ValueError("Cache does not match code/dataset/config; choose a new cache path")
        base_records = cached["records"]
    else:
        base_records = ds.build_windows(augment_fall_train=False, window_ms=875,
                                       overlap_thresh=args.overlap_thresh, jobs=args.preprocess_workers, label_rule=args.label_rule)
        if args.cache:
            with open(args.cache + ".tmp", "wb") as f:
                pickle.dump({"key": cache_key, "records": base_records}, f, protocol=5)
            os.replace(args.cache + ".tmp", args.cache)
    retained = {r["trial_id"] for r in base_records if r["is_fall"]}
    positives = {r["trial_id"] for r in base_records if r["label"] == 1}
    diagnostic = {"label_rule": args.label_rule, "n_windows": len(base_records), "retained_falls": len(retained),
                  "falls_without_positive_window": len(retained - positives),
                  "window_class_counts": np.bincount([r["label"] for r in base_records], minlength=2).tolist(),
                  "excluded_fall_ids": sorted(t.trial_id for t in ds.trials if t.is_fall and t.trial_id not in retained),
                  "fold_test_subjects": [sorted({base_records[i]["subject_id"] for i in test})
                                         for _, test in _make_splits(base_records)]}
    if len(retained) != 2342:
        raise ValueError(f"Expected paper cohort of 2342 falls, found {len(retained)}")
    if args.label_rule in ("any_overlap", "observable_fraction") and retained - positives:
        raise ValueError("Annotation-overlap experiment has falls without positive supervision")
    print(json.dumps(diagnostic, indent=2), flush=True)
    if args.prepare_only:
        return
    print(f"Preprocessed {len(base_records)} windows in {time.time()-t0:.1f}s")
    print(f"Fall trials indexed: {sum(t.is_fall for t in ds.trials)}")
    print(f"ADL trials indexed: {sum(not t.is_fall for t in ds.trials)}")

    print("\n[MAIN] 5-fold subject-wise GroupKFold")
    # Release mode writes only the final result table.
    result = _run_cv(base_records, args, "four_branch", None)
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
    summary["folds_completed"] = len(result["per_fold_trial"])
    summary["status"] = manifest["status"]
    filename = "results_table1.json" if args.fold is None else f"results_fold{args.fold}.json"
    with open(os.path.join(args.out_dir, filename), "w") as f:
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
        recs = ds.build_windows(window_ms=w, label_rule=args.label_rule)
        r = _run_cv(recs, args, "four_branch", f"ablation_window_{w}")
        win[str(w)] = {"trial_level_mean": r["trial_level_mean"]}
    with open(os.path.join(args.out_dir, "ablation_window_length.json"), "w") as f:
        json.dump(win, f, indent=2)

    print("\n[ABLATION 4.3.3] Madgwick beta")
    beta = {}
    for b in (0.01, 0.05, 0.1, 0.2, 0.5):
        bds = KFallDataset(args.kfall_root, beta=b)
        recs = bds.build_windows(window_ms=875, label_rule=args.label_rule)
        r = _run_cv(recs, args, "four_branch", f"ablation_beta_{b}")
        beta[str(b)] = {"trial_level_mean": r["trial_level_mean"]}
    with open(os.path.join(args.out_dir, "ablation_beta.json"), "w") as f:
        json.dump(beta, f, indent=2)

    print("\n[ABLATION 4.4.2] Forward vs reverse")
    fv = {}
    for sampling in ("reverse", "forward"):
        recs = ds.build_windows(window_ms=875, fall_sampling=sampling, label_rule=args.label_rule)
        r = _run_cv(recs, args, "four_branch", f"ablation_{sampling}")
        fv[sampling] = {"trial_level_mean": r["trial_level_mean"]}
    with open(os.path.join(args.out_dir, "ablation_forward_reverse.json"), "w") as f:
        json.dump(fv, f, indent=2)


if __name__ == "__main__":
    main()
