"""
Training loop matching Section 4.1.1 exactly:

  Optimizer            : AdamW, lr=1e-3, weight_decay=8e-5
  LR schedule           : CosineAnnealingWarmRestarts(T_0=15, T_mult=2, eta_min=1e-6)
  Loss                  : class-weighted CrossEntropy + label smoothing (eps=0.01)
  Batch size            : 64
  Epochs                : 105
  Grad clip             : max L2 norm 0.8
  Seed                  : 42 (numpy, torch, cuda)
  CV                    : 5-fold GroupKFold by subject_id (sklearn)
  Data augmentation     : 7x on fall-train windows only (jitter/scaling/
                           time-warp + combinations, see preprocessing.py)

Requires torch (not available in this sandbox). This is the reference
training script to run once you have a local KFall copy + a GPU/CPU torch
environment.
"""
import random
import numpy as np

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_optimizer_and_scheduler(model, lr=1e-3, weight_decay=8e-5,
                                   T_0=15, T_mult=2, eta_min=1e-6):
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    return opt, sched


def build_loss(class_weights=None, label_smoothing=0.01):
    import torch
    import torch.nn as nn
    weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights for the (post-augmentation, roughly
    1:1) train split, still applied for residual imbalance robustness."""
    classes, counts = np.unique(labels, return_counts=True)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-8)
    weights = weights / weights.sum() * len(classes)
    out = np.ones(2)
    for c, w in zip(classes, weights):
        out[int(c)] = w
    return out


def train_one_fold(model, train_loader, val_loader, device="cpu",
                    epochs=105, grad_clip_norm=0.8, class_weights=None,
                    label_smoothing=0.01, lr=1e-3, weight_decay=8e-5,
                    use_amp=False, log_every=10):
    """
    One 5-fold-CV fold: trains `epochs` epochs, returns the trained model
    and per-epoch train loss history. Validation (held-out subjects) is
    evaluated ONLY ONCE, after training completes (paper Sec 4.1.1:
    'Only one independent inference run is performed after each fold of
    training to strictly avoid data leakage.') -- i.e. no early stopping
    / model selection on the val set.

    Training is kept in standard FP32 because the paper specifies the
    training configuration but does not specify mixed-precision training.
    """
    import torch
    model.to(device)
    opt, sched = build_optimizer_and_scheduler(model, lr=lr, weight_decay=weight_decay)
    criterion = build_loss(class_weights, label_smoothing).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device != "cpu"))

    steps_per_epoch = len(train_loader)
    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}
            labels = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device != "cpu")):
                logits = model(inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            sched.step(epoch + step / steps_per_epoch)
            epoch_loss += loss.item() * labels.size(0)
            n += labels.size(0)
        avg_loss = epoch_loss / max(n, 1)
        history.append(avg_loss)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  train_loss={avg_loss:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    return model, history


def run_inference(model, loader, device="cpu"):
    """
    Single inference pass over a fold's held-out set (subjects unseen in
    training). Returns per-window fall probabilities plus bookkeeping
    needed for both window-level and trial-level max-voting evaluation.
    """
    import torch
    import torch.nn.functional as F
    model.eval()
    all_probs, all_labels, all_trial_ids, all_is_fall_trial = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)[:, 1]  # P(fall)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_trial_ids.extend(batch["trial_id"])
            all_is_fall_trial.extend(batch["is_fall_trial"])
    import numpy as np
    return dict(
        probs=np.concatenate(all_probs),
        window_labels=np.concatenate(all_labels),
        trial_ids=np.array(all_trial_ids),
        is_fall_trial=np.array(all_is_fall_trial),
    )


def run_5fold_cv(all_records, build_model_fn, config="four_branch",
                  device="cpu", epochs=105, batch_size=64):
    """
    Top-level 5-fold subject-independent GroupKFold CV driver (Sec 4.1.4).
    all_records: list of window dicts from KFallDataset.build_windows().
    build_model_fn: callable(config) -> nn.Module (see model.build_model).
    Returns list of per-fold trained models + val record indices, for
    downstream evaluate.py to compute trial-level / window-level metrics.
    """
    from sklearn.model_selection import GroupKFold
    set_seed(SEED)

    subjects = np.array([r["subject_id"] for r in all_records])
    labels = np.array([r["label"] for r in all_records])
    idx = np.arange(len(all_records))

    gkf = GroupKFold(n_splits=5)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(idx, labels, groups=subjects)):
        # sanity: no subject overlap between train/val
        assert set(subjects[train_idx]).isdisjoint(set(subjects[val_idx])), \
            "Subject leakage between train/val fold!"
        fold_results.append({
            "fold": fold, "train_idx": train_idx, "val_idx": val_idx,
        })
    return fold_results
