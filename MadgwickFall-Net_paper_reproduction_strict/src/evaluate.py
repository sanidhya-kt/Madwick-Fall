"""
Evaluation metrics matching Section 4.1.4 exactly.

Two strategies:
  1. Window-level: standard per-window Acc/Rec/Spe/Prec/F1 (auxiliary,
     for comparability with prior literature).
  2. Trial-level max-voting (PRIMARY, used in Table 1 / Fig 6): for every
     trial, take the max predicted fall-probability over all its windows
     as the trial score; threshold at 0.5 -> trial-level prediction.

Plus:
  - Pre-impact lead-time rolling-inference protocol (Sec 4.4.1).
  - Paired Wilcoxon signed-rank + paired t-test across 5 folds (Sec 4.2).
"""
import numpy as np


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def metrics_from_counts(tp, tn, fp, fn, eps=1e-8):
    acc = (tp + tn) / max(tp + tn + fp + fn, eps)
    rec = tp / max(tp + fn, eps)          # Eq. 23
    spe = tn / max(tn + fp, eps)          # Eq. 24
    prec = tp / max(tp + fp, eps)         # Eq. 25
    f1 = 2 * prec * rec / max(prec + rec, eps)  # Eq. 26
    return dict(accuracy=acc, recall=rec, specificity=spe, precision=prec, f1=f1)


def window_level_eval(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    return metrics_from_counts(tp, tn, fp, fn)


def trial_level_max_voting_eval(trial_ids, y_true_trial_label, y_prob,
                                 threshold=0.5):
    """
    trial_ids: array-like, per-window trial identifier.
    y_true_trial_label: dict trial_id -> 0/1 ground-truth (fall/ADL trial).
    y_prob: per-window predicted fall probability.
    Returns metrics dict + per-trial predictions.
    """
    trial_ids = np.asarray(trial_ids)
    y_prob = np.asarray(y_prob)
    trial_scores = {}
    for tid in np.unique(trial_ids):
        mask = trial_ids == tid
        trial_scores[tid] = float(np.max(y_prob[mask]))

    trues, preds = [], []
    for tid, score in trial_scores.items():
        trues.append(y_true_trial_label[tid])
        preds.append(int(score >= threshold))
    tp, tn, fp, fn = confusion_counts(trues, preds)
    m = metrics_from_counts(tp, tn, fp, fn)
    m["confusion"] = dict(tp=tp, tn=tn, fp=fp, fn=fn)
    m["trial_scores"] = trial_scores
    return m


def pre_impact_lead_time(alarm_times: dict, impact_times: dict):
    """
    Sec 4.4.1, Eq. (27): delta_t = t_impact - t_alarm, computed only for
    trials where an alarm fired (score crossed 0.5) before impact.
    alarm_times / impact_times: dict trial_id -> time in ms (or None if no
    alarm fired for that trial => miss).
    Returns dict with median/mean lead time, detection rate, and the
    compliance-rate breakdown used in Fig. 9.
    """
    lead_times = []
    n_total = len(impact_times)
    for tid, t_impact in impact_times.items():
        t_alarm = alarm_times.get(tid)
        if t_alarm is not None:
            lead_times.append(t_impact - t_alarm)
    lead_times = np.array(lead_times, dtype=float)
    n_detected = len(lead_times)
    out = dict(
        n_total=n_total, n_detected=n_detected,
        detection_rate=n_detected / max(n_total, 1),
        median_lead_ms=float(np.median(lead_times)) if n_detected else None,
        mean_lead_ms=float(np.mean(lead_times)) if n_detected else None,
    )
    for thresh in (100, 150, 200, 250):
        if n_detected:
            out[f"pct_ge_{thresh}ms"] = float(np.mean(lead_times >= thresh))
        else:
            out[f"pct_ge_{thresh}ms"] = None
    return out


def paired_significance_tests(f1_scores_a, f1_scores_b):
    """
    Sec 4.2: paired Wilcoxon signed-rank test (one-tailed) + paired t-test
    across the 5 per-fold F1 scores of two models, plus Cohen's d.
    """
    from scipy.stats import wilcoxon, ttest_rel
    a = np.asarray(f1_scores_a, dtype=float)
    b = np.asarray(f1_scores_b, dtype=float)
    diff = a - b
    try:
        w_stat, w_p = wilcoxon(a, b, alternative="greater")
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    t_stat, t_p_two_sided = ttest_rel(a, b)
    t_p_one_sided = t_p_two_sided / 2 if t_stat > 0 else 1 - t_p_two_sided / 2
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohens_d = float(diff.mean() / pooled_std) if pooled_std > 0 else float("nan")
    return dict(wilcoxon_stat=w_stat, wilcoxon_p=w_p,
                t_stat=t_stat, t_p_one_sided=t_p_one_sided, cohens_d=cohens_d)
