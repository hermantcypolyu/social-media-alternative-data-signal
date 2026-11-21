"""Model evaluation, fold-level reporting and leakage-safe backtest scaffolding.

Two clearly separated halves:

1. **Snapshot NLP evaluation** - stratified cross-validation over the 119-row
   label set. This measures label prediction, *not* financial alpha.
2. **Backtest scaffolding** - chronological walk-forward splits with purging and
   embargo, plus a transaction-cost model. It is used only for engineering
   validation on synthetic data; it makes no economic claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

DEFAULT_SEED = 42


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #


def make_majority_baseline(strategy: str = "most_frequent") -> DummyClassifier:
    """B0: majority-class (or stratified) baseline.

    This exists to show how little a classifier must do on an imbalanced label
    set. Any model that does not clearly beat it is not adding value.
    """
    return DummyClassifier(strategy=strategy, random_state=DEFAULT_SEED)


def make_logistic_regression(C: float = 1.0, class_weight: str | None = "balanced") -> LogisticRegression:
    """Regularised logistic regression with class weighting for imbalance."""
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=5000,
        solver="lbfgs",
        random_state=DEFAULT_SEED,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def classification_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    """Class-imbalance-aware metrics for a single fold or a pooled prediction.

    Accuracy is reported for completeness but is never used as evidence of
    investment value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "n_samples": int(len(y_true)),
        "n_classes_true": int(len(np.unique(y_true))),
        "n_classes_pred": int(len(labels)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
    }


def per_class_report(y_true: Sequence, y_pred: Sequence) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support table."""
    labels = sorted(pd.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "class": labels,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": s.astype(int),
        }
    )


def confusion_frame(y_true: Sequence, y_pred: Sequence) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows = true, cols = predicted)."""
    labels = sorted(pd.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                        columns=[f"pred_{l}" for l in labels])


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric: Callable[[Sequence, Sequence], float] = lambda a, b: f1_score(a, b, average="macro", zero_division=0),
    n_boot: int = 1000,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a single metric.

    The interval reflects sampling noise in 119 rows, which is large. A wide
    interval is itself an honest finding, not a defect to be tuned away.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (np.nan, np.nan)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        values[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Cross-validation with fold-internal fitting
# --------------------------------------------------------------------------- #


@dataclass
class CVResult:
    """Fold-level cross-validation output and the pooled out-of-fold predictions."""

    model_name: str
    fold_metrics: pd.DataFrame
    oof_true: np.ndarray
    oof_pred: np.ndarray
    fold_assignments: np.ndarray
    summary: dict = field(default_factory=dict)
    oof_repeat_counts: np.ndarray | None = None

    @property
    def mean_metrics(self) -> dict[str, float]:
        numeric = self.fold_metrics.select_dtypes(include=[np.number])
        return {c: float(numeric[c].mean()) for c in numeric.columns}

    @property
    def std_metrics(self) -> dict[str, float]:
        numeric = self.fold_metrics.select_dtypes(include=[np.number])
        return {c: float(numeric[c].std(ddof=0)) for c in numeric.columns}


def cross_validate_classifier(
    X: np.ndarray,
    y: Sequence,
    estimator: BaseEstimator,
    model_name: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = DEFAULT_SEED,
) -> CVResult:
    """Repeated stratified CV where the estimator (including any vectoriser) is
    refit inside every fold.

    With 21 positives, a single 5-fold split gives roughly four positives per test
    fold, so one split would be dominated by which rows happened to land where.
    Repeated stratified CV is therefore used and the dispersion *across repeats*
    is reported as the honest uncertainty measure.

    ``estimator`` must be an unfitted sklearn estimator or ``Pipeline``. It is
    cloned per fold, so any vectoriser or scaler inside the pipeline is fitted only
    on that fold's training rows. This is asserted by
    :func:`assert_no_index_overlap` on every fold.

    Notes
    -----
    Pooled out-of-fold predictions collapse repeats by majority vote per row, so
    the pooled confusion matrix is stable enough to read. Fold-level metrics are
    the primary variance estimate.
    """
    X = np.asarray(X) if not hasattr(X, "iloc") else X
    y = np.asarray(y)

    min_class = int(pd.Series(y).value_counts().min())
    if min_class < n_splits:
        n_splits = max(2, min_class)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    fold_rows: list[dict] = []
    vote_counts: list[dict] = [dict() for _ in range(len(y))]
    fold_ids = np.full(len(y), -1, dtype=int)

    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        assert_no_index_overlap(tr, te, fold=fold)
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        pred = est.predict(X[te])
        for i, p in zip(te, pred):
            vote_counts[int(i)][p] = vote_counts[int(i)].get(p, 0) + 1
        fold_ids[te] = fold % n_splits
        row = {
            "fold": fold,
            "repeat": fold // n_splits,
            "n_train": len(tr),
            "n_test": len(te),
        }
        row.update(classification_metrics(y[te], pred))
        fold_rows.append(row)

    oof_pred = np.array(
        [max(v.items(), key=lambda kv: kv[1])[0] if v else y[i] for i, v in enumerate(vote_counts)],
        dtype=y.dtype,
    )
    fold_metrics = pd.DataFrame(fold_rows)
    return CVResult(
        model_name=model_name,
        fold_metrics=fold_metrics,
        oof_true=y,
        oof_pred=oof_pred,
        fold_assignments=fold_ids,
        summary={"n_splits": n_splits, "n_repeats": n_repeats, "seed": seed},
        oof_repeat_counts=np.array([sum(v.values()) for v in vote_counts], dtype=int),
    )


def assert_no_index_overlap(train_idx: Iterable[int], test_idx: Iterable[int], fold: int | None = None) -> None:
    """Guard against accidental train/test index leakage in a fold."""
    overlap = set(np.asarray(train_idx).ravel()) & set(np.asarray(test_idx).ravel())
    if overlap:
        where = f" (fold {fold})" if fold is not None else ""
        raise AssertionError(
            f"Train/test index overlap{where}: {len(overlap)} shared rows. "
            "This would invalidate the cross-validation result."
        )


def build_pipeline(vectorizer: BaseEstimator, scaler: BaseEstimator | None, classifier: BaseEstimator) -> Pipeline:
    """Assemble ``vectorizer -> [scaler] -> classifier`` as one fitted object.

    Putting preprocessing inside the pipeline is the mechanism that guarantees
    the vectoriser and scaler never see evaluation rows.
    """
    steps: list[tuple[str, BaseEstimator]] = [("vectorizer", vectorizer)]
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Backtest scaffolding (methodology, not performance)
# --------------------------------------------------------------------------- #


@dataclass
class Split:
    """A single walk-forward split with purge and embargo metadata."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_days: int
    embargo_days: int


def walk_forward_splits(
    dates: Sequence[pd.Timestamp],
    n_splits: int = 4,
    min_train_frac: float = 0.5,
    purge_days: int = 5,
    embargo_days: int = 2,
) -> list[Split]:
    """Chronological walk-forward splits with purging and an embargo gap.

    Train windows are expanding. For each split:

    * ``training`` uses observations strictly before ``purge_start``;
    * ``purge_days`` are dropped between train and test so that overlapping
      forward-return labels cannot leak into training;
    * ``embargo_days`` are dropped after the test block before the next
      training window, so that overlapping labels cannot leak forward.

    Dates are assumed to be the **signal date**, and execution is assumed to
    happen at the next tradable bar. Returns are never used here.

    Parameters
    ----------
    dates
        Observation signal dates (unsorted is fine).
    n_splits
        Number of test blocks.
    min_train_frac
        Share of the timeline reserved for the first training block.
    purge_days, embargo_days
        Gap lengths in calendar days.

    Returns
    -------
    list[Split]
        Splits in chronological order.
    """
    d = pd.to_datetime(pd.Series(dates)).dropna().sort_values().reset_index(drop=True)
    if d.empty:
        raise ValueError("dates must contain at least one valid timestamp")

    start, end = d.iloc[0], d.iloc[-1]
    span_days = (end - start).days
    if span_days <= 0:
        raise ValueError("dates must span more than one day to build walk-forward splits")

    first_train_end_frac = min_train_frac
    block = (1.0 - first_train_end_frac) / n_splits

    splits: list[Split] = []
    for k in range(n_splits):
        test_frac_start = first_train_end_frac + k * block
        test_frac_end = test_frac_start + block
        train_end = start + pd.Timedelta(days=span_days * test_frac_start) - pd.Timedelta(days=purge_days)
        test_start = start + pd.Timedelta(days=span_days * test_frac_start)
        test_end = start + pd.Timedelta(days=span_days * test_frac_end)
        splits.append(
            Split(
                name=f"WF{k + 1}",
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_days=purge_days,
                embargo_days=embargo_days,
            )
        )
    return splits


def split_masks(dates: Sequence[pd.Timestamp], split: Split, holding_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean train/test masks honouring purge and embargo.

    A training observation is dropped when its label window
    (``signal_date + holding_days``) would cross into the test block. This is
    the purge. Test observations are limited to the declared test window.
    """
    d = pd.to_datetime(pd.Series(dates))
    label_end = d + pd.Timedelta(days=holding_days)
    embargo_start = split.test_end + pd.Timedelta(days=split.embargo_days)

    train = (d >= split.train_start) & (label_end < split.test_start)
    test = (d >= split.test_start) & (d < split.test_end) & (label_end <= embargo_start)
    return train.to_numpy(), test.to_numpy()


@dataclass
class CostModel:
    """Explicit transaction-cost assumptions.

    Costs are charged on **traded notional**, not on a half-rotated turnover
    figure. Opening a long-short book with gross exposure 1.0 trades 1.0 of
    notional (0.5 bought, 0.5 sold short), so it must cost
    ``1.0 x one_way_bps``. Charging on ``0.5 * sum|dw|`` instead would halve
    every cost and flatter any strategy, so the two quantities are reported
    separately: ``traded_notional`` is charged, ``turnover`` is the
    industry-standard one-way figure.
    """

    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    borrow_bps_per_day: float = 0.0

    @property
    def one_way_bps(self) -> float:
        """Cost in basis points per unit of traded notional."""
        return self.commission_bps + self.slippage_bps

    def cost_fraction(self, traded_notional: float, holding_days: int = 1) -> float:
        """Cost as a fraction of portfolio value for a given traded notional."""
        if traded_notional < 0:
            raise ValueError("traded notional must be non-negative")
        cost = traded_notional * self.one_way_bps / 1e4
        if self.borrow_bps_per_day:
            cost += traded_notional * self.borrow_bps_per_day / 1e4 * holding_days
        return float(cost)


def portfolio_return(
    weights: np.ndarray,
    forward_returns: np.ndarray,
    prev_weights: np.ndarray | None = None,
    costs: CostModel | None = None,
    holding_days: int = 1,
) -> dict[str, float]:
    """Gross and net return for one rebalance period.

    ``traded_notional`` is ``sum(|w_t - w_{t-1}|)``; this is what costs are
    charged on. ``turnover`` is reported as the standard one-way figure
    ``0.5 * sum(|w_t - w_{t-1}|)`` so it is comparable with published numbers.
    A fully-invested long-only book rebalances at ``turnover = 1.0`` and
    ``traded_notional = 2.0``; that difference is real money and is not hidden.
    """
    w = np.asarray(weights, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    if w.shape != r.shape:
        raise ValueError("weights and forward_returns must have the same shape")
    prev = np.zeros_like(w) if prev_weights is None else np.asarray(prev_weights, dtype=float)

    gross = float(np.nansum(w * r))
    traded = float(np.nansum(np.abs(w - prev)))
    turnover = 0.5 * traded
    cost = costs.cost_fraction(traded, holding_days) if costs else 0.0
    return {
        "gross_return": gross,
        "net_return": gross - cost,
        "turnover": turnover,
        "traded_notional": traded,
        "cost": cost,
    }


def signal_to_weights(scores: pd.Series, top_quantile: float = 0.2, gross_leverage: float = 1.0) -> pd.Series:
    """Long top quantile / short bottom quantile, equal-weighted, capped.

    Only assets with a finite non-zero score are eligible. A single asset can
    never exceed ``gross_leverage / 2`` per side, which caps concentration.
    """
    s = pd.Series(scores).astype(float)
    s = s[np.isfinite(s)]
    if s.empty:
        return pd.Series(dtype=float)

    n = len(s)
    k = max(1, int(np.floor(n * top_quantile)))
    ranked = s.sort_values()
    shorts = ranked.index[:k]
    longs = ranked.index[-k:]

    w = pd.Series(0.0, index=s.index)
    cap = gross_leverage / 2.0 / k if k else 0.0
    per_name = min(cap, gross_leverage / 2.0 / k)
    w[longs] += per_name
    w[shorts] -= per_name
    return w