"""Investment-signal construction from labelled posts.

Row-level signal::

    q_i = r_i * d_i * c_i * n_i * a_i
    a_i = clip(1 + alpha * z(log1p(views_i)), 0.5, 1.5)

Theme-level aggregate::

    S_a = sum_i q_i / (sum_i r_i + eps)

Both formulas are deliberately simple and fully documented. The attention
weight is capped so that views cannot dominate, and the denominator uses
relevance rather than post count so that off-topic posts do not dilute a theme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import capped_attention

DEFAULT_ALPHA = 0.20
DEFAULT_EPS = 1e-6

#: Component definitions used by the ablation. ``None`` means "drop it".
ABLATION_VARIANTS: dict[str, dict[str, object]] = {
    "S0_relevance_attention": {"direction": False, "confidence": False, "novelty": False, "attention": True},
    "S1_direction_only": {"direction": True, "confidence": False, "novelty": False, "attention": False},
    "S2_direction_confidence": {"direction": True, "confidence": True, "novelty": False, "attention": False},
    "S3_plus_attention": {"direction": True, "confidence": True, "novelty": False, "attention": True},
    "S4_plus_novelty": {"direction": True, "confidence": True, "novelty": True, "attention": True},
}


def validate_labels(df: pd.DataFrame) -> None:
    """Validate the label columns used by the signal.

    Raises
    ------
    ValueError
        With a precise message listing offending rows when a constraint fails.
    """
    required = ["tradable_relevance", "direction", "confidence", "novelty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label column(s): {missing}")

    if not df["direction"].isin([-1, 0, 1]).all():
        bad = df.loc[~df["direction"].isin([-1, 0, 1]), "row_id"].tolist()
        raise ValueError(f"direction must be in {{-1,0,1}}; offending row_id: {bad}")

    for col in ("confidence", "novelty", "tradable_relevance"):
        series = df[col].astype(float)
        if series.isna().any():
            bad = df.loc[series.isna(), "row_id"].tolist()
            raise ValueError(f"{col} has missing values at row_id {bad}")
        if ((series < 0) | (series > 1)).any():
            bad = df.loc[(series < 0) | (series > 1), "row_id"].tolist()
            raise ValueError(f"{col} must lie in [0,1]; offending row_id: {bad}")


def compute_row_signals(
    df: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    direction_on: bool = True,
    confidence_on: bool = True,
    novelty_on: bool = True,
    attention_on: bool = True,
    relevance_as_indicator: bool = True,
) -> pd.DataFrame:
    """Compute the row-level signal ``q_i`` and its multiplicative components.

    Parameters
    ----------
    df
        Labelled frame containing ``tradable_relevance``, ``direction``,
        ``confidence``, ``novelty`` and ``views_numeric``.
    alpha
        Slope of the attention transformation. Fixed at ``0.20`` and **not**
        tuned against any evaluation result.
    direction_on, confidence_on, novelty_on, attention_on
        Component switches used by the ablation study.
    relevance_as_indicator
        If ``True``, use ``r`` as a [0,1] weight; if ``False``, use a binary
        0/1 indicator. The weight view is the primary specification.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with added columns ``r``, ``d``, ``c``, ``n``, ``a``,
        and ``q``.
    """
    validate_labels(df)
    out = df.copy()

    out["r"] = out["tradable_relevance"].astype(float)
    if not relevance_as_indicator:
        out["r"] = (out["r"] > 0.5).astype(float)

    out["d"] = out["direction"].astype(float) if direction_on else 1.0
    out["c"] = out["confidence"].astype(float) if confidence_on else 1.0
    out["n"] = out["novelty"].astype(float) if novelty_on else 1.0
    out["a"] = (
        capped_attention(out["views_numeric"], alpha=alpha) if attention_on else 1.0
    )

    out["q"] = out["r"] * out["d"] * out["c"] * out["n"] * out["a"]
    return out


def aggregate_by_theme(df: pd.DataFrame, signal_col: str = "q", theme_col: str = "asset_or_theme") -> pd.DataFrame:
    """Aggregate row signals into theme scores ``S_a``.

    ``S_a = sum(q_i) / (sum(r_i) + eps)``. Themes with no relevant posts return
    ``NaN`` rather than a spurious zero, because zero would be indistinguishable
    from a genuinely neutral reading. ``NONE`` themes are excluded.
    """
    if theme_col not in df.columns:
        raise ValueError(f"Missing theme column: {theme_col}")

    work = df[df[theme_col].astype(str).str.upper() != "NONE"].copy()
    rel = work["tradable_relevance"].astype(float) if "tradable_relevance" in work else 1.0

    grouped = work.assign(_rel=rel).groupby(theme_col, dropna=False)
    rows = []
    for theme, g in grouped:
        rel_sum = float(g["_rel"].sum())
        q_sum = float(g[signal_col].sum())
        n_rel = int((g["_rel"] > 0).sum())
        rows.append(
            {
                theme_col: theme,
                "n_posts": int(len(g)),
                "n_relevant_posts": n_rel,
                "relevance_mass": rel_sum,
                "signal_sum": q_sum,
                "theme_score": q_sum / (rel_sum + DEFAULT_EPS) if rel_sum > 0 else np.nan,
                "mean_views": float(g["views_numeric"].mean()),
            }
        )

    cols = [theme_col, "n_posts", "n_relevant_posts", "relevance_mass", "signal_sum", "theme_score", "mean_views"]
    result = pd.DataFrame(rows, columns=cols)
    return result.sort_values("theme_score", ascending=False, na_position="last").reset_index(drop=True)


def ablation_table(df: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Compare the documented signal variants on the frozen snapshot.

    Returns descriptive statistics only. There is no return series, therefore
    no Sharpe, no information coefficient and no performance claim.
    """
    validate_labels(df)
    rows = []
    for name, cfg in ABLATION_VARIANTS.items():
        scored = compute_row_signals(
            df,
            alpha=alpha,
            direction_on=bool(cfg["direction"]),
            confidence_on=bool(cfg["confidence"]),
            novelty_on=bool(cfg["novelty"]),
            attention_on=bool(cfg["attention"]),
        )
        themes = aggregate_by_theme(scored)
        relevant = scored[scored["tradable_relevance"] > 0.5]
        rows.append(
            {
                "variant": name,
                "direction": cfg["direction"],
                "confidence": cfg["confidence"],
                "novelty": cfg["novelty"],
                "attention": cfg["attention"],
                "n_relevant_posts": int(len(relevant)),
                "nonzero_relevant_signals": int((relevant["q"].abs() > 1e-9).sum()),
                "mean_row_signal": float(relevant["q"].mean()) if len(relevant) else np.nan,
                "std_row_signal": float(relevant["q"].std()) if len(relevant) else np.nan,
                "n_themes_covered": int(themes[theme_score_col(themes)].notna().sum()),
                "theme_dispersion": float(themes[theme_score_col(themes)].std()),
            }
        )
    return pd.DataFrame(rows)


def theme_score_col(df: pd.DataFrame) -> str:
    """Return the theme-score column name, tolerating both spellings."""
    return "theme_score" if "theme_score" in df.columns else df.columns[3]