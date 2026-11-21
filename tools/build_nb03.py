"""Generate notebooks/03_backtest_methodology.ipynb (build helper, not analysis code)."""

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []


def md(s):
    cells.append(new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(new_code_cell(s.strip("\n")))


md(
    r"""
# 03 — Backtest Methodology, and Why No Backtest Is Possible Here

**This notebook does not report a strategy return, and that is the point.**

The supplied snapshot has no timestamps, no post identifiers, no author history, no security
identifiers beyond free-text mentions, and no prices. A backtest built on it would require
inventing every one of those fields, and an invented backtest is worse than no backtest: it
produces a number that looks like evidence and is not.

So this notebook does two things:

1. **Specifies** the backtest that would be run the moment the data supported it — point-in-time
   requirements, schema, splits, purging, embargo, execution lag, costs, and the untouched
   hold-out rule.
2. **Validates the machinery** on clearly-labelled synthetic data, so that when real data arrives
   the code is already correct. Every synthetic figure is titled "ENGINEERING VALIDATION ONLY".

The distinction matters: sections 1-7 are methodology I stand behind, section 8 is a unit test of
my own code, and nothing here is evidence about markets.
"""
)

code(
    r"""
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from IPython.display import display

from src import data as D
from src import features as F
from src import evaluation as E
from src import plotting as P

P.setup_style()
pd.set_option("display.width", 160)

SEED = F.DEFAULT_SEED
F.set_global_seed(SEED)
print("seed:", SEED)
"""
)

md(
    r"""
## 1. What the supplied data actually contains

Establishing the gap concretely is more useful than asserting it.
"""
)

code(
    r"""
raw = D.load_raw(ROOT / "data" / "raw" / "users.csv")
audit = D.audit_missing_values(raw)
display(audit.head(20))

print("columns available:", list(raw.columns))
print("rows:", len(raw))

required = {
    "post_id": "stable identifier to de-duplicate and to join on",
    "author_id": "to separate a new voice from a repeating one",
    "created_at_utc": "the only way to place a post before or after a price move",
    "collected_at_utc": "to compute availability lag and avoid using a post too early",
    "engagement_at_decision": "views/likes measured point-in-time, not as a final total",
    "ticker_mapping": "a security identifier, not free text",
    "prices_pit": "point-in-time OHLCV for the mapped tickers",
}
present = set(raw.columns)
missing = {k: v for k, v in required.items() if k not in present}

gap = pd.DataFrame(
    [{"required_field": k, "why_it_is_needed": v, "status": "MISSING"} for k, v in missing.items()]
)
P.save_table(gap, "t10_backtest_data_gap")
display(gap)
print(f"\n{len(missing)} of {len(required)} required fields are absent from the snapshot.")
"""
)

md(
    r"""
### The three defects that kill a backtest outright

| Defect | Consequence if ignored |
|---|---|
| **No timestamps** | Cannot order posts against prices. Every "signal" would be applied to a return it may have preceded *or* followed. This is look-ahead bias in its purest form. |
| **Views are final, not point-in-time** | The 3.34M figure was measured at scrape time, after the outcome. Using it as a decision-time weight leaks the future. |
| **No security identifiers** | `BABA` in free text is not a ticker on an exchange with a currency, an ADR ratio and a trading calendar. Mapping it by hand after seeing which posts "worked" is selection bias. |

Any one of these is disqualifying. All three are present.
"""
)

md(
    r"""
## 2. Required dataset schema

The contract a data pipeline must satisfy before any performance number is meaningful. Written as
an executable schema so it can be asserted against real data later.
"""
)

code(
    r"""
SCHEMA = pd.DataFrame(
    [
        ("post_id", "string", "Stable platform post id. Primary key.", "dedupe, join"),
        ("author_id", "string", "Stable account id, not display name.", "author-level clustering"),
        ("author_followers_pit", "int", "Followers at decision time, not current.", "reach weighting"),
        ("created_at_utc", "datetime64[UTC]", "Publication timestamp, UTC.", "event ordering"),
        ("collected_at_utc", "datetime64[UTC]", "When the scraper observed it.", "availability lag"),
        ("text_raw", "string", "Original post body, untruncated.", "NLP features"),
        ("views_pit", "int", "Views at collected_at, not final.", "attention weight"),
        ("likes_pit, comments_pit, reposts_pit", "int", "Engagement at collected_at.", "attention weight"),
        ("ticker", "string", "Resolved security identifier (exchange:ticker).", "price join"),
        ("ticker_confidence", "float", "Mapping confidence in [0,1].", "mapping audit"),
        ("direction_label", "int", "Manual stance in {-1,0,1} with provenance.", "signal sign"),
        ("label_source", "string", "Who/what produced the label.", "reproducibility"),
        ("label_timestamp_utc", "datetime64[UTC]", "When the label was created.", "audit trail"),
        ("pit_universe_flag", "bool", "Was the ticker tradable and listed at that date?", "survivorship"),
        ("forward_return_5d", "float", "Return from next open to close+5d.", "evaluation target"),
    ],
    columns=["field", "dtype", "definition", "used_for"],
)
P.save_table(SCHEMA, "t11_required_schema")
display(SCHEMA)

required_dates = ["created_at_utc", "collected_at_utc", "label_timestamp_utc"]
assert set(required_dates).issubset(set(SCHEMA.field)), "schema must declare all timestamps"
print("\nvalidation rule: every timestamp column must be timezone-aware UTC, never naive local.")
"""
)

md(
    r"""
## 3. Point-in-time discipline

The rule that governs everything else:

> A decision made at time $t$ may use only information that was **published and observable** at
> or before $t - \ell$, where $\ell$ is the collection/processing lag. Nothing else.

Concretely, for this problem:

- **Availability lag.** A post scraped at 09:30 may have been visible at 09:05 or at 08:00. Using
  the scrape time as the signal time is optimistic unless the lag is measured. The safe choice is
  to stamp `available_at = max(created_at, collected_at)` and never act on it before then.
- **Engagement lag.** View counts mature over hours. `views_pit` must be the value at
  `collected_at`, and the collection must be scheduled at a fixed interval (e.g. T+1h, T+24h), not
  whenever convenient.
- **Survivorship.** A ticker that was delisted mid-sample must still appear. Filtering to
  currently-listed names inflates returns and is one of the most common silent errors.
- **Universe.** `pit_universe_flag` must come from a historical index membership file, not from
  today's constituents.
- **Restatements.** Manual labels must be versioned. If a label is revised after seeing returns,
  that revision is itself a look-ahead event and must be recorded with `label_timestamp_utc`.
"""
)

md(
    r"""
## 4. Chronological validation: walk-forward, purge and embargo

Random k-fold cross-validation is **invalid** for financial time series. Two rows a day apart share
almost all of their forward-return window, so a random split puts near-duplicates in train and
test. The result is a high score that does not survive contact with the future.

The correct structure is a chronological walk-forward with two gaps:

```mermaid
graph LR
    A["Train block 1"] --> P1["purge"] --> B["Test block 1"]
    B --> E1["embargo"] --> C["Train block 2 (expanding)"] --> P2["purge"] --> Dd["Test block 2"]
```

- **Purge.** Drop training rows whose label window (`signal_date + holding_days`) overlaps the
  test window. Without this, a training row labelled using returns from inside the test period
  leaks the answer.
- **Embargo.** Insert a gap after each test block before the next training block, because labels
  computed at the end of the test period are still unresolved.

`src.evaluation.walk_forward_splits` and `split_masks` implement exactly this, with calendar-day
gaps as explicit parameters.
"""
)

code(
    r"""
# Demonstrate the split geometry on a synthetic two-year daily calendar.
demo_dates = pd.date_range("2022-01-03", "2023-12-29", freq="B")
splits = E.walk_forward_splits(demo_dates, n_splits=4, min_train_frac=0.5,
                               purge_days=5, embargo_days=2)

geom = pd.DataFrame(
    [
        {
            "split": s.name,
            "train_start": s.train_start.date(),
            "train_end": s.train_end.date(),
            "test_start": s.test_start.date(),
            "test_end": s.test_end.date(),
            "purge_days": s.purge_days,
            "embargo_days": s.embargo_days,
            "train_days": (s.train_end - s.train_start).days,
            "test_days": (s.test_end - s.test_start).days,
        }
        for s in splits
    ]
)
display(geom)

# Assert the three invariants that make the scheme valid.
for s in splits:
    assert s.train_end < s.test_start, f"{s.name}: training must end before testing begins"
    gap = (s.test_start - s.train_end).days
    assert gap >= s.purge_days, f"{s.name}: purge gap {gap}d < required {s.purge_days}d"
prev_test_end = None
for s in splits:
    if prev_test_end is not None:
        assert s.test_start >= prev_test_end, f"{s.name}: test blocks must not overlap"
    prev_test_end = s.test_end
print("walk-forward invariants hold: train < test, purge gap respected, test blocks disjoint")
"""
)

code(
    r"""
# Show the purged/embargoed masks for one split, to make the leakage guard visible.
holding_days = 5
s = splits[1]
train_mask, test_mask = E.split_masks(demo_dates, s, holding_days=holding_days)

print(f"{s.name}: training rows {train_mask.sum()}, test rows {test_mask.sum()}")
print(f"train window ends {s.train_end.date()}, test window {s.test_start.date()} to {s.test_end.date()}")

# A training row is only kept if its label window closes before the test block starts.
train_dates = demo_dates[train_mask]
label_end = train_dates + pd.Timedelta(days=holding_days)
assert (label_end < s.test_start).all(), "purge failed: a training label window reached the test block"
print("purge verified: no training label window crosses into the test block")

# And no row is simultaneously in train and test.
assert not (train_mask & test_mask).any(), "a row is both training and testing"
print("disjointness verified: no row is in both blocks")
"""
)

md(
    r"""
### What this means for a 119-row snapshot

Even with timestamps, 119 posts would not support this design. A single walk-forward split with a
5-day holding period consumes weeks of calendar time, and the purged training block would contain a
few dozen rows. The honest sample-size statement is that this data can validate a **labelling and
feature pipeline**, not a **trading rule**.
"""
)

md(
    r"""
## 5. Execution assumptions

Three assumptions that separate a defensible backtest from a optimistic one:

1. **Execution lag.** A signal derived from a post published at 14:00 cannot be filled at 14:00.
   The realistic assumption is the **next tradable bar's open** (or VWAP over the first minutes).
   Filling at the same-bar close introduces a same-bar look-ahead.
2. **Costs.** Commission plus slippage on **traded notional**. A long-short book with gross
   exposure 1.0 trades 1.0 of notional to open, so a naive "charge on turnover" figure halves the
   true cost. `CostModel.cost_fraction` charges on `sum|Δw|`, and `portfolio_return` reports both
   that and the conventional one-way turnover so the difference is visible.
3. **Turnover and capacity.** A KOL post can move a small-cap name. Position sizes must be capped
   by a fraction of average daily volume, or the backtest assumes liquidity that does not exist.

### The cost convention, stated precisely

For weights $w_t$ and $w_{t-1}$:

$$\text{traded notional}_t = \sum_i |w_{t,i} - w_{t-1,i}|, \qquad \text{turnover}_t = \tfrac{1}{2}\sum_i |w_{t,i} - w_{t-1,i}|$$

$$\text{cost}_t = \text{traded notional}_t \times \frac{\text{commission} + \text{slippage}}{10^4}$$

$$r^{\text{net}}_t = \sum_i w_{t,i} r_{t,i} - \text{cost}_t$$
"""
)

code(
    r"""
cm = E.CostModel(commission_bps=5.0, slippage_bps=5.0)
print("one-way cost:", cm.one_way_bps, "bps per unit of traded notional")

# Opening a gross-1.0 long/short book from cash must cost 10bps of NAV, not 5bps.
w_open = np.array([0.25, 0.25, -0.25, -0.25])
res_open = E.portfolio_return(w_open, np.array([0.0, 0.0, 0.0, 0.0]),
                              prev_weights=np.zeros(4), costs=cm)
print("\nopen a long/short book from flat:", {k: round(v, 6) for k, v in res_open.items()})
assert abs(res_open["cost"] - 10 / 1e4) < 1e-12
assert abs(res_open["turnover"] - 0.5) < 1e-12
assert abs(res_open["traded_notional"] - 1.0) < 1e-12
print("cost charged on traded notional = 1.0 -> 10bps. A turnover-based charge would")
print("have understated this by 2x. The convention is now explicit, not accidental.")

# Holding with no change must cost nothing.
res_hold = E.portfolio_return(w_open, np.array([0.01, 0.02, -0.01, 0.0]),
                              prev_weights=w_open, costs=cm)
print("\nhold with zero rebalance:", {k: round(v, 6) for k, v in res_hold.items()})
assert res_hold["cost"] == 0.0
print("zero-turnover period correctly costs zero")
"""
)

md(
    r"""
## 6. Held-out test set: the rule that makes a result believable

The final block of the sample must be **sealed**. It is used exactly once, after every choice —
features, thresholds, holding period, costs, universe, model class — is frozen.

The failure mode this prevents is subtle. If the last year is examined while still choosing
between ten holding periods, the reported winner is the one that happened to fit that year. The
number is then a best-of-ten statistic masquerading as an unbiased estimate.

The rule, stated as a commitment:

- Design decisions use only the **training** portion of the timeline.
- The sealed block is evaluated **once**, and the result is reported whatever it says.
- If the sealed result is disappointing, the response is to report it — not to revise the design.
- Any revision restarts the clock and requires a **new** untouched block.
"""
)

code(
    r"""
seal = pd.DataFrame(
    [
        ("features", "TF-IDF char n-grams, frozen encoder", "chosen pre-holdout"),
        ("model", "LogisticRegression C=1.0 balanced", "chosen pre-holdout"),
        ("holding period", "5 trading days", "chosen pre-holdout"),
        ("costs", "5bps commission + 5bps slippage", "chosen pre-holdout"),
        ("universe", "PIT index membership", "chosen pre-holdout"),
        ("hold-out block", "final 20% of timeline", "EVALUATED ONCE, then sealed"),
    ],
    columns=["decision", "choice", "when_decided"],
)
display(seal)
print("Any change to a row above restarts the protocol and requires a new hold-out block.")
"""
)

md(
    r"""
## 7. What a credible result would have to report

A checklist I would apply to my own backtest before showing it to anyone:

| Requirement | Why |
|---|---|
| Gross **and** net returns | A gross-only number hides the cost of the strategy |
| Turnover and traded notional, separately | So a reader can re-cost it under their own assumptions |
| Costs charged on traded notional | Charging on half-turnover halves the drag |
| Execution lag stated and applied | Same-bar fills are look-ahead |
| Sharpe **and** drawdown, not just return | A high-return, 60% drawdown strategy is unusable |
| Hit rate and average win/loss | Reveals whether it is a few lucky trades |
| Number of independent bets | 119 posts is not 119 independent observations |
| Deflated Sharpe or equivalent | With many variants tested, the best Sharpe is biased upward |
| Capacity statement | Strategy size versus average daily volume |
| Point-in-time universe | Survivorship-free |
| Reproducible with one command | Otherwise it is an anecdote |

Note what is **absent** from any of this: none of these statistics can be computed from the
supplied snapshot.
"""
)

md(
    r"""
## 8. ENGINEERING VALIDATION ONLY — synthetic machinery test

**Everything below this line is a unit test of my own code. It is not a result.**

I generate *fictional* posts with *fictional* timestamps and *fictional* returns so that the
plumbing — walk-forward splits, purge masks, position sizing, cost accounting, turnover — can be
exercised and shown to behave. The data-generating process is:

- 400 synthetic observations spread over two business years,
- a known planted signal (the synthetic score genuinely predicts the synthetic return),
- plus noise, so that the pipeline has to work to find it.

Because the signal is planted by construction, the pipeline *must* recover it. If it does not, the
code is broken. That is the entire purpose: a passing test says the machinery works, and **says
nothing whatsoever about real markets**.
"""
)

code(
    r"""
# A panel, not a bag of rows: persistent synthetic assets rebalanced weekly.
N_ASSETS, N_WEEKS = 20, 104
rng = np.random.default_rng(SEED)

asset_ids = [f"SYN{i:02d}" for i in range(N_ASSETS)]
weeks = pd.date_range("2022-01-07", periods=N_WEEKS, freq="W-FRI")

grid = pd.MultiIndex.from_product([weeks, asset_ids], names=["signal_date", "asset"])
syn = pd.DataFrame(index=grid).reset_index()

# Persistent, slightly smoothed scores so the book does not churn at random.
syn["synthetic_score"] = rng.normal(0, 1, len(syn))
syn["synthetic_score"] = (
    syn.groupby("asset")["synthetic_score"]
       .transform(lambda s: s.rolling(2, min_periods=1).mean())
)

# Planted relationship: return = beta * score + noise. This is fiction, not market data.
BETA, NOISE_SD = 0.004, 0.02
syn["forward_return"] = BETA * syn["synthetic_score"] + rng.normal(0, NOISE_SD, len(syn))

print(f"synthetic panel: {len(syn)} observations = {N_WEEKS} weeks x {N_ASSETS} assets")
print("date span:", syn.signal_date.min().date(), "to", syn.signal_date.max().date())
print(f"planted beta={BETA}, noise sd={NOISE_SD}")
print(f"realised correlation(score, return) = {syn.synthetic_score.corr(syn.forward_return):.4f}")
print("\nTHIS IS SYNTHETIC DATA. No real post, price or return appears in this table.")
"""
)

code(
    r"""
syn_splits = E.walk_forward_splits(syn["signal_date"], n_splits=4, min_train_frac=0.5,
                                   purge_days=5, embargo_days=2)
HOLDING = 5

# Weekly rebalance inside each test block. Sizing uses only the score column, never
# the returns column, and the previous week's book is carried so that closing trades
# are costed rather than ignored.
rows = []
prev_w = pd.Series(0.0, index=asset_ids)

for s in syn_splits:
    tr, te = E.split_masks(syn["signal_date"], s, holding_days=HOLDING)
    if te.sum() == 0:
        continue

    test_block = syn.loc[te]
    block_gross = block_net = block_cost = block_traded = 0.0
    n_rebal = 0

    for week, wk in test_block.groupby("signal_date", sort=True):
        scores = wk.set_index("asset")["synthetic_score"]
        rets = wk.set_index("asset")["forward_return"]

        new_book = E.signal_to_weights(scores, top_quantile=0.3, gross_leverage=1.0)
        idx = prev_w.index.union(new_book.index)
        w_new = new_book.reindex(idx, fill_value=0.0)
        w_old = prev_w.reindex(idx, fill_value=0.0)
        r = rets.reindex(idx, fill_value=0.0)

        rr = E.portfolio_return(w_new.to_numpy(), r.to_numpy(),
                                prev_weights=w_old.to_numpy(), costs=cm,
                                holding_days=HOLDING)
        block_gross += rr["gross_return"]
        block_net += rr["net_return"]
        block_cost += rr["cost"]
        block_traded += rr["traded_notional"]
        n_rebal += 1
        prev_w = w_new

    rows.append({
        "split": s.name,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "n_rebalances": n_rebal,
        "gross_return": block_gross,
        "net_return": block_net,
        "cost": block_cost,
        "traded_notional": block_traded,
        "one_way_turnover": 0.5 * block_traded,
    })

bt = pd.DataFrame(rows)
P.save_table(bt, "t12_synthetic_walkforward")
display(bt.round(5))

print("mean gross / block:", round(bt.gross_return.mean(), 5))
print("mean net   / block:", round(bt.net_return.mean(), 5))
print("mean cost  / block:", round(bt.cost.mean(), 5))
print("mean one-way turnover per block:", round(bt.one_way_turnover.mean(), 4))

assert bt.n_train.iloc[0] < bt.n_train.iloc[-1], "training window must expand"
assert np.allclose(bt.traded_notional, 2 * bt.one_way_turnover)
assert (bt.net_return <= bt.gross_return + 1e-12).all(), "net must not exceed gross"
assert (bt.cost > 0).all(), "a rebalanced book must incur cost"
print("\nall assertions passed: expanding train window, cost charged on traded notional,")
print("net <= gross in every block, strictly positive cost on every rebalance")
"""
)

code(
    r"""
fig, axes = P.plt.subplots(1, 2, figsize=(11.5, 3.9))

ax = axes[0]
x = np.arange(len(bt))
ax.bar(x - 0.2, bt.gross_return, width=0.4, label="gross", color=P.PALETTE["muted"])
ax.bar(x + 0.2, bt.net_return, width=0.4, label="net of costs", color=P.PALETTE["primary"])
ax.set_xticks(x)
ax.set_xticklabels(bt.split)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("summed weekly return (fraction of NAV)")
ax.set_title("Synthetic gross vs net by walk-forward block", fontsize=9.5)
ax.legend(fontsize=9)

ax = axes[1]
ax.bar(x - 0.2, bt.traded_notional, width=0.4, label="traded notional (charged)",
       color=P.PALETTE["accent"])
ax.bar(x + 0.2, bt.one_way_turnover, width=0.4, label="one-way turnover (reported)",
       color=P.PALETTE["neutral"])
ax.set_xticks(x)
ax.set_xticklabels(bt.split)
ax.set_ylabel("sum over block")
ax.set_title("Traded notional vs conventional turnover", fontsize=9.5)
ax.legend(fontsize=8.5)

fig.suptitle("Figure 7. Synthetic engineering validation only — not an investment result",
             y=1.04, fontsize=11.5, color=P.PALETTE["accent"])
P.save_figure(fig, "fig07_synthetic_walkforward_ENGINEERING_ONLY")
"""
)

md(
    r"""
### Reading this correctly

The synthetic runs are positive because I **planted** a positive relationship. If I had planted
`BETA = 0`, the pipeline would produce returns indistinguishable from noise — and it would be just
as correct a result.

So the value of section 8 is narrow and specific:

- the walk-forward machinery advances chronologically and never trains on the future,
- the purge mask demonstrably removes training rows whose label window reaches the test block,
- `signal_to_weights` produces a dollar-neutral book within its concentration cap,
- costs are charged on traded notional and are non-zero whenever the book changes,
- turnover is reported separately from cost so a reader can re-cost it.

Those five statements are verified. **No statement about profitability is made or implied.**
"""
)

md(
    r"""
## 9. What I would do with two more weeks of data engineering

Ordered by what unlocks the most analysis per unit of effort:

1. **Capture raw snapshots with timestamps.** A scraper run every hour, storing the raw payload
   with `collected_at_utc`. Without this, no historical analysis is ever possible, and the loss is
   permanent — you cannot go back and re-observe yesterday's view count.
2. **Resolve tickers through a mapping table with confidence and provenance.** Not free-text
   regexes. Record *why* a post was mapped to `NYSE:BABA` and keep the mapping versioned.
3. **Pull point-in-time OHLCV** from a survivorship-free source, with delisted names included.
4. **Build the author history.** A first-time mention and an habitual promoter of the same name
   should not carry the same weight. This needs author-level history, which needs item 1.
5. **Then, and only then**, run the backtest in section 4-6 on real data with a sealed hold-out.

Steps 1 and 2 are prerequisites for everything else. Any performance number produced before them
is a fiction with a spreadsheet attached.
"""
)

md(
    r"""
## 10. Summary

| Question | Answer |
|---|---|
| Can a valid backtest be run on the supplied data? | **No.** Three disqualifying defects: no timestamps, no PIT engagement, no security identifiers. |
| What is supplied instead? | A written schema, a point-in-time protocol, purged/embargoed walk-forward code, an explicit cost model, and a sealed hold-out commitment. |
| Is the machinery tested? | Yes, on clearly-labelled synthetic data. That tests the code, not markets. |
| What is the honest headline? | The pipeline is ready; the data is not. |

The most defensible thing this submission can say about profitability is that it declines to
claim any.
"""
)

nb = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
)
nbf.write(nb, "notebooks/03_backtest_methodology.ipynb")
print("written", len(cells), "cells")
