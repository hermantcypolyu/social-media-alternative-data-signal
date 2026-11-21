# PROGRESS — work log, corrections, and what failed

A chronological record of what was built, what was wrong and got fixed, and what
did not work. Written so that a reviewer can see the reasoning trail, not just the
polished result. The report itself is [`solution.md`](solution.md); the checklist
verdict is in [`FINAL_AUDIT.md`](FINAL_AUDIT.md).

---

## 1. Scope actually delivered

Two working days. The instruction was to build a defensible research prototype,
not a product. What exists:

| Layer | Deliverable | Status |
|---|---|---|
| Data | `users.csv` audited, cleaned, deduplicated, never modified | done |
| Labels | Written 7-stage rubric (`ANNOTATION_GUIDE.md`), 119 rows labelled | done |
| Baseline | Majority class + TF-IDF + logistic regression | done |
| Semantic model | Frozen multilingual MiniLM embeddings + LR | done |
| Signal | `q = r·d·c·n·a`, capped attention, theme aggregation | done |
| Sensitivity | 5-variant ablation + rank-stability test | done |
| Backtest design | Walk-forward splits, purge, embargo, cost model | done |
| Backtest result | **Not reported** — data cannot support it | blocked, documented |
| Synthetic proof | 4-window walk-forward on a planted-signal panel | done, labelled |
| Report | `solution.md`, 3,981 words, 14 sections + appendix | done |

The one thing that is *not* delivered is a return result. That is the correct
outcome, not an omission — see §5.

---

## 2. Timeline

### Hours 1–2 — Audit before touching anything

Read the CSV cold before writing a parser. Established early:

- 119 rows, 4 columns: `nickname`, `content`, `views`, `retweets`.
- The `views` column is a **string with mixed units**: plain integers, thousands
  separators (`1,234`), Latin suffixes (`12K`, `1.5M`), and Chinese suffixes
  (`3.4万`, `1.2亿`).
- Text is heavily mixed: Traditional Chinese, Simplified Chinese, English, and
  code-switched posts within a single row.
- Some rows have their entire content inside a URL.
- Text is truncated around 265 characters.

**Decision made here:** record a SHA-256 of `users.csv` and assert it unchanged at
the end of notebook 01. The original file is never opened for writing; all
transformation happens on an in-memory copy. Both `users.csv` and
`data/raw/users.csv` hash to
`8d8d63aa2c936aeda18e033d92f6b9569fb637f2823697df3d891c5e0e1c683c`.

**The finding that shaped everything else:** there is no timestamp column, no post
ID, and no price data. I checked for indirect substitutes (row order, engagement
totals, nickname clustering) and found none that survive scrutiny. This is not a
cleaning problem; it is a missing-data problem, and no amount of engineering fixes
it.

### Hours 2–3 — Cleaning and annotation

Built `src/data.py` with a deliberately strict parser:

- `parse_views` handles all four formats and returns **`NaN` for anything it cannot
  parse** — never `0`. Coercing an unparseable view count to zero would invent a
  data point.
- `normalise_text` applies NFKC, replaces URLs with `<URL>` and mentions with
  `<USER>`, normalises fullwidth punctuation, and **does not stem, lemmatise or
  remove stop-words**. Negation and cashtags carry investment meaning; stripping
  them would destroy the signal before modelling began.
- `cluster_near_duplicates` uses union-find on cosine similarity at a 0.92
  threshold.

Result: zero data loss, zero duplicate rows, seven URLs recovered, median cleaned
length 119 characters, **zero near-duplicate clusters**.

Wrote `data/processed/ANNOTATION_GUIDE.md` **before** labelling anything. Seven
stages (A–G) with explicit inclusion and exclusion criteria, and an explicit list
of what counts as *not* tradable relevance (personal updates, tutorials, pure
promotion, unactionable sentiment). Labelled all 119 rows against it:
**21 positives, 98 negatives** (17.6% positive), 13 themes, direction +1 on 12,
−1 on 7, 0 on 2.

**Honest weakness recorded at the time:** the annotator is a single AI research
assistant working from the written rubric. There is **no inter-annotator
agreement** and no second opinion. This is stated in the guide, in §5 of the
report, and in the limitations. It caps how much any classifier result can mean.

### Hours 3–5 — Models

Five models, all under `RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
random_state=42)`:

| ID | Model | Macro F1 (mean ± SD) |
|---|---|---|
| B0 | Majority class | 0.4516 ± 0.0049 |
| B1 | TF-IDF (char_wb 2–4) + LR | 0.4513 ± 0.0365 |
| B2 | Frozen MiniLM embeddings + LR | **0.7251 ± 0.0591** |
| B3 | Embeddings + log-views | 0.7226 ± 0.0654 |
| B4 | Metadata only | 0.4089 ± 0.0825 |

Because 21 positives is small, added a **200-permutation label-shuffle test**:
observed 0.7251, null mean 0.4879, SD 0.0558, 95th percentile 0.5877, null maximum
0.6352, empirical **p = 0.0050**. The p-value uses the finite-sample correction
`(1 + #{null ≥ observed}) / (1 + N_PERM)`, so it can never be reported as exactly
zero.

### Hours 5–6 — Signal and ablation

Defined `q = r · d · c · n · a` with `a = clip(1 + 0.20·z(log1p(views)), 0.5, 1.5)`
and z clipped to ±3, α = 0.20 fixed **before** inspecting any result. Aggregated to
`S_a = Σq / (Σr + 1e-6)` per theme, excluding `NONE` and propagating `NaN`.

Ran a 5-variant ablation (S0 relevance+attention down to S4 full) and a
rank-stability check. S0 vs S1 Spearman 0.577; S0 vs S4 1.000. The ablation shows
**direction and confidence carry most of the ranking information**, and the
attention cap is a refinement (S3 vs S2 Spearman 0.808; mean signal 0.110 → 0.141),
not a driver.

### Hours 6–7 — Backtest design, and the decision not to fake it

Wrote `walk_forward_splits` (4 splits, min train fraction 0.5, `purge_days=5`,
`embargo_days=2`) and `CostModel(commission_bps=5.0, slippage_bps=5.0)` → 10 bps
one-way. Wrote `portfolio_return` and `signal_to_weights`.

Then hit the wall: `t10_backtest_data_gap.csv` records **7 of 7 required fields
MISSING** (`post_id`, `author_id`, `created_at_utc`, `collected_at_utc`,
`engagement_at_decision`, `ticker_mapping`, `prices_pit`). Without a decision
timestamp and a forward price there is no label, no execution price, and no way to
distinguish forecast from hindsight.

**Decision: do not fabricate.** Instead, ran the machinery once on a synthetic
panel — 20 assets × 104 weekly rows (2,080 rows, 2022-01-07 → 2023-12-29), planted
β = 0.004, σ = 0.02, realised correlation 0.1186 — purely to prove the plumbing
executes. Four windows, all with positive gross and net, all with costs charged:

| Window | Train / test | Gross | Net | Cost | Turnover |
|---|---|---|---|---|---|
| WF1 | 1,020 / 240 | 0.0421 | 0.0311 | 0.0110 | 5.50 |
| WF2 | 1,280 / 240 | 0.0354 | 0.0229 | 0.0125 | 6.25 |
| WF3 | 1,540 / 240 | 0.0476 | 0.0349 | 0.0127 | 6.33 |
| WF4 | 1,800 / 240 | 0.0280 | 0.0155 | 0.0125 | 6.25 |

These numbers are **meaningless as finance**. They exist to show that splits are
chronological and disjoint, purge masks remove the right rows, costs are charged,
and turnover arithmetic is consistent. The figure is named
`fig07_synthetic_walkforward_ENGINEERING_ONLY.png` and its title reads *"Synthetic
engineering validation only — not an investment result"*.

### Hours 7–8 — Write, then re-run everything

Wrote `solution.md`, then trimmed it to fit the 2,500–4,000 word limit (final:
**3,981 strict prose words**; Executive Summary **247** raw words against a 250
limit). Re-executed all three notebooks from clean kernels. Produced this file,
`README.md`, `requirements.txt` and `FINAL_AUDIT.md`.

---

## 3. Corrections made

### 3.1 Transaction-cost convention — a 2× error, found and fixed

**The bug.** The first implementation charged costs on portfolio *turnover*
(`0.5 × Σ|Δw|`) instead of on *traded notional* (`Σ|Δw|`). Because turnover is
conventionally half the traded notional, this understated every cost by exactly
2×. A strategy that should have paid 10 bps appeared to pay 5.

**Why it was invisible.** The code ran, produced plausible numbers, and had no
failing test. A cost model that is wrong by a constant factor looks exactly like a
cost model that is right.

**How it surfaced.** By asking what "10 bps" is a price *of* — the answer is "a
unit of traded notional" — and then constructing the simplest possible case: open a
gross-1.0 book from flat and check the charge. With `w = [0.25, 0.25, −0.25,
−0.25]` from flat, traded notional is exactly 1.0, so the cost must be exactly
0.001 (10 bps). The buggy version returned 0.0005.

**The fix.** `cost = traded × (commission_bps + slippage_bps) / 1e4` where
`traded = Σ|w − prev|`. Turnover is now computed separately as `0.5 × traded` and
reported as its own column so the two can never be confused again. Notebook 03
asserts the flat-open case explicitly and also asserts that a zero-turnover hold
costs exactly `0.0`.

**Lesson recorded in the report (§14):** a 2× error in turnover convention is
invisible in a unit test and devastating in a backtest. Gross return without net
beside it is not a result.

### 3.2 Over-claiming "validates the label"

Three places in notebook 02 implied that the permutation test validated the
*annotation rubric*. It does not — it tests whether the text representations carry
information associated with the labels I assigned. If the rubric were wrong, the
classifier would happily learn the wrong thing and still score well.

Fixed by rewording all three:

- Section 7b retitled *"Is the B2 advantage distinguishable from chance? A
  label-permutation test"*.
- The printed conclusion now reads: *"the text representations contain information
  associated with the manually assigned tradable-relevance labels. This does not
  independently validate the annotation rubric, and it does not establish
  investment predictability."*
- The figure title now ends *"label classification only; not a return or
  profitability claim"*.

### 3.3 Synthetic figure that looked like a result

Figure 7's original title read like a performance chart. Rewritten to
*"Synthetic engineering validation only — not an investment result"*, with the
filename carrying `ENGINEERING_ONLY`, and the caption in the report repeating it.
This satisfies the requirement that synthetic demonstrations be labelled **in the
figure title itself**, not only in surrounding prose.

### 3.4 Word-counting bug in my own tooling
My first word counter stripped LaTeX blocks with `re.sub(r"(?ms)^\$\$.*?^\$\$",
"", text)`. Under `(?ms)`, `^` matches at *every* line start, so the pattern matched
from the first `$$` to the last `$$` and deleted thousands of characters of real
prose. It reported 3,915 words when the true count was ~4,395.

Caught by comparing the whole-document count against the sum of per-section counts:
4,395 ≠ 4,255 is arithmetically impossible unless stripping is position-dependent.
Fixed with `^\$\$.*?\$\$[ \t]*$` — non-greedy, closing fence anchored to end of
line. Worth recording because a broken measurement tool silently produces a
confident, wrong answer.

### 3.5 Absolute filesystem paths echoed into committed notebooks

`save_figure` and `save_table` returned a `pathlib.Path`. Jupyter echoes the return
value of the last expression in a cell, so the notebooks captured
`PosixPath('/home/<user>/retu/outputs/figures/fig01_views_distribution.png')` as a
cell result. That leaks a local username and absolute path into a file intended to
be portable.

Neither return value was ever consumed, so both helpers now return `None` and keep
printing the repo-relative path instead. All three notebooks were re-executed
afterwards; the file sizes dropped by ~700 bytes each as the leaked strings were
removed. The only remaining match for `/home/` in the repository is the explanatory
comment in `src/plotting.py` that documents this fix.

---

## 4. What failed, and is kept as failed

These are in the report because deleting them would misrepresent the work.

**B1 — TF-IDF + logistic regression: 0.4513 against a 0.4516 majority baseline.**
A net effect of −0.0003. Char n-grams on 119 short, code-switched, truncated
documents cannot separate the classes; the model essentially learns the prior. Kept
as a reported row, not quietly dropped, because "TF-IDF achieves 0.45 macro F1"
would sound like a result if the baseline were absent.

**B4 — metadata only: 0.4089, worse than the majority baseline.** Nickname and
retweet count carry no tradable-relevance information here. Reported as a negative
result.

**Direction modelling.** Only 21 positives carry a direction, split 12 up / 7 down
/ 2 neutral. That is far too few to fit or evaluate a direction classifier, so
direction enters the signal as a fixed signed input from the annotation, not as a
prediction. Stated plainly rather than dressed up.

**Theme sparsity.** 13 themes over 21 positives means the maximum is 3 positives in
a single theme. Per-theme scores are therefore indicative orderings, not estimates
with meaningful precision. Only 21 themes are non-zero under S0 and 19 under the
others.

**Near-duplicate detection found nothing.** The union-find clustering at 0.92
returned zero clusters. That is a true negative on this snapshot, but it also means
the deduplication logic is **untested against a real positive case** here. Flagged
as a residual risk.

**Ticker mapping is thin.** 12 of 19 tickers in the lexicon are never mentioned.
AAPL 4, BABA 4, TENCENT 3, GOOGL 2, XIAOMI 2, and BTC/MU/PLTR/TM/XPEV at 1 each.
Entity-based mapping cannot carry the signal; the theme vocabulary does the work.

---

## 5. Why there is no backtest result

This is the most important decision in the project, so it is stated explicitly.

A backtest needs, at minimum: a decision time, a signal known at that time, an
execution price after that time, and a forward return. The snapshot supplies
**none** of the first three and cannot supply the fourth. `t10` lists all seven
missing fields with the reason each one is needed:

| Missing | Why it blocks the backtest |
|---|---|
| `post_id` | cannot dedupe or join |
| `author_id` | cannot separate a new voice from a repeating one |
| `created_at_utc` | **cannot place a post before or after a price move** |
| `collected_at_utc` | cannot compute availability lag |
| `engagement_at_decision` | views are a final total, not a point-in-time value |
| `ticker_mapping` | free text is not a security identifier |
| `prices_pit` | no point-in-time prices to compute a return |

`t11_required_schema.csv` specifies the 15 fields that *would* be needed, with
dtype and purpose, so the gap is actionable rather than just a complaint.

Using the current `views` figure as if it were historical would be look-ahead bias
by construction: the view count is a cumulative total observed *after* whatever
happened next. Any return computed from it would be contaminated and unfalsifiable.

So: backtest machinery written, tested on synthetic data, **no performance number
reported for the real data**. Any reviewer who wanted a Sharpe ratio would have to
accept one manufactured from data that cannot produce it — which is the failure
mode this project was designed to avoid.

---

## 6. Verification status

| Check | Result |
|---|---|
| `users.csv` SHA-256 unchanged | pass — `8d8d63aa…e0e1c683c` |
| Notebooks execute top-to-bottom, 0 errors | pass — 01 (34 cells), 02 (51 cells), 03 (24 cells) |
| CV fold index overlap | pass — `assert_no_index_overlap` on all 25 folds |
| Vectoriser fit on training folds only | pass — per-fold vocab sizes 1,970 / 1,934 / 1,920 / 1,913 / 2,011 |
| Scaling fit on training folds only | pass — per-fold train-only log-views means 9.94308 / 10.00112 / 9.84721 / 10.06152 / 10.00933 vs full-data 9.97253 |
| Majority baseline present | pass — B0 |
| TF-IDF baseline present | pass — B1 |
| Semantic embedding tested | pass — B2, frozen MiniLM, 384-d |
| Class-imbalance-aware metrics | pass — macro F1, balanced accuracy, per-class precision/recall |
| Fold-level scores reported, not just the mean | pass — mean ± fold SD in `t04` |
| Costs charged on traded notional | pass — asserted in notebook 03 |
| Synthetic result labelled in figure title | pass — `fig07_…_ENGINEERING_ONLY` |
| `solution.md` within 2,500–4,000 words | pass — 3,981 strict prose words |
| Executive summary ≤ 250 words | pass — 247 raw words |

---

## 7. If I had another week

In priority order, and only the things that would actually change a conclusion:

1. **Get timestamps and point-in-time prices.** Everything else is downstream. With
   `created_at_utc`, `collected_at_utc` and `prices_pit`, the existing pipeline runs
   a real walk-forward backtest unchanged — the machinery is already built and
   tested.
2. **A second annotator.** Even 30 overlapping rows would let me report a kappa and
   put a number on label noise. Right now the 0.7251 macro F1 has an unknown
   ceiling set by rubric reliability.
3. **More positives.** 21 is the binding constraint on every model comparison. 200+
   would let direction become a prediction target instead of a fixed input, and
   would make the B2/B3 difference (+0.0025) resolvable.
4. **A proper entity linker.** A 19-ticker lexicon missing 12 names is a placeholder.
   Real cashtag and company-name resolution with confidence scores would make
   `ticker_mapping` trustworthy.

Explicitly *not* on the list: fine-tuning a transformer on 21 examples, adding more
models to a comparison that fold noise already dominates, or building a dashboard.
None of those would change what this dataset can support.
