# Social-Media Alternative-Data Investment Signal

A two-day research prototype that asks a narrow question: **can business-relevant
trading claims be separated from noise in KOL social-media posts, and can they be
turned into a transparent, cost-aware cross-sectional signal?**

The short answer is *partly*: a frozen multilingual sentence encoder separates the
manually assigned tradable-relevance labels far better than a lexical baseline or
the majority class, but the supplied snapshot contains **no timestamps, no post
identifiers and no prices**, so **no return result is reported and none is
claimed**.

Start with **[`solution.md`](solution.md)** — it is the report. This file only
explains how to run the code.

---

## Headline results (label classification only)

Macro F1, 5-fold × 5-repeat stratified cross-validation, mean ± fold SD:

| ID | Model | Macro F1 | Balanced accuracy | Δ vs baseline |
|---|---|---|---|---|
| B0 | Majority class | 0.4516 ± 0.0049 | 0.5000 | — |
| B1 | TF-IDF + logistic regression | 0.4513 ± 0.0365 | 0.4907 | −0.0003 |
| B2 | Frozen multilingual embedding + LR | **0.7251 ± 0.0591** | 0.7639 | **+0.2735** |
| B3 | Embedding + log-views | 0.7226 ± 0.0654 | 0.7638 | +0.2711 |
| B4 | Metadata only | 0.4089 ± 0.0825 | 0.4369 | −0.0427 |

A 200-permutation label-shuffle test places the observed B2 score outside the null
distribution (null mean 0.4879, SD 0.0558, 95th percentile 0.5877, maximum 0.6352,
empirical **p = 0.0050**).

**These are label-recovery results.** They measure how learnable my own annotation
rubric is from text. They are **not** a return, alpha, profitability or economic
significance claim, and the report never makes one.

---

## Requirements

- Python 3.10 (tested on 3.10.12)
- ~2 GB disk for the environment, plus ~470 MB for the pretrained encoder
- A CUDA GPU is *optional*; it only speeds up batch embedding. All reported numbers
  are identical on CPU.
- No API keys, no paid data, no network access beyond the one-time Hugging Face
  model download.

Exact versions are pinned in [`requirements.txt`](requirements.txt).

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific PyTorch build, install it first and then the rest:

```bash
# CUDA 13.0 build (as tested)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130

# CPU-only build
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Optional sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

---

## Run everything

Execute the three notebooks in order from a **clean kernel**. Each notebook is
self-contained and imports only from `src/`.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit_and_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_nlp_signal_prototype.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_backtest_methodology.ipynb
```

Or open them interactively:

```bash
jupyter notebook notebooks/
```

Runtime is roughly 2–4 minutes per notebook on CPU, faster with a GPU. The first
notebook run downloads the sentence encoder once.

### Ordering matters

| Notebook | Produces |
|---|---|
| `01_data_audit_and_cleaning.ipynb` | `data/processed/posts_clean.csv`, `outputs/tables/t01`–`t03`, `audit_embeddings.npz`, figures 1–2 |
| `02_nlp_signal_prototype.ipynb` | `data/processed/posts_labelled.csv`, `t04`–`t09`, `embed_cache.npz`, figures 3–6 |
| `03_backtest_methodology.ipynb` | `t10`–`t12`, figure 7 |

Notebook 02 reads the output of 01; 03 reads the outputs of 01 and 02. Running 02
before 01 will fail loudly rather than silently.

---

## What is in the repository

```
.
├── solution.md                      # the report — read this first
├── requirements.txt                 # pinned environment
├── README.md                        # this file
├── PROGRESS.md                      # work log, corrections, what failed
├── FINAL_AUDIT.md                   # pass/fail against the submission checklist
├── users.csv                        # original supplied data (never modified)
│
├── data/
│   ├── raw/users.csv                # byte-identical copy of the original
│   └── processed/
│       ├── ANNOTATION_GUIDE.md      # 7-stage rubric, written before labelling
│       ├── annotations.csv          # 119 rows, 21 positives / 98 negatives
│       ├── posts_clean.csv
│       └── posts_labelled.csv
│
├── notebooks/
│   ├── 01_data_audit_and_cleaning.ipynb
│   ├── 02_nlp_signal_prototype.ipynb
│   └── 03_backtest_methodology.ipynb
│
├── src/
│   ├── data.py                      # loading, parsing, cleaning, dedup
│   ├── features.py                  # TF-IDF, frozen embeddings, attention
│   ├── signal.py                    # q = r·d·c·n·a, theme aggregation, ablation
│   ├── evaluation.py                # CV, permutation, walk-forward, costs
│   └── plotting.py                  # shared style and output paths
│
├── tools/                           # notebook generators (development only)
│   ├── build_nb02.py
│   ├── build_nb03.py
│   └── trim_solution.py
│
└── outputs/
    ├── figures/                     # 8 PNG figures
    └── tables/                      # 13 CSV tables + 2 NPZ caches
```

---

## Key design choices

**Frozen encoder, not fine-tuning.** 21 positive examples cannot support
fine-tuning a transformer. The encoder is used as a fixed feature extractor and a
regularised logistic regression sits on top. This is a deliberate constraint, not
a shortcut.

**Attention is capped at [0.5, 1.5].** Views measure reach, not skill. The
multiplier `a = clip(1 + 0.20·z(log1p(views)), 0.5, 1.5)` lets popularity nudge a
post's contribution but never outvote relevance, direction or confidence. α = 0.20
was fixed before any result was inspected.

**No stemming, no stop-word removal.** Stripping negation or cashtags would destroy
investment meaning. Text is normalised (NFKC, fullwidth punctuation, `<URL>` and
`<USER>` placeholders) but not lemmatised.

**`NaN`, never `0`.** A theme with no relevant coverage returns `NaN`, not `0`.
Zero would assert "neutral", which is a claim; `NaN` correctly asserts "no
information".

**Costs are charged on traded notional, not turnover.** `cost = Σ|Δw| × (commission
+ slippage)/10⁴`; turnover is `0.5 × Σ|Δw|` and is reported separately. An earlier
version charged on turnover and understated every cost by 2× — see §9.3 of the
report and `PROGRESS.md`.

**Walk-forward only.** `walk_forward_splits` with `purge_days=5`, `embargo_days=2`
and an untouched final block. Financial data is never randomly split.

---

## Outputs

`outputs/figures/`

| File | Content |
|---|---|
| `fig01_views_distribution.png` | Raw vs log-views distribution |
| `fig02_language_and_length.png` | Language mix and text-length histogram |
| `fig03_label_distribution.png` | Label, direction and theme counts |
| `fig04_model_comparison.png` | B0–B4 macro F1 with fold SD |
| `fig04b_permutation_test.png` | Null distribution with observed score |
| `fig05_confusion_matrix.png` | Pooled out-of-fold confusion, B2 |
| `fig06_signal_ablation.png` | S0–S4 theme scores |
| `fig07_synthetic_walkforward_ENGINEERING_ONLY.png` | Synthetic plumbing test |

`outputs/tables/` — `t01` views summary, `t02` entity counts, `t03` data-quality
summary, `t04` model comparison, `t05`/`t05b` confusion and per-class metrics,
`t06` theme scores, `t07` signal ablation, `t08` rank stability, `t09` experiment
log, `t10` backtest data gap, `t11` required schema, `t12` synthetic walk-forward.

---

## Important caveat

`fig07` and `t12` come from a **synthetic** panel with a deliberately planted
signal. They prove the backtest plumbing executes — disjoint chronological splits,
verified purge, positive costs, correct turnover arithmetic. They are **engineering
validation only** and are not, and must not be read as, an investment result.

The real backtest is not reported because the dataset lacks all seven required
fields (`post_id`, `author_id`, `created_at_utc`, `collected_at_utc`,
`engagement_at_decision`, `ticker_mapping`, `prices_pit`). See `t10` and §4 of the
report.

---

## Reproducibility notes

- Global seed 42 (`DEFAULT_SEED` in `src/features.py`).
- All preprocessing lives inside a scikit-learn `Pipeline` refit per fold;
  per-fold TF-IDF vocabulary sizes (1,970 / 1,934 / 1,920 / 1,913 / 2,011) and
  per-fold train-only `log_views` means are printed in the notebooks as empirical
  proof, not assertion.
- `assert_no_index_overlap` runs on all 25 CV folds.
- The original `users.csv` is never written to. Its SHA-256
  (`8d8d63aa2c936aeda18e033d92f6b9569fb637f2823697df3d891c5e0e1c683c`) is recorded
  and re-verified at the end of notebook 01.
- Embeddings are cached to `outputs/tables/embed_cache.npz`; delete it to force a
  recompute.
