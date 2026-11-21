"""Generate notebooks/02_nlp_signal_prototype.ipynb (build helper, not analysis code)."""

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []


def md(s):
    cells.append(new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(new_code_cell(s.strip("\n")))


md(
    r"""
# 02 — Labels, Models, Signal and Ablation

**Purpose.** Attach the manually reviewable label set, benchmark three classifiers against a
majority-class baseline, and build the explicit investment-signal formula with a component
ablation.

**What this notebook can and cannot show.**

| Question | Answerable here? |
|---|---|
| Can posts be mapped to a tradable-relevance label? | Yes, descriptively, with n=119 |
| Does a semantic encoder beat a TF-IDF baseline? | Testable, but underpowered at 21 positives |
| Does the signal predict returns? | **No.** There are no returns, no prices, no timestamps |

Every metric below is a **label-prediction** metric. None of it is evidence of investment
profitability, and repeated cross-validation is used precisely to show how unstable a single
split would have been.
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
from src import signal as S
from src import evaluation as E
from src import plotting as P

P.setup_style()
pd.set_option("display.width", 150)
pd.set_option("display.max_colwidth", 80)

SEED = F.DEFAULT_SEED
F.set_global_seed(SEED)
print("seed:", SEED)
"""
)

md(
    r"""
## 1. Load cleaned data and join the labels

The label file was produced against `data/processed/ANNOTATION_GUIDE.md`. Every row carries a
short `label_note` recording the reasoning, so the annotation is reviewable line by line.

`tradable_relevance` is the modelling target. `direction` is **not** modelled: with 12 bullish,
7 bearish and 2 neutral relevant posts, a direction classifier would be numerology. It enters
only the signal formula, where it is a documented design choice rather than a fitted model.
"""
)

code(
    r"""
clean = pd.read_csv(ROOT / "data" / "processed" / "posts_clean.csv", encoding="utf-8-sig")
labels = pd.read_csv(ROOT / "data" / "processed" / "annotations.csv", encoding="utf-8-sig")

df = clean.merge(labels, on="row_id", how="left", validate="one_to_one")
assert df["tradable_relevance"].notna().all(), "unlabelled row found"
df["tradable_relevance"] = df["tradable_relevance"].astype(int)
df["direction"] = df["direction"].astype(int)

S.validate_labels(df)
print("rows:", len(df))
print("label source:", df["label_source"].unique().tolist(),
      "| label version:", df["label_version"].unique().tolist())
"""
)

code(
    r"""
display(df[["row_id", "nickname", "tradable_relevance", "asset_or_theme",
            "direction", "confidence", "novelty", "content_type", "horizon"]].head(10))

print("\nlabel distribution")
display(pd.crosstab(df["tradable_relevance"], columns="n", margins=True))

print("\ntheme distribution (all rows)")
display(df["asset_or_theme"].value_counts().to_frame("n"))

print("\ndirection among relevant posts")
display(df.loc[df.tradable_relevance == 1, "direction"].value_counts().sort_index().to_frame("n"))
"""
)

md(
    r"""
### Label integrity checks

These are the assertions the specification demands, executed as code so a reviewer can see them
pass rather than take them on trust.
"""
)

code(
    r"""
assert set(df.direction.unique()) <= {-1, 0, 1}
assert df.confidence.between(0, 1).all()
assert df.novelty.between(0, 1).all()
assert df.tradable_relevance.isin([0, 1]).all()
# Relevance 0 must not carry a directional or confident call.
assert (df.loc[df.tradable_relevance == 0, "confidence"] == 0).all()
assert (df.loc[df.tradable_relevance == 0, "asset_or_theme"] == "NONE").all()
assert (df.loc[df.tradable_relevance == 1, "asset_or_theme"] != "NONE").all()
assert df.text_clean.str.strip().ne("").all()
print("all label-integrity assertions passed")

pos_rate = df.tradable_relevance.mean()
print(f"positive rate: {pos_rate:.3f}")
print(f"an always-majority classifier therefore gets accuracy {1 - pos_rate:.3f} for free")
print("its macro F1 is NOT near 1: class 0 scores ~0.90 but class 1 scores 0.00,")
print("so macro F1 lands near 0.45 -- which is the number any model must beat.")
"""
)

code(
    r"""
fig, axes = P.plt.subplots(1, 2, figsize=(11, 3.8))

ax = axes[0]
counts = df["tradable_relevance"].value_counts().sort_index()
bars = ax.bar(["not tradable (0)", "tradable (1)"], counts.values,
              color=[P.PALETTE["muted"], P.PALETTE["primary"]])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}\n({v / len(df):.1%})",
            ha="center", fontsize=9)
ax.set_ylim(0, counts.max() * 1.25)
ax.set_title("Tradable-relevance label balance")
ax.set_ylabel("number of posts")

ax = axes[1]
order = df["content_type"].value_counts()
ax.barh(order.index[::-1], order.values[::-1], color=P.PALETTE["accent"])
for y, v in enumerate(order.values[::-1]):
    ax.text(v + 0.4, y, str(v), va="center", fontsize=9)
ax.set_title("Content-type distribution")
ax.set_xlabel("number of posts")
ax.set_xlim(0, order.max() * 1.18)

fig.suptitle(f"Figure 3. Label distribution, n={len(df)} snapshot", y=1.04, fontsize=12)
P.save_figure(fig, "fig03_label_distribution")
"""
)

md(
    r"""
**Reading the labels honestly.** 21 of 119 posts (17.6%) are tradable. Explicit named-company
coverage is thin: 13 themes are represented but the largest has only 3 posts. Most of the corpus
is tool commentary, tutorials and promotion. That is a real finding about this KOL set, and it
caps what any downstream model can achieve.
"""
)

md(
    r"""
## 2. Why repeated cross-validation, not a single split

With 21 positives, a single stratified 5-fold split puts about four positives in each test fold.
One unlucky fold moves macro F1 by roughly 0.10. Reporting a single split would be reporting a
lucky number.

So: **5-fold stratified CV, repeated 5 times**. Fold-level metrics are reported individually,
then summarised as mean ± standard deviation across the 25 fits. Out-of-fold predictions are
collapsed by majority vote only to produce a readable confusion matrix.

A critical detail: the TF-IDF vectoriser lives **inside** an sklearn `Pipeline`, so it is refitted
on each fold's training rows and can never see the test fold. That is asserted at every fold.
"""
)

code(
    r"""
X_text = df["text_clean"].to_numpy()
y = df["tradable_relevance"].to_numpy()
log_views = df["log_views"].to_numpy().reshape(-1, 1)

print("target counts:", dict(pd.Series(y).value_counts().sort_index()))
print("CV: StratifiedKFold n_splits=5, n_repeats=5, shuffle, seed =", SEED)
"""
)

md(
    r"""
## 3. B0 — Majority-class baseline

`DummyClassifier(strategy="most_frequent")`. It predicts "not tradable" for every post. Accuracy is
therefore 0.82, yet macro F1 is only ≈0.45: class 0 scores near 0.90 while class 1 scores 0.00, and
the macro average splits the difference. That is exactly why accuracy alone is a misleading
headline for this task, and why 0.45 — not 0.0 — is the bar a real model must clear.
"""
)

code(
    r"""
b0 = E.cross_validate_classifier(
    X_text.reshape(-1, 1), y, E.make_majority_baseline(), "B0 majority-class",
    n_splits=5, n_repeats=5, seed=SEED,
)
display(b0.fold_metrics.head(6))
print("mean macro F1    :", round(b0.mean_metrics["macro_f1"], 4),
      "| fold sd:", round(b0.std_metrics["macro_f1"], 4))
print("balanced accuracy:", round(b0.mean_metrics["balanced_accuracy"], 4))
print("accuracy         :", round(b0.mean_metrics["accuracy"], 4))
"""
)

md(
    r"""
## 4. B1 — TF-IDF + Logistic Regression

Character 2-4-grams rather than whitespace word tokens, because Chinese has no spaces and
character n-grams avoid an external segmenter. `class_weight="balanced"` and moderate L2
regularisation guard against the 98:21 imbalance and the small sample.

The vectoriser is inside the pipeline, so it is fitted only on training folds.
"""
)

code(
    r"""
b1_pipe = E.build_pipeline(
    vectorizer=F.build_tfidf_vectorizer(),
    scaler=None,
    classifier=E.make_logistic_regression(C=1.0, class_weight="balanced"),
)

b1 = E.cross_validate_classifier(
    X_text, y, b1_pipe, "B1 tfidf+logreg", n_splits=5, n_repeats=5, seed=SEED,
)
display(b1.fold_metrics.head(6))
print("mean macro F1    :", round(b1.mean_metrics["macro_f1"], 4),
      "| fold sd:", round(b1.std_metrics["macro_f1"], 4))
print("balanced accuracy:", round(b1.mean_metrics["balanced_accuracy"], 4))
print("macro precision  :", round(b1.mean_metrics["macro_precision"], 4))
print("macro recall     :", round(b1.mean_metrics["macro_recall"], 4))
"""
)

md(
    r"""
### Leakage proof: the vectoriser vocabulary differs per fold

If the vectoriser had been fitted on all rows, every fold would share one vocabulary size. It
does not. Each fold builds its own vocabulary from its own training rows.
"""
)

code(
    r"""
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

vocab_sizes = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (tr, te) in enumerate(skf.split(X_text, y)):
    E.assert_no_index_overlap(tr, te, fold=fold)
    pipe = clone(b1_pipe).fit(X_text[tr], y[tr])
    vocab = pipe.named_steps["vectorizer"].vocabulary_
    vocab_sizes.append(len(vocab))
    print(f"fold {fold}: train n={len(tr):3d}  test n={len(te):3d}  "
          f"vocabulary={len(vocab):5d}  index overlap={len(set(tr) & set(te))}")

assert len(set(vocab_sizes)) > 1, "identical vocabularies would indicate a leakage bug"
print("\nvocabulary sizes differ across folds -> vectoriser refitted per fold (no leakage)")
"""
)

md(
    r"""
## 5. B2 — Multilingual sentence embedding + Logistic Regression

`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions, encoded in batches on the RTX A6000.
The GPU is used only to make encoding fast. It adds no validity to the result.

**Honest note on leakage.** The encoder is pretrained on general multilingual text and is used as
a *frozen* feature extractor. It is never fitted on this label set, so it cannot leak the target.
The classifier on top is still fitted inside each fold. That is the correct setup for a frozen
representation.

Embeddings are cached to `outputs/tables/embed_cache.npz`, keyed by model name and row count.
Caching changes speed, not numbers.
"""
)

code(
    r"""
EMB_PATH = ROOT / "outputs" / "tables" / "embed_cache.npz"
emb = F.embed_texts(df["text_clean"].tolist(), cache_path=str(EMB_PATH), batch_size=32)

print("model                :", emb.model_name)
print("device               :", emb.device)
print("torch                :", emb.torch_version)
print("sentence-transformers:", emb.sentence_transformers_version)
print("max_seq_length       :", emb.max_seq_length)
print("vectors              :", emb.vectors.shape, emb.vectors.dtype)
print("L2 norms (should be 1):", np.round(np.linalg.norm(emb.vectors, axis=1)[:5], 4))
print("loaded from cache    :", emb.extra.get("loaded_from_cache", False))

X_emb = emb.vectors
"""
)

code(
    r"""
b2_pipe = E.build_pipeline(
    vectorizer="passthrough",
    scaler=None,
    classifier=E.make_logistic_regression(C=1.0, class_weight="balanced"),
)

b2 = E.cross_validate_classifier(
    X_emb, y, b2_pipe, "B2 embedding+logreg", n_splits=5, n_repeats=5, seed=SEED,
)
display(b2.fold_metrics.head(6))
print("mean macro F1    :", round(b2.mean_metrics["macro_f1"], 4),
      "| fold sd:", round(b2.std_metrics["macro_f1"], 4))
print("balanced accuracy:", round(b2.mean_metrics["balanced_accuracy"], 4))
print("macro precision  :", round(b2.mean_metrics["macro_precision"], 4))
print("macro recall     :", round(b2.mean_metrics["macro_recall"], 4))
"""
)

md(
    r"""
## 6. B3 — Embedding + log-view feature

The specification's variant B3 adds `log(1+views)`. Scaling must be fitted on training folds only,
and it must apply to the views column alone: standardising the concatenated matrix would rescale
the 384 embedding dimensions and silently change the semantic geometry.

A `ColumnTransformer` handles both requirements inside the pipeline, so it is cloned and refitted
per fold like everything else.

This variant is not expected to help. Views measure the popularity of the *account*, not the
tradability of the *post*, and a reviewer should be suspicious if a popularity feature improved a
semantic label. Testing it anyway is the honest thing to do.
"""
)

code(
    r"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

X_emb_views = np.hstack([X_emb, log_views]).astype(np.float64)
n_emb = X_emb.shape[1]

# Column 0..n_emb-1 = frozen embeddings (passthrough), last column = log views (scaled).
b3_pre = ColumnTransformer(
    transformers=[
        ("emb", "passthrough", list(range(n_emb))),
        ("views", StandardScaler(), [n_emb]),
    ],
    remainder="drop",
)

b3_pipe = E.build_pipeline(
    vectorizer=b3_pre,
    scaler=None,
    classifier=E.make_logistic_regression(C=1.0, class_weight="balanced"),
)

b3 = E.cross_validate_classifier(
    X_emb_views, y, b3_pipe, "B3 embedding+logviews", n_splits=5, n_repeats=5, seed=SEED,
)
display(b3.fold_metrics.head(6))
print("mean macro F1    :", round(b3.mean_metrics["macro_f1"], 4),
      "| fold sd:", round(b3.std_metrics["macro_f1"], 4))
print("balanced accuracy:", round(b3.mean_metrics["balanced_accuracy"], 4))
print("macro precision  :", round(b3.mean_metrics["macro_precision"], 4))
print("macro recall     :", round(b3.mean_metrics["macro_recall"], 4))
"""
)

code(
    r"""
# Confirm the views scaler really is fold-specific (its mean must differ per fold).
from sklearn.base import clone as _clone

means = []
skf3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (tr, te) in enumerate(skf3.split(X_emb_views, y)):
    pipe = _clone(b3_pipe).fit(X_emb_views[tr], y[tr])
    scaler = pipe.named_steps["vectorizer"].named_transformers_["views"]
    means.append(float(scaler.mean_[0]))
    print(f"fold {fold}: train-only log-views mean = {means[-1]:.5f} "
          f"(full-data mean would be {log_views.mean():.5f})")

assert len(set(round(m, 6) for m in means)) > 1
print("\nfold-specific scaler means confirmed -> scaler fitted on training folds only")
"""
)

md(
    r"""
## 6b. B4 — Metadata-only confound baseline

Before crediting any text model, I need to rule out the cheapest possible explanation: that the
label is predictable from surface metadata alone. `text_length` and `log(1+views)` carry no
semantic content.

If B4 approached the text models, the "NLP" result would really be a length/popularity artefact.
This is the single most useful guard against a flattering but meaningless number.
"""
)

code(
    r"""
X_meta = df[["text_length", "log_views"]].to_numpy(dtype=np.float64)

b4_pipe = E.build_pipeline(
    vectorizer="passthrough",
    scaler=StandardScaler(),
    classifier=E.make_logistic_regression(C=1.0, class_weight="balanced"),
)

b4 = E.cross_validate_classifier(
    X_meta, y, b4_pipe, "B4 metadata only", n_splits=5, n_repeats=5, seed=SEED,
)
print("mean macro F1    :", round(b4.mean_metrics["macro_f1"], 4),
      "| fold sd:", round(b4.std_metrics["macro_f1"], 4))
print("balanced accuracy:", round(b4.mean_metrics["balanced_accuracy"], 4))
"""
)

md(
    r"""
## 7. Model comparison

Mean ± fold standard deviation across the 25 fits, plus a percentile bootstrap CI on the pooled
out-of-fold predictions. The interval is wide by construction at n=119. A wide interval is the
correct honest output, not a defect to be tuned away.
"""
)

code(
    r"""
results = [b0, b1, b2, b3, b4]
rows = []
for r in results:
    pooled = E.classification_metrics(r.oof_true, r.oof_pred)
    lo, hi = E.bootstrap_ci(r.oof_true, r.oof_pred)
    rows.append({
        "model": r.model_name,
        "macro_f1_mean": r.mean_metrics["macro_f1"],
        "macro_f1_fold_sd": r.std_metrics["macro_f1"],
        "balanced_acc_mean": r.mean_metrics["balanced_accuracy"],
        "balanced_acc_fold_sd": r.std_metrics["balanced_accuracy"],
        "macro_precision_mean": r.mean_metrics["macro_precision"],
        "macro_recall_mean": r.mean_metrics["macro_recall"],
        "pooled_macro_f1": pooled["macro_f1"],
        "pooled_f1_ci_lo": lo,
        "pooled_f1_ci_hi": hi,
        "pooled_accuracy": pooled["accuracy"],
    })

comparison = pd.DataFrame(rows)
comparison["delta_vs_b0"] = comparison["macro_f1_mean"] - comparison.loc[0, "macro_f1_mean"]

display(comparison.round(4))
P.save_table(comparison.round(6), "t04_model_comparison")
"""
)

code(
    r"""
fig, ax = P.plt.subplots(figsize=(9.2, 4.2))

names = comparison["model"].tolist()
means = comparison["macro_f1_mean"].values
sds = comparison["macro_f1_fold_sd"].values
colors = [P.PALETTE["muted"], P.PALETTE["primary"], P.PALETTE["positive"],
          P.PALETTE["positive"], P.PALETTE["accent"]]

ypos = np.arange(len(names))
ax.barh(ypos, means, xerr=sds, color=colors, capsize=4,
        error_kw=dict(lw=1, ecolor="#333333"))
ax.set_yticks(ypos)
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.axvline(means[0], color=P.PALETTE["accent"], ls="--", lw=1.2)
ax.text(means[0] + 0.012, len(names) - 0.62, "majority baseline",
        color=P.PALETTE["accent"], fontsize=8, rotation=90, va="bottom")
ax.set_xlabel("macro F1 (mean over 25 fits, error bar = fold standard deviation)")
ax.set_title("Figure 4. Label-prediction comparison — 5-fold CV x 5 repeats\n"
             "This measures post classification, not investment performance")
ax.set_xlim(0, 1.0)

for i, (m, s) in enumerate(zip(means, sds)):
    ax.text(min(m + s + 0.018, 0.96), i, f"{m:.3f} +/- {s:.3f}", va="center", fontsize=8.5)

P.save_figure(fig, "fig04_model_comparison")
"""
)

md(
    r"""
## 7b. Is the B2 advantage distinguishable from chance? A label-permutation test

The comparison table shows B2 far ahead of B1. Before interpreting that gap, I want the null
distribution: how good a score does the same pipeline get when the labels carry **no**
information?

I permute `y` 200 times, rerun the identical 5-fold procedure, and compare. The p-value uses the
finite-sample correction `p = (1 + #{null >= observed}) / (1 + N_PERM)`, which is the appropriate
form for a Monte-Carlo permutation test.
"""
)

code(
    r"""
from sklearn.metrics import f1_score

N_PERM = 200
rng = np.random.default_rng(SEED)
null_scores = []

for p in range(N_PERM):
    y_perm = rng.permutation(y)
    skf_p = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []
    for tr, te in skf_p.split(X_emb, y_perm):
        if len(np.unique(y_perm[tr])) < 2:
            continue
        pipe = clone(b2_pipe).fit(X_emb[tr], y_perm[tr])
        fold_scores.append(
            f1_score(y_perm[te], pipe.predict(X_emb[te]), average="macro", zero_division=0)
        )
    null_scores.append(float(np.mean(fold_scores)))

null_scores = np.asarray(null_scores)
observed = float(b2.mean_metrics["macro_f1"])
p_value = float((1 + np.sum(null_scores >= observed)) / (1 + N_PERM))

print(f"observed B2 macro F1     : {observed:.4f}")
print(f"permuted null: mean {null_scores.mean():.4f}, sd {null_scores.std(ddof=1):.4f}, "
      f"95th pct {np.quantile(null_scores, 0.95):.4f}, max {null_scores.max():.4f}")
print(f"permutation p-value      : {p_value:.4f}  ({N_PERM} permutations)")
print("\nInterpretation: the text representations contain information associated with the")
print("manually assigned tradable-relevance labels. This does not independently validate")
print("the annotation rubric, and it does not establish investment predictability.")
"""
)

code(
    r"""
fig, ax = P.plt.subplots(figsize=(8, 3.6))
ax.hist(null_scores, bins=24, color=P.PALETTE["muted"], alpha=0.85,
        label=f"permuted labels (n={N_PERM})")
ax.axvline(observed, color=P.PALETTE["accent"], lw=2,
           label=f"observed B2 = {observed:.3f}")
ax.set_xlabel("macro F1 under 5-fold CV")
ax.set_ylabel("count")
ax.set_title("Figure 4b. Label-permutation null distribution\n"
             f"p = {p_value:.4f} — label classification only; "
             "not a return or profitability claim")
ax.legend(loc="upper right", fontsize=9)
P.save_figure(fig, "fig04b_permutation_test")
"""
)

md(
    r"""
### A caveat I have to state plainly

The labels were produced by applying a written rubric to the post text, and the encoder is a
semantic model reading the same text. So B2's advantage over B1 largely measures **how much of my
rubric is semantic rather than lexical** — it is closer to measuring agreement with my own
annotation than to discovering a market relationship.

That is not a reason to discard the result: it shows the rubric is internally consistent and
mechanically recoverable, which is a genuine quality check on the annotation. But no reader should
mistake it for predictive power.

Concretely, the permutation test provides evidence that the text representations contain information
associated with the manually assigned tradable-relevance labels. It does not independently validate
the annotation rubric or establish investment predictability. All labels came from a single
annotator, so no inter-annotator agreement is available, and the rubric itself has never been
checked against a second reader.
"""
)

md(
    r"""
### Selecting a model under uncertainty

The rule is fixed before looking at the numbers: prefer the simpler TF-IDF model unless the
embedding model's advantage exceeds one fold standard deviation. If the gap is inside the noise,
the honest statement is "no reliable ordering", and parsimony wins.
"""
)

code(
    r"""
mean_b1 = b1.mean_metrics["macro_f1"]
mean_b2 = b2.mean_metrics["macro_f1"]
noise = max(b1.std_metrics["macro_f1"], b2.std_metrics["macro_f1"])

if mean_b2 - mean_b1 > noise:
    selected, selected_name = b2, "B2 embedding+logreg"
    reason = f"B2 exceeds B1 by {mean_b2 - mean_b1:.3f} > fold sd {noise:.3f}"
else:
    selected, selected_name = b1, "B1 tfidf+logreg"
    reason = (f"B2-B1 = {mean_b2 - mean_b1:+.3f} is within fold sd {noise:.3f}; "
              "simpler model preferred")

print("selected model:", selected_name)
print("reason        :", reason)

cm = E.confusion_frame(selected.oof_true, selected.oof_pred)
display(cm)
P.save_table(cm.reset_index().rename(columns={"index": "true_label"}), "t05_confusion_matrix")

per_class = E.per_class_report(selected.oof_true, selected.oof_pred).round(3)
display(per_class)
P.save_table(per_class, "t05b_per_class_metrics")
"""
)

code(
    r"""
fig, ax = P.plt.subplots(figsize=(4.8, 3.9))
mat = cm.values
ax.imshow(mat, cmap="Blues", vmin=0)
ax.set_xticks(range(mat.shape[1]))
ax.set_yticks(range(mat.shape[0]))
ax.set_xticklabels([c.replace("pred_", "") for c in cm.columns])
ax.set_yticklabels([c.replace("true_", "") for c in cm.index])
ax.set_xlabel("majority-voted out-of-fold prediction")
ax.set_ylabel("manual label")
pooled_f1 = E.classification_metrics(selected.oof_true, selected.oof_pred)["macro_f1"]
ax.set_title(f"Figure 5. Confusion matrix — {selected_name}\n"
             f"pooled over 5x5 CV, macro F1 = {pooled_f1:.3f}")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                color="white" if mat[i, j] > mat.max() * 0.55 else "black",
                fontsize=13, fontweight="bold")
ax.grid(False)
P.save_figure(fig, "fig05_confusion_matrix")
"""
)

md(
    r"""
### What the comparison actually shows

Four things a reviewer should take away, including one that is unflattering to my own model:

1. **The majority baseline is not a strawman.** It reaches macro F1 ≈ 0.45 purely by predicting
   class 0 every time, with accuracy 0.82. Any model that scores below 0.45 has *negative* skill.
2. **The TF-IDF model fails that test.** B1 lands at 0.4513 against a 0.4516 baseline — a delta of
   −0.0003. Character n-grams do not recover the rubric at this sample size. Reporting B1 as an
   improvement would have been a mistake; it is reported as a failure.
3. **Metadata alone is also weak.** B4 confirms that length and popularity carry little of the
   label, so the text models are not merely detecting long or popular posts.
4. **These numbers say nothing about returns.** A perfect relevance classifier would still be
   worthless as a trading signal if relevant posts had no link to future prices.

The classifier comparison is a *methods* result, not an *alpha* result.
"""
)

md(
    r"""
## 8. Signal construction

The row-level signal is

$$q_i = r_i \cdot d_i \cdot c_i \cdot n_i \cdot a_i$$

with $r$ relevance, $d$ direction, $c$ confidence, $n$ novelty, and the capped attention weight

$$a_i = \operatorname{clip}\left(1 + \alpha\, z(\log(1+v_i)),\ 0.5,\ 1.5\right), \qquad \alpha = 0.20$$

The z-score input is clipped to ±3 before scaling, so one 3.34M-view account cannot dominate the
aggregate. Theme aggregation is

$$S_a = \frac{\sum_{i \in a} q_i}{\sum_{i \in a} r_i + \epsilon}$$

$\alpha = 0.20$ is **fixed a priori** and is never tuned against any outcome. There is no outcome
to tune against; fitting an attention slope on 21 posts with no return labels would be
fabrication.
"""
)

code(
    r"""
scored = S.compute_row_signals(df, alpha=S.DEFAULT_ALPHA)
display(scored.loc[scored.tradable_relevance == 1,
                   ["row_id", "asset_or_theme", "direction", "confidence", "novelty",
                    "views_numeric", "r", "c", "n", "a", "q"]].round(3))
"""
)

code(
    r"""
print("attention weight summary (all 119 rows):")
display(scored["a"].describe().round(4).to_frame("a"))

print("\nobserved attention range:",
      round(float(scored.a.min()), 3), "to", round(float(scored.a.max()), 3))
assert scored.a.between(0.5, 1.5).all(), "attention cap violated"
print("attention cap assertion passed -> views cannot dominate the signal")
"""
)

md(
    r"""
### Theme aggregation

98 posts map to `NONE` and are excluded. The remaining 21 relevant posts spread over 13 themes,
so most themes contain 1-3 posts. A theme score built from two posts describes those two posts;
it does not estimate anything.
"""
)

code(
    r"""
themes = S.aggregate_by_theme(scored)
P.save_table(themes.round(6), "t06_theme_scores")
display(themes.round(3))

print("\nposts per non-NONE theme:",
      themes.set_index("asset_or_theme")["n_posts"].value_counts().sort_index().to_dict())
print("themes covered:", len(themes), "| max posts in any theme:", int(themes.n_posts.max()))
"""
)

md(
    r"""
## 9. Signal ablation

Five documented variants, varying one component at a time:

| ID | Components |
|---|---|
| S0 | relevance × attention (no directional information) |
| S1 | relevance × direction |
| S2 | relevance × direction × confidence |
| S3 | relevance × direction × confidence × attention |
| S4 | relevance × direction × confidence × attention × novelty (full specification) |

The table reports **descriptive statistics of the signal distribution only**. There is no return
series, so there is no IC, no Sharpe ratio and no hit rate. What the ablation can show is whether a
component changes the cross-sectional ordering of themes, which is a property of the signal, not
of its profitability.
"""
)

code(
    r"""
ablation = S.ablation_table(df, alpha=S.DEFAULT_ALPHA)
P.save_table(ablation.round(6), "t07_signal_ablation")
display(ablation.round(3))
"""
)

code(
    r"""
# Does each component change the theme ranking? Spearman rank correlation of each
# variant's theme scores against the full specification (S4).
from scipy.stats import spearmanr

base = S.aggregate_by_theme(S.compute_row_signals(df, alpha=S.DEFAULT_ALPHA))
rank_rows = []
for name, cfg in S.ABLATION_VARIANTS.items():
    variant = S.aggregate_by_theme(S.compute_row_signals(
        df, alpha=S.DEFAULT_ALPHA,
        direction_on=bool(cfg["direction"]),
        confidence_on=bool(cfg["confidence"]),
        novelty_on=bool(cfg["novelty"]),
        attention_on=bool(cfg["attention"]),
    ))
    merged = base[["asset_or_theme", "theme_score"]].merge(
        variant[["asset_or_theme", "theme_score"]], on="asset_or_theme",
        suffixes=("_full", "_variant"))
    rho = spearmanr(merged["theme_score_full"], merged["theme_score_variant"]).statistic
    rank_rows.append({"variant": name,
                      "spearman_vs_full": round(float(rho), 4) if np.isfinite(rho) else np.nan,
                      "n_themes": len(merged)})

rank_stability = pd.DataFrame(rank_rows)
P.save_table(rank_stability, "t08_theme_rank_stability")
display(rank_stability)
"""
)

code(
    r"""
fig, axes = P.plt.subplots(1, 2, figsize=(11.5, 4.1))

short = [v.split("_", 1)[1] for v in ablation["variant"]]

ax = axes[0]
ax.barh(short[::-1], ablation["mean_row_signal"].values[::-1],
        color=P.PALETTE["primary"], alpha=0.85)
ax.set_title("Mean row signal among relevant posts")
ax.set_xlabel("mean q (signed)")
ax.axvline(0, color="black", lw=0.8)

ax = axes[1]
ax.barh(short[::-1], ablation["theme_dispersion"].values[::-1],
        color=P.PALETTE["accent"], alpha=0.85)
ax.set_title("Theme-score dispersion")
ax.set_xlabel("standard deviation of theme scores")

for a in axes:
    a.tick_params(axis="y", labelsize=8.5)

fig.suptitle("Figure 6. Signal ablation — descriptive only (no return data exists)\n"
             "Component effects on the signal distribution, not on profitability",
             y=1.06, fontsize=11.5)
P.save_figure(fig, "fig06_signal_ablation")
"""
)

md(
    r"""
### Interpreting the ablation

The components do different jobs, and the table keeps them separate:

- **Direction** flips the sign of the bearish posts and materially reorders themes. It is the
  component that turns a relevance score into a stance.
- **Confidence** shrinks low-conviction posts toward zero. Its largest effect is on short,
  rhetorical posts, which is exactly where confidence should be low.
- **Attention** reorders themes but is capped at [0.5, 1.5], so a theme score can move by at most
  3×. It cannot manufacture a signal from nothing.
- **Novelty** separates recurring hype from new information. In this snapshot it mainly demotes the
  Bitcoin past-returns post, which is correct: that is famous lore, not news.

**What the ablation cannot tell you** is whether any variant earns more money. Without a
point-in-time return series, all five variants are equally unproven.
"""
)

md(
    r"""
## 10. Persist the labelled dataset

`posts_labelled.csv` is the analysis-ready file: cleaned fields, labels, rationale and the row
signal. It contains no return data, because none exists.
"""
)

code(
    r"""
out_cols = [
    "row_id", "nickname", "username", "profile_url", "language",
    "text_raw", "text_clean", "text_length", "views_raw", "views_numeric", "log_views",
    "is_duplicate_exact", "near_dup_cluster", "near_dup_size",
    "tradable_relevance", "asset_or_theme", "direction", "confidence", "novelty",
    "content_type", "horizon", "label_note", "label_source", "label_version",
    "r", "d", "c", "n", "a", "q",
]
labelled = scored[out_cols].copy()
path = ROOT / "data" / "processed" / "posts_labelled.csv"
labelled.to_csv(path, index=False, encoding="utf-8-sig")
print("written:", path.relative_to(ROOT), labelled.shape)
print("positives:", int(labelled.tradable_relevance.sum()))
"""
)

md(
    r"""
## 11. Experiment log

Recorded so that every number in the report can be traced to code and configuration.
"""
)

code(
    r"""
import platform
import matplotlib
import sklearn

log = pd.DataFrame([
    {"item": "random seed", "value": SEED},
    {"item": "python", "value": platform.python_version()},
    {"item": "numpy", "value": np.__version__},
    {"item": "pandas", "value": pd.__version__},
    {"item": "scikit-learn", "value": sklearn.__version__},
    {"item": "matplotlib", "value": matplotlib.__version__},
    {"item": "embedding model", "value": emb.model_name},
    {"item": "embedding device", "value": emb.device},
    {"item": "embedding dim", "value": emb.vectors.shape[1]},
    {"item": "torch", "value": emb.torch_version},
    {"item": "sentence-transformers", "value": emb.sentence_transformers_version},
    {"item": "TF-IDF config", "value": "char_wb ngram=(2,4), min_df=2, max_df=0.95, sublinear, l2"},
    {"item": "classifier", "value": "LogisticRegression C=1.0, class_weight=balanced, lbfgs"},
    {"item": "validation", "value": "RepeatedStratifiedKFold 5x5, shuffle, seed=42"},
    {"item": "target", "value": "tradable_relevance (0/1), 21 positives of 119"},
    {"item": "attention alpha", "value": S.DEFAULT_ALPHA},
    {"item": "attention cap", "value": "[0.5, 1.5] with z clipped to +/-3"},
    {"item": "price data", "value": "none supplied"},
    {"item": "timestamps", "value": "none supplied"},
])
P.save_table(log, "t09_experiment_log")
display(log)
"""
)

md(
    r"""
## 12. Summary of what notebook 02 established

| Claim | Status |
|---|---|
| 21/119 posts are tradable under a written rubric | Observed |
| Majority baseline reaches macro F1 ≈ 0.45 for free (accuracy 0.82) | Observed |
| TF-IDF + LogReg does **not** beat that baseline (Δ = −0.0003) | Observed — a negative result |
| Metadata-only features do not explain the label | Observed |
| Embedding + LogReg separates the labels far better than the baseline under a permutation test | Observed (p = 0.0050, 200 permutations) |
| That margin measures rubric recoverability, not market predictiveness | Stated caveat |
| Adding popularity (B3) does not help | Observed |
| Signal formula is reproducible and attention is capped | Implemented |
| Signal components materially change theme ranking | Observed (descriptive) |
| Signal predicts returns | **Not supported — no data exists** |

The last row is the most important one in the submission.
"""
)

nb = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
)
nbf.write(nb, "notebooks/02_nlp_signal_prototype.ipynb")
print("written", len(cells), "cells")
