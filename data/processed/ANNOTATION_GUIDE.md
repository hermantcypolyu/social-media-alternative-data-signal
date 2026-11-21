# Annotation Guide — v1.0

**Scope:** all 119 rows of the supplied `users.csv` snapshot.
**Annotator:** AI research assistant working from this written rubric, with row-level
rationale recorded in `data/processed/posts_labelled.csv` (`label_note` column).
**Status:** single-annotator. No independent second annotator and therefore **no
inter-annotator agreement statistic**. This is a documented limitation, not a hidden one.

The rubric is written *before* labelling so that the label set can be reviewed,
challenged and reproduced. Every label must be justifiable from the post text alone —
no outside knowledge of what happened to any price afterwards.

---

## Stage A — `tradable_relevance` ∈ {0, 1}

> Does this post's **primary content** contain information or opinion that could
> reasonably relate to the future price, risk or competitive position of a liquid
> asset or an investable theme?

Set `tradable_relevance = 1` only if the post makes a claim with **business or market
framing**, for example:

- demand, pricing, supply, margins, or capacity;
- competitive position, market share, or product-launch strategy;
- regulatory or policy impact on an industry;
- capital markets events (listings, funding, index inclusion);
- an explicit directional view on an asset, sector or theme.

Set `tradable_relevance = 0` when the post is:

- a personal update, purchase log, or career anecdote;
- a tutorial, installation guide, or configuration snippet;
- promotional or affiliate content;
- generic product enthusiasm ("this tool is amazing") with no market framing;
- commentary on a **private** company where no liquid proxy is claimed;
- entertainment, memes, or hashtag-only posts.

**Worked exclusions.** "Product X is amazing, you must install it" is *not* tradable
under this rubric. AGENT.md §8 Stage C explicitly warns that positive wording about a
product does not imply a bullish stock view. A product post qualifies only when the
author additionally makes a demand, pricing, or competitive claim.

**Deliberate limitation.** This rubric yields a low positive rate. That is the honest
base rate for AI-tool chatter, and the low rate is itself a finding.

---

## Stage B — `asset_or_theme`

Controlled vocabulary. Use `NONE` when Stage A is 0. Use the **primary** theme only;
do not stack tags.

| Code | Meaning | Liquid proxy (illustrative) |
|---|---|---|
| `NVDA` | NVIDIA | NVDA |
| `MSFT` | Microsoft | MSFT |
| `GOOGL` | Alphabet | GOOGL |
| `AAPL` | Apple | AAPL |
| `META` | Meta | META |
| `AMZN` | Amazon | AMZN |
| `TSLA` | Tesla | TSLA |
| `AMD` | AMD | AMD |
| `TSM` | TSMC | TSM |
| `AVGO` | Broadcom | AVGO |
| `MU` | Micron | MU |
| `PLTR` | Palantir | PLTR |
| `BABA` | Alibaba | BABA |
| `TENCENT` | Tencent | 0700.HK |
| `XIAOMI` | Xiaomi | 1810.HK |
| `XPEV` | XPeng | XPEV |
| `TM` | Toyota | TM |
| `BTC` | Bitcoin | BTC-USD |
| `ETH` | Ethereum | ETH-USD |
| `SEMI_SECTOR` | Semiconductors / memory / AI hardware | SOXX, SMH |
| `CRYPTO_SECTOR` | Crypto market as a whole | e.g. BTC-USD, total market cap |
| `CHINA_TECH` | China technology policy / self-sufficiency | KWEB, FXI |
| `AI_SECTOR` | AI software/infrastructure, no single name | e.g. software baskets |
| `NONE` | Not tradable, or no defensible mapping | — |

Rules:

1. Prefer the **narrowest defensible** mapping.
2. Never force a ticker. A sector code is better than a wrong single name.
3. A private company mention maps to `NONE` unless the post also makes a claim about a
   listed competitor or supplier.
4. `UNCLEAR` is not used in this snapshot; if a post is that ambiguous it is
   `relevance = 0`.

---

## Stage C — `direction` ∈ {-1, 0, +1}

The author's stance **toward the mapped asset or theme**, not general tone.

- `+1` — the content implies improving fundamentals, demand, or competitive position.
- `0` — descriptive, mixed, ambiguous, or a factual report with no stance.
- `-1` — the content implies deteriorating fundamentals, lost share, or criticism of the
  asset's strategy/execution.

Rules:

1. Judge stance toward the *asset*, not enthusiasm about the *product*.
2. Criticism of execution or strategy is `-1`. Factual reporting of a neutral event is `0`.
3. Sarcasm and rhetorical mocking count as `-1` only when the target is the mapped asset.
4. If both positive and negative elements are material, use `0`.

---

## Stage D — `confidence` ∈ [0, 1]

Confidence in the **directional call**, not in the relevance decision.

Set `confidence = 0.0` when `tradable_relevance = 0`, because there is no directional
call to be confident about. Lower confidence when:

- the asset mapping is indirect (theme rather than a named company);
- the horizon is unclear;
- the post is very short or partly truncated;
- the claim is rhetorical rather than analytical;
- conflicting elements are present.

Reference points: `≥0.7` explicit, analytical, named asset. `0.4–0.6` implied or
rhetorical. `<0.4` weak or incidental mapping.

---

## Stage E — `novelty` ∈ [0, 1]

How much *new* information the post carries relative to the corpus, ignoring whether it
is tradable.

- `≥0.7` — new event, new data point, original analysis.
- `0.4–0.7` — familiar theme with a specific new detail.
- `≤0.3` — reposted lore, promotional slogan, recurring hype, or a well-known fact.

---

## Stage F — `content_type`

`news` | `analysis` | `opinion` | `promotion` | `personal` | `tutorial` | `other`

- `news` — reports an event or announcement (often with a date or source).
- `analysis` — reasoned argument supported by facts or numbers.
- `opinion` — stance or reaction with little supporting evidence.
- `promotion` — sells, links, or pushes a product, course, or account.
- `personal` — diary, purchase, career, or social content.
- `tutorial` — instructions, configuration, or how-to.
- `other` — none of the above (memes, hashtag-only, spam).

---

## Stage G — `horizon`

`intraday` | `1_5_days` | `medium_term` | `long_term` | `unclear`

The horizon the post's claim would plausibly affect. Use `unclear` when Stage A is 0.
`medium_term` covers weeks to a quarter; `long_term` covers beyond a quarter.

---

## Known weaknesses of this label set

1. **Single annotator, AI-assisted.** No second rater, so no Cohen's κ. Labels should be
   treated as a documented reference set, not ground truth.
2. **Text is truncated** at roughly 265 characters by the upstream export. Some posts are
   cut mid-sentence, which can only reduce confidence.
3. **No timestamps.** "Novelty" and "horizon" are judged against an undated snapshot.
4. **21 positives out of 119 (17.6%).** This is too few for a well-powered classifier
   comparison. All model metrics must be read as descriptive, with wide uncertainty, and
   confirmed with repeated cross-validation rather than a single split.
5. **Theme sparsity.** 21 relevant posts spread across 13 non-`NONE` themes (maximum 3
   posts in any one theme) means a cross-sectional portfolio cannot be formed. This is
   reported, not worked around.
6. **Direction is not modellable.** Among the 21 relevant posts, direction is +1 for 12,
   -1 for 7 and 0 for 2. A three-class direction classifier on 21 examples would be
   meaningless, so direction is used only inside the signal formula, never as a model
   target with reported performance.
