"""Data loading, validation, cleaning and audit utilities.

Design rules
------------
* The original ``users.csv`` is never modified. All derived fields are added
  alongside the raw columns so that every transformation is auditable.
* Invalid view counts become ``NaN``, never ``0``. ``0`` is only kept when it
  is explicitly present in the source.
* Text normalisation keeps negation, cashtags, hashtags, numbers and emoji,
  because all of these can carry investment meaning.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

#: Accepted aliases for each required field (Chinese and English).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "nickname": ("昵称", "nickname", "name", "display_name", "用户昵称"),
    "profile_url": ("首页地址", "profile_url", "url", "homepage", "profile"),
    "views_raw": ("阅读数", "views", "views_raw", "average_views"),
    "text_raw": ("最近一条推文", "latest_post", "text", "latest_tweet", "推文"),
}

REQUIRED_FIELDS: tuple[str, ...] = tuple(COLUMN_ALIASES)

#: Canonical column order of the cleaned snapshot.
CLEAN_COLUMNS: tuple[str, ...] = (
    "row_id",
    "nickname",
    "profile_url",
    "username",
    "views_raw",
    "views_numeric",
    "views_parse_status",
    "log_views",
    "text_raw",
    "text_clean",
    "text_has_url",
    "n_urls",
    "n_cashtags",
    "n_hashtags",
    "language",
    "text_length",
    "word_count_latin",
    "is_duplicate_exact",
    "is_duplicate_author_text",
    "near_dup_cluster",
    "near_dup_size",
)


class SchemaError(ValueError):
    """Raised when a required field cannot be located in the input table."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _normalise_column_name(name: str) -> str:
    """Strip whitespace, BOM and zero-width characters from a column name."""
    return str(name).replace("\ufeff", "").replace("\u200b", "").strip()


def _resolve_columns(columns: Iterable[str]) -> dict[str, str]:
    """Map canonical field names to the actual column names present.

    Raises
    ------
    SchemaError
        If any required field is missing, with a message listing what was found.
    """
    normalised = {_normalise_column_name(c): c for c in columns}
    lowered = {k.lower(): v for k, v in normalised.items()}
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for field, aliases in COLUMN_ALIASES.items():
        hit: str | None = None
        for alias in aliases:
            if alias in normalised:
                hit = normalised[alias]
                break
            if alias.lower() in lowered:
                hit = lowered[alias.lower()]
                break
        if hit is None:
            missing.append(field)
        else:
            resolved[field] = hit

    if missing:
        raise SchemaError(
            "Missing required field(s): "
            f"{missing}. Columns found: {list(columns)}. "
            f"Expected aliases: {COLUMN_ALIASES}"
        )
    return resolved


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw KOL snapshot and rename columns to canonical names.

    The file is read as UTF-8. The returned frame keeps the original values
    untouched; only column names are canonicalised.

    Parameters
    ----------
    path
        Path to ``users.csv``.

    Returns
    -------
    pandas.DataFrame
        Frame with canonical column names and one row per source record.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    mapping = _resolve_columns(raw.columns)
    out = raw.rename(columns={v: k for k, v in mapping.items()})

    # Keep only recognised fields; anything else is preserved with a prefix so
    # that nothing is silently dropped.
    extras = [c for c in raw.columns if c not in mapping.values()]
    if extras:
        out = pd.concat(
            [out, raw[extras].add_prefix("extra__")],
            axis=1,
        )

    for field in REQUIRED_FIELDS:
        out[field] = out[field].astype("string").fillna("").str.strip()

    if out.empty:
        raise ValueError(f"No rows loaded from {path}")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Views parsing
# --------------------------------------------------------------------------- #

_MULTIPLIERS: dict[str, float] = {
    "k": 1e3,
    "K": 1e3,
    "千": 1e3,
    "m": 1e6,
    "M": 1e6,
    "mn": 1e6,
    "b": 1e9,
    "B": 1e9,
    "万": 1e4,
    "萬": 1e4,
    "亿": 1e8,
    "億": 1e8,
}

_VIEWS_CLEAN_RE = re.compile(r"[,\s_'\u00a0]")


def parse_views(value: object) -> float:
    """Parse a single views value into a float.

    Handles ``"1,234"``, ``"12K"``, ``"1.5M"``, ``"3.4万"``, ``"1.2亿"``,
    plain numbers and common suffixes such as ``"次"`` / ``"views"``.
    Returns ``float('nan')`` for blank or malformed input. An explicit ``0``
    stays ``0.0``.

    Parameters
    ----------
    value
        Raw value from the views column.

    Returns
    -------
    float
        Parsed count, or ``nan`` when the value cannot be interpreted.
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if not np.isnan(value) else float("nan")

    s = _VIEWS_CLEAN_RE.sub("", str(value))
    if s == "":
        return float("nan")

    # Drop trailing units that carry no numeric meaning.
    s = re.sub(r"(次|views?|浏览|閱讀|阅读)", "", s, flags=re.IGNORECASE)
    s = s.replace("+", "")
    if s == "":
        return float("nan")

    lowered = s.lower()
    multiplier = 1.0
    for suffix in ("mn", "亿", "億", "万", "萬", "k", "m", "b", "千"):
        if lowered.endswith(suffix.lower()):
            multiplier = _MULTIPLIERS[suffix]
            s = s[: -len(suffix)]
            break

    s = s.strip()
    if s == "":
        # e.g. "K" with no number -> not a valid count.
        return float("nan")

    try:
        number = float(s)
    except ValueError:
        return float("nan")

    if not np.isfinite(number) or number < 0:
        return float("nan")
    return number * multiplier


def parse_views_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Vectorised views parsing with an explicit status column.

    Returns
    -------
    tuple of (numeric, status)
        ``status`` is one of ``{'explicit_zero', 'parsed', 'missing'}`` so that
        a failed parse is never confused with a genuine zero.
    """
    numeric = series.map(parse_views).astype(float)

    status = np.where(
        numeric.isna(),
        "missing",
        np.where(numeric == 0, "explicit_zero", "parsed"),
    )
    return numeric, pd.Series(status, index=series.index, dtype="string")


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

# The upstream export inserts spaces inside URLs ("https:// pan.quark.cn/x",
# "youtu.be/uPT...?si =abc"). A plain ``\S+`` pattern therefore misses almost
# every link. This pattern matches the scheme, then URL-ish characters, and
# continues across a space only when the next chunk is itself URL-shaped (it
# contains a "." or "/"), which avoids swallowing surrounding natural language.
_URL_BODY = r"[\w\-./?=&%#~+:@!$'*,;]"
_URL_RE = re.compile(
    rf"(?:(?:https?://|www\.){_URL_BODY}*)"
    rf"(?:\s+(?={_URL_BODY}*[./=?]){_URL_BODY}*)*",
    flags=re.IGNORECASE,
)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]{1,30}")
_WS_RE = re.compile(r"[ \t\u00a0\u3000]+")
_NEWLINE_RE = re.compile(r"\n\s*\n+")
# Keep $cashtags and #hashtags intact: they are investment-meaningful tokens.
_CASHTAG_RE = re.compile(r"\$[A-Za-z][A-Za-z0-9_.]{0,9}")
_HASHTAG_RE = re.compile(r"#[^\s#]+")
_FULLWIDTH_PUNCT = {"，": ",", "。": ".", "：": ":", "；": ";", "！": "!", "？": "?"}


def extract_urls(text: str) -> list[str]:
    """Return all URLs found in ``text`` (raw form, for later auditing)."""
    return _URL_RE.findall(str(text))


def normalise_text(text: str, url_token: str = "<URL>", mention_token: str = "<USER>") -> str:
    """Normalise post text while preserving financially meaningful tokens.

    Steps: NFKC normalisation, URL and @mention replacement, whitespace
    collapse. Cashtags, hashtags, numbers, negation words and emoji are kept.
    No stemming is applied, which is important for Chinese and mixed-language
    text where stemming would destroy meaning.

    Parameters
    ----------
    text
        Raw post text.
    url_token, mention_token
        Placeholders substituted for URLs and @mentions.

    Returns
    -------
    str
        Cleaned text suitable for TF-IDF / embedding models.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = _URL_RE.sub(f" {url_token} ", s)
    s = _MENTION_RE.sub(f" {mention_token} ", s)
    for fw, ascii_p in _FULLWIDTH_PUNCT.items():
        s = s.replace(fw, ascii_p)
    s = _NEWLINE_RE.sub("\n", s)
    s = _WS_RE.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    return s.strip()


# --------------------------------------------------------------------------- #
# Language and simple text statistics
# --------------------------------------------------------------------------- #

_RE_HAN = re.compile(r"[\u4e00-\u9fff]")
_RE_KANA = re.compile(r"[\u3040-\u30ff]")
_RE_HANGUL = re.compile(r"[\uac00-\ud7af]")
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_LATIN_WORD = re.compile(r"[A-Za-z]+")


def detect_language(text: str) -> str:
    """Heuristic language tag: ``zh``, ``ja``, ``ko``, ``en``, ``mixed``, ``other``.

    Japanese/Korean are detected first through their syllabaries. Otherwise a
    post is ``zh`` when Han characters dominate, ``en`` when Latin letters
    dominate, and ``mixed`` when both scripts carry substantial weight. This is
    a deterministic heuristic, not a trained language identifier; the resulting
    distribution is reported as an approximation.
    """
    s = str(text)
    n_han = len(_RE_HAN.findall(s))
    n_kana = len(_RE_KANA.findall(s))
    n_hangul = len(_RE_HANGUL.findall(s))
    n_latin = len(_RE_LATIN.findall(s))
    total = n_han + n_kana + n_hangul + n_latin

    if total == 0:
        return "other"
    if n_kana / total > 0.10:
        return "ja"
    if n_hangul / total > 0.10:
        return "ko"

    han_share = n_han / total
    latin_share = n_latin / total
    if han_share >= 0.60:
        return "zh"
    if latin_share >= 0.80:
        return "en"
    if han_share >= 0.15 and latin_share >= 0.15:
        return "mixed"
    return "other"


def count_latin_words(text: str) -> int:
    """Number of Latin alphabetic words, a crude proxy for code-switching."""
    return len(_RE_LATIN_WORD.findall(str(text)))


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #


def flag_exact_duplicates(df: pd.DataFrame, text_col: str = "text_clean") -> pd.Series:
    """Flag rows whose cleaned text appears more than once (keeper = first)."""
    keys = df[text_col].astype(str).str.strip()
    return keys.duplicated(keep="first")


def flag_author_text_duplicates(
    df: pd.DataFrame, author_col: str = "nickname", text_col: str = "text_clean"
) -> pd.Series:
    """Flag repeated (author, cleaned text) pairs, keeping the first occurrence."""
    keys = df[author_col].astype(str) + "\u241f" + df[text_col].astype(str)
    return keys.duplicated(keep="first")


def cluster_near_duplicates(
    similarity: np.ndarray, threshold: float = 0.92
) -> tuple[np.ndarray, np.ndarray]:
    """Single-linkage clustering from a cosine-similarity matrix.

    Parameters
    ----------
    similarity
        Square (n, n) cosine-similarity matrix.
    threshold
        Pairs at or above this cosine similarity are linked. The default of
        ``0.92`` follows the project specification.

    Returns
    -------
    tuple of (cluster_id, cluster_size)
        ``cluster_id`` is an integer label per row (0 is the largest cluster
        after ordering by first occurrence); ``cluster_size`` is the size of the
        row's cluster. Near-duplicates are flagged, not deleted.
    """
    sim = np.asarray(similarity, dtype=float)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity must be a square matrix")

    n = sim.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    rows, cols = np.where(np.triu(sim, k=1) >= threshold)
    for i, j in zip(rows, cols):
        union(int(i), int(j))

    roots = np.array([find(i) for i in range(n)])
    _, cluster_id = np.unique(roots, return_inverse=True)
    sizes = np.bincount(cluster_id)
    return cluster_id, sizes[cluster_id]


# --------------------------------------------------------------------------- #
# Full cleaning pipeline
# --------------------------------------------------------------------------- #


def build_clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the processed snapshot with raw and derived columns side by side.

    Parameters
    ----------
    raw
        Output of :func:`load_raw`.

    Returns
    -------
    pandas.DataFrame
        One row per source record with columns in :data:`CLEAN_COLUMNS`.
    """
    df = raw.copy()

    df.insert(0, "row_id", np.arange(len(df), dtype=int))
    df["username"] = (
        df["profile_url"]
        .astype(str)
        .str.rstrip("/")
        .str.split("/")
        .str[-1]
        .replace({"": np.nan})
    )

    views_numeric, views_status = parse_views_series(df["views_raw"])
    df["views_numeric"] = views_numeric
    df["views_parse_status"] = views_status
    df["log_views"] = np.log1p(views_numeric)

    df["text_raw"] = df["text_raw"].astype(str)
    df["text_clean"] = df["text_raw"].map(normalise_text)
    df["text_has_url"] = df["text_raw"].map(lambda s: bool(extract_urls(s)))
    df["n_urls"] = df["text_raw"].map(lambda s: len(extract_urls(s)))
    df["n_cashtags"] = df["text_raw"].map(lambda s: len(_CASHTAG_RE.findall(s)))
    df["n_hashtags"] = df["text_raw"].map(lambda s: len(_HASHTAG_RE.findall(s)))
    df["language"] = df["text_raw"].map(detect_language)
    df["text_length"] = df["text_clean"].str.len().astype(int)
    df["word_count_latin"] = df["text_clean"].map(count_latin_words).astype(int)

    df["is_duplicate_exact"] = flag_exact_duplicates(df)
    df["is_duplicate_author_text"] = flag_author_text_duplicates(df)
    # Filled in later by ``features.attach_near_duplicate_flags`` once
    # embeddings are available; defaults make the frame usable without them.
    df["near_dup_cluster"] = 0
    df["near_dup_size"] = 1

    ordered = [c for c in CLEAN_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Audit helpers
# --------------------------------------------------------------------------- #


def audit_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness summary (count and share of blank/NaN)."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        blank = s.isna() | (s.astype(str).str.strip() == "")
        n_missing = int(blank.sum())
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_missing": n_missing,
                "pct_missing": round(100.0 * n_missing / max(n, 1), 2),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def audit_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of duplicate types checked in the pipeline."""
    checks = {
        "full_row_duplicates": int(df.duplicated().sum()),
        "exact_text_duplicates": int(df["is_duplicate_exact"].sum()),
        "author_text_duplicates": int(df["is_duplicate_author_text"].sum()),
        "duplicate_profile_urls": int(df["profile_url"].duplicated().sum()),
        "duplicate_nicknames": int(df["nickname"].duplicated().sum()),
        "near_duplicate_pairs_0.92": int(
            (df["near_dup_size"] > 1).sum() // 2 if "near_dup_size" in df else 0
        ),
    }
    return pd.DataFrame(
        {"check": list(checks), "count": list(checks.values())}
    )


def views_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for raw and log views.

    ``views_numeric`` is clipped only for display in derived columns; the raw
    values are never overwritten.
    """
    v = df["views_numeric"].astype(float)
    lv = df["log_views"].astype(float)
    stats = {
        "n_non_missing": int(v.notna().sum()),
        "n_missing": int(v.isna().sum()),
        "n_explicit_zero": int((v == 0).sum()),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "p05": float(v.quantile(0.05)),
        "p25": float(v.quantile(0.25)),
        "median": float(v.median()),
        "p75": float(v.quantile(0.75)),
        "p95": float(v.quantile(0.95)),
        "max": float(v.max()),
        "skew_raw": float(v.skew()),
        "skew_log": float(lv.skew()),
        "kurtosis_raw": float(v.kurtosis()),
    }
    return pd.DataFrame({"statistic": list(stats), "value": list(stats.values())})