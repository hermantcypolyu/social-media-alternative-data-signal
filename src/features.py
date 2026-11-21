"""Feature construction: TF-IDF, multilingual embeddings and duplicate flags.

All fitted transformers live here so that the notebooks can fit them on
training folds only. Nothing in this module is fitted at import time.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src import data as data_mod

DEFAULT_SEED = 42

#: Compact multilingual encoder chosen by the specification.
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and (if available) PyTorch/CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch is optional for the TF-IDF baseline
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# TF-IDF
# --------------------------------------------------------------------------- #


def build_tfidf_vectorizer(seed: int = DEFAULT_SEED) -> TfidfVectorizer:
    """Word + character n-gram TF-IDF suited to short Chinese/English posts.

    Character n-grams are essential because they provide a substitute for
    Chinese word segmentation without needing an extra dependency. The
    vectoriser must be fitted on training folds only.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents=None,
        norm="l2",
    )


def build_word_vectorizer(seed: int = DEFAULT_SEED) -> TfidfVectorizer:
    """Word-level TF-IDF for the Latin subset of the lexicon.

    Chinese text has no whitespace tokenisation here, so this vectoriser is only
    used as an additional view of the same data in the ablation, never as the
    sole representation.
    """
    return TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)[\$#]?[^\W\d_][^\s]{0,24}",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        norm="l2",
    )


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


@dataclass
class EmbeddingResult:
    """Container for generated embeddings plus reproducibility metadata."""

    vectors: np.ndarray
    model_name: str
    device: str
    batch_size: int
    max_seq_length: int
    torch_version: str
    sentence_transformers_version: str
    n_texts: int
    normalized: bool = True
    extra: dict = field(default_factory=dict)


def _resolve_device(prefer_cuda: bool = True) -> str:
    """Pick CUDA when available and requested, else CPU."""
    if not prefer_cuda:
        return "cpu"
    try:  # pragma: no cover - hardware dependent
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    prefer_cuda: bool = True,
    normalize: bool = True,
    cache_path: str | None = None,
) -> EmbeddingResult:
    """Encode texts with a pretrained multilingual sentence-transformer.

    The GPU (RTX A6000) is used only to make batch encoding fast. It adds no
    scientific validity: the same vectors would be produced on CPU.

    Parameters
    ----------
    texts
        Texts to encode. Empty strings are allowed but will produce a
        degenerate vector; callers should filter them upstream.
    model_name
        HuggingFace model id of a compact multilingual encoder.
    batch_size
        Encoding batch size.
    prefer_cuda
        Use CUDA when available.
    normalize
        L2-normalise outputs so that dot product equals cosine similarity.
    cache_path
        Optional ``.npz`` cache. Caching keeps notebooks fast and reproducible
        without changing the numbers.

    Returns
    -------
    EmbeddingResult
        Vectors plus metadata needed for the experiment log.
    """
    import torch
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    texts = [("" if t is None else str(t)) for t in texts]
    device = _resolve_device(prefer_cuda)

    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if int(cached["n_texts"]) == len(texts) and str(cached["model_name"]) == model_name:
            return EmbeddingResult(
                vectors=cached["vectors"],
                model_name=model_name,
                device=str(cached["device"]),
                batch_size=int(cached["batch_size"]),
                max_seq_length=int(cached["max_seq_length"]),
                torch_version=str(cached["torch_version"]),
                sentence_transformers_version=str(cached["st_version"]),
                n_texts=len(texts),
                normalized=bool(cached["normalized"]),
                extra={"loaded_from_cache": True},
            )

    model = SentenceTransformer(model_name, device=device)
    max_seq_length = int(model.max_seq_length)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    ).astype(np.float32)

    result = EmbeddingResult(
        vectors=vectors,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        torch_version=torch.__version__,
        sentence_transformers_version=sentence_transformers.__version__,
        n_texts=len(texts),
        normalized=normalize,
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            vectors=vectors,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            torch_version=torch.__version__,
            st_version=sentence_transformers.__version__,
            n_texts=len(texts),
            normalized=normalize,
        )
    return result


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity for L2-normalised vectors (or normalise on the fly)."""
    v = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v = v / norms
    return v @ v.T


def attach_near_duplicate_flags(
    df: pd.DataFrame, vectors: np.ndarray, threshold: float = 0.92
) -> pd.DataFrame:
    """Add ``near_dup_cluster`` and ``near_dup_size`` from embedding similarity.

    Near-duplicates are *flagged*, never deleted automatically, per the spec.
    """
    sim = cosine_similarity_matrix(vectors)
    cluster_id, cluster_size = data_mod.cluster_near_duplicates(sim, threshold=threshold)
    out = df.copy()
    out["near_dup_cluster"] = cluster_id
    out["near_dup_size"] = cluster_size
    return out


# --------------------------------------------------------------------------- #
# Attention feature
# --------------------------------------------------------------------------- #


def capped_attention(
    views: pd.Series | np.ndarray,
    alpha: float = 0.20,
    z_clip: float = 3.0,
    lo: float = 0.5,
    hi: float = 1.5,
) -> np.ndarray:
    """Convert views into a capped attention weight.

    ``a = clip(1 + alpha * z(log1p(views)), lo, hi)`` where ``z`` is a standard
    score computed **on the current snapshot**. The z-score is clipped before
    scaling so that a single viral post cannot dominate the aggregate.

    The current snapshot is cross-sectional, so this z-score is not a
    point-in-time quantity. It is a research feature, not a live signal.
    """
    v = np.asarray(views, dtype=float)
    logv = np.log1p(v)
    mean = np.nanmean(logv) if np.isfinite(logv).any() else 0.0
    std = np.nanstd(logv) if np.isfinite(logv).any() else 1.0
    if not np.isfinite(std) or std == 0:
        std = 1.0
    z = np.clip((logv - mean) / std, -z_clip, z_clip)
    a = 1.0 + alpha * z
    a = np.clip(a, lo, hi)
    return np.where(np.isfinite(a), a, 1.0)


class LogViewScaler:
    """Wrap a scaler so it can only ever be fitted on a training subset.

    The class exists to make leakage prevention explicit and testable: the
    notebook demonstrably calls ``fit`` on train indices only.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, values: np.ndarray) -> "LogViewScaler":
        v = np.asarray(values, dtype=float).reshape(-1, 1)
        self.scaler.fit(v)
        self.fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("LogViewScaler must be fitted before transform")
        v = np.asarray(values, dtype=float).reshape(-1, 1)
        return self.scaler.transform(v).ravel()

    def fit_transform(self, train_values: np.ndarray) -> np.ndarray:
        return self.fit(train_values).transform(train_values)


# --------------------------------------------------------------------------- #
# Literal entity-mention counting
# --------------------------------------------------------------------------- #

#: Conservative surface lexicon for *literal* mention counting only. This is not
#: the annotation layer and makes no claim about a post's stance. Traditional and
#: simplified Chinese forms are both included because the corpus mixes scripts.
ENTITY_LEXICON: dict[str, tuple[str, ...]] = {
    "NVDA": ("NVDA", "NVIDIA", "英伟达", "英偉達"),
    "AAPL": ("AAPL", "Apple", "苹果", "蘋果"),
    "MSFT": ("MSFT", "Microsoft", "微软", "微軟"),
    "GOOGL": ("GOOGL", "Google", "谷歌"),
    "META": ("META", "Meta", "Facebook"),
    "AMZN": ("AMZN", "Amazon", "亚马逊", "亞馬遜"),
    "TSLA": ("TSLA", "Tesla", "特斯拉"),
    "AMD": ("AMD", "超威", "超微"),
    "TSM": ("TSM", "TSMC", "台积电", "台積電"),
    "AVGO": ("AVGO", "Broadcom", "博通"),
    "MU": ("MU", "Micron", "美光"),
    "PLTR": ("PLTR", "Palantir"),
    "BABA": ("BABA", "Alibaba", "阿里巴巴", "阿里", "千问", "千問", "Qwen"),
    "TENCENT": ("Tencent", "腾讯", "騰訊"),
    "XIAOMI": ("Xiaomi", "小米"),
    "XPEV": ("XPeng", "小鹏", "小鵬"),
    "TM": ("Toyota", "丰田", "豐田"),
    "BTC": ("BTC", "Bitcoin", "比特币", "比特幣"),
    "ETH": ("ETH", "Ethereum", "以太坊"),
}


def count_entity_mentions(texts: Sequence[str], lexicon: dict[str, tuple[str, ...]] | None = None) -> pd.DataFrame:
    """Count posts containing each ticker's literal surface aliases.

    Matching is a plain case-insensitive substring scan on cleaned text. It is
    intentionally naive: the purpose is to show how thin literal single-name
    coverage is, not to perform entity linking.

    Parameters
    ----------
    texts
        Cleaned post texts.
    lexicon
        Ticker -> surface-form tuple. Defaults to :data:`ENTITY_LEXICON`.

    Returns
    -------
    pandas.DataFrame
        Columns ``ticker``, ``aliases``, ``n_posts_mentioning``, ``pct_of_corpus``,
        sorted by descending mention count.
    """
    import re

    lex = ENTITY_LEXICON if lexicon is None else lexicon
    series = pd.Series(list(texts)).astype(str)
    n = len(series)
    rows = []
    for ticker, aliases in lex.items():
        pattern = "|".join(re.escape(a) for a in aliases)
        hits = series.str.contains(pattern, case=False, regex=True, na=False)
        rows.append(
            {
                "ticker": ticker,
                "aliases": ", ".join(aliases),
                "n_posts_mentioning": int(hits.sum()),
                "pct_of_corpus": round(100.0 * hits.mean(), 1) if n else 0.0,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["n_posts_mentioning", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )