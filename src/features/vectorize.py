from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Union, Optional, Dict, Any

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

TextInput = Union[Iterable[str], list[str]]


@dataclass
class TextVectorizer:
    """
    Minimal, MLOps-friendly TF-IDF wrapper.

    - Keeps config (max_features/ngram_range/stop_words) together with the fitted vectorizer.
    - Has a small API: fit / transform / fit_transform / save / load
    - No pandas dependency, no demo code, no feature-inspection utilities.
    """
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    stop_words: Optional[str] = "english"
    # Optional extra kwargs passed to TfidfVectorizer (e.g., min_df, max_df)
    extra_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        kwargs = dict(self.extra_kwargs or {})
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            **kwargs,
        )

    def fit(self, texts: TextInput) -> "TextVectorizer":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: TextInput) -> csr_matrix:
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: TextInput) -> csr_matrix:
        self.fit(texts)
        return self.vectorizer.fit_transform(texts)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save both the fitted vectorizer and its configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "config": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "stop_words": self.stop_words,
                "extra_kwargs": self.extra_kwargs or {},
            },
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TextVectorizer":
        """
        Load a fitted vectorizer and restore its configuration.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {path}")

        payload = joblib.load(path)
        cfg = payload.get("config", {}) or {}

        inst = cls(
            max_features=cfg.get("max_features", 5000),
            ngram_range=tuple(cfg.get("ngram_range", (1, 2))),
            stop_words=cfg.get("stop_words", "english"),
            extra_kwargs=cfg.get("extra_kwargs", {}) or {},
        )
        inst.vectorizer = payload["vectorizer"]
        return inst


def build_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: Optional[str] = "english",
    **kwargs: Any,
) -> TextVectorizer:
    """
    Convenience factory mirroring your notebook params.
    """
    return TextVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        extra_kwargs=kwargs or {},
    )
