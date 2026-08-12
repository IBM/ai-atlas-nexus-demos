"""
ai_risk_classifier.py
─────────────────────
Multi-label AI risk classifier using ai-atlas-nexus ontology or static taxonomy.

Architecture
────────────
Hybrid scorer combining two complementary signals:

  1. TF-IDF cosine similarity  (surface-level vocabulary overlap)
     The use-case is vectorised and compared against risk descriptions/anchors.
     Score = max cosine similarity across a risk's text anchors.

  2. Domain keyword matching  (semantic coverage for domain terminology)
     Keywords extracted from risk name/description/concern.
     The use-case is tokenised into words + bigrams, light-stemmed, and intersected
     with risk keywords.  Score = hits / normaliser (capped 1).

  Final score = alpha * tfidf + (1-alpha) * keyword_score

The classifier can be initialized with:
  - Risk objects from ai-atlas-nexus (live ontology, Option B)
  - Static taxonomy dict (legacy, Option A)
  - Custom taxonomy

Dependencies
────────────
    pip install scikit-learn numpy
    pip install ai-atlas-nexus  (optional, for nexus integration)

Usage
─────
    from ai_risk_classifier import AIRiskClassifier

    # Use IBM Risk Atlas from ai-atlas-nexus
    clf = AIRiskClassifier.from_nexus(taxonomy='ibm-risk-atlas')

    results = clf.identify_risks_from_usecases(
        usecases=["An LLM answers patient questions about medications."]
    )
    for r in results:
        print(r)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_STEM_SUFFIXES = ("ing", "tion", "ations", "ments", "ment", "ers", "ises",
                  "izes", "ises", "izes", "ised", "ized", "ed", "es", "s")

def _stem(word: str) -> str:
    """Very light suffix stripping — avoids dependency on NLTK."""
    for sfx in _STEM_SUFFIXES:
        if word.endswith(sfx) and len(word) - len(sfx) >= 3:
            return word[: -len(sfx)]
    return word


def _build_extraction_text(risk: Any) -> str:
    """
    Assemble text from a Risk object for corpus-level keyword extraction.
    Concatenates name, description, concern, descriptor fields.
    """
    parts = []

    if hasattr(risk, "name") and risk.name:
        parts.append(risk.name)

    if hasattr(risk, "description") and risk.description:
        parts.append(risk.description)

    if hasattr(risk, "concern") and risk.concern:
        parts.append(risk.concern)

    if hasattr(risk, "descriptor") and risk.descriptor:
        if isinstance(risk.descriptor, list):
            parts.extend(risk.descriptor)
        else:
            parts.append(str(risk.descriptor))

    return " ".join(str(p) for p in parts if p)


# Domain-generic words to filter only in compound phrases (bigrams).
# Single-word keywords survive here; they're already filtered by max_df in the TfidfVectorizer.
# These are extremely common generics that never distinguish one risk from another.
_DOMAIN_GENERIC_COMPOUND_WORDS = {
    'data', 'model', 'system',  # too generic even in bigrams
}


def _extract_keywords_corpus(
    texts: list[str],
    top_k: int = 20,
    max_df: float = 0.5,
    min_word_len: int = 3,
) -> list[list[str]]:
    """
    Extract keywords from a corpus of risk texts using TF-IDF scoring.

    Applies corpus-level filtering to eliminate generic terms:
    - Removes English stopwords (a, the, in, on, etc.)
    - Drops terms appearing in >max_df fraction of documents (e.g. "user", "system")
    - Filters domain-generic words ("model", "data", "system", etc.)
    - Ranks remaining terms by TF-IDF weight (frequent in one risk, rare elsewhere)

    Parameters
    ----------
    texts : list[str]
        Risk descriptions, one per element (aligned with returned keyword lists).
    top_k : int
        How many top TF-IDF keywords to keep per risk (default 20).
    max_df : float
        Ignore terms appearing in more than max_df fraction of documents (default 0.5).
        E.g. 0.5 drops any term in >50% of risks.
    min_word_len : int
        Ignore tokens shorter than this (default 3, filters "a", "ai", "or").

    Returns
    -------
    list[list[str]]
        Keywords per risk, aligned by index with `texts`.
        Each sublist is sorted by TF-IDF weight descending.
    """
    if not texts:
        return []

    # Fit TF-IDF vectorizer across all risk texts
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=max_df,
        min_df=1,
        ngram_range=(1, 2),
        token_pattern=rf"(?u)\b[a-zA-Z]{{{min_word_len},}}\b",
        lowercase=True,
        strip_accents="unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Extract top-k keywords per document based on TF-IDF weight
    result = []
    for row_idx in range(tfidf_matrix.shape[0]):
        # Get nonzero TF-IDF scores for this document
        scores = tfidf_matrix[row_idx].toarray().flatten()
        # Get indices sorted by score descending
        sorted_indices = np.argsort(-scores)
        # Take top-k nonzero features, filtering only the most generic compound phrases
        top_features = []
        for idx in sorted_indices:
            if scores[idx] > 0 and len(top_features) < top_k:
                feature = feature_names[idx]
                # Filter compound phrases containing extremely generic terms.
                # Single-word keywords already handled by max_df and stopwords.
                if ' ' in feature:
                    words_in_feature = feature.split()
                    if any(w in _DOMAIN_GENERIC_COMPOUND_WORDS for w in words_in_feature):
                        continue
                top_features.append(feature)
        result.append(top_features)

    return result


def _tokenize(text: str) -> set[str]:
    """
    Return a set of unigrams + bigrams (original + light-stemmed).
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    unigrams = set(words)
    bigrams  = {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}
    stemmed  = {_stem(w) for w in words}
    return unigrams | bigrams | stemmed


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IdentifiedRisk:
    id: str
    label: str
    group: str
    sources: List[str]
    score: float
    tfidf_score: float
    keyword_score: float
    relevance: str
    matched_keywords: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        bar = "█" * int(self.score * 30)
        kw = f"  kw=[{', '.join(self.matched_keywords[:4])}]" if self.matched_keywords else ""
        return (
            f"[{self.relevance.upper():6}] {self.label:<42} "
            f"score={self.score:.3f}  {bar}{kw}"
        )


@dataclass
class UsecaseRiskResult:
    usecase: str
    risks: List[IdentifiedRisk]

    @property
    def high(self) -> List[IdentifiedRisk]:
        return [r for r in self.risks if r.relevance == "high"]

    @property
    def medium(self) -> List[IdentifiedRisk]:
        return [r for r in self.risks if r.relevance == "medium"]

    @property
    def low(self) -> List[IdentifiedRisk]:
        return [r for r in self.risks if r.relevance == "low"]

    def to_dict(self) -> dict:
        return {
            "usecase": self.usecase,
            "risks": [
                {"id": r.id, "label": r.label, "group": r.group,
                 "relevance": r.relevance, "score": r.score,
                 "matched_keywords": r.matched_keywords, "sources": r.sources}
                for r in self.risks
            ],
        }

    def __repr__(self) -> str:
        snippet = self.usecase[:90] + ("…" if len(self.usecase) > 90 else "")
        lines = [f"\nUse case: {snippet}", f"Identified {len(self.risks)} risk(s):"]
        for r in self.risks:
            lines.append(f"  {r}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class AIRiskClassifier:
    """
    Multi-label AI risk classifier (TF-IDF + keyword hybrid).
    Works with Risk objects from ai-atlas-nexus or custom risk taxonomy dicts.

    Parameters
    ----------
    risks : list[Any]
        Risk objects (from ai-atlas-nexus) or dicts (custom taxonomy).
        Required parameter. No default taxonomy provided.
    threshold_high : float
        Score ≥ this → "high" relevance.  Default 0.30.
    threshold_medium : float
        Score ≥ this → "medium" relevance.  Default 0.12.
    alpha : float
        Weight of TF-IDF component (0–1).  Default 0.35.
    aggregation : {"max", "mean"}
        How to pool per-anchor TF-IDF scores.  Default "max".
    keyword_top_k : int
        Max keywords per risk after corpus-level extraction (default 20).
    keyword_max_df : float
        Drop terms appearing in >max_df fraction of risks (default 0.5).
    keyword_min_word_len : int
        Ignore tokens shorter than this (default 3).

    Examples
    --------
    # Use IBM Risk Atlas from ai-atlas-nexus
    clf = AIRiskClassifier.from_nexus(taxonomy='ibm-risk-atlas')

    # Use custom risk taxonomy
    my_risks = [{
        'id': 'my-risk',
        'label': 'My Risk',
        'group': 'custom',
        'sources': [],
        'anchors': ['risk description'],
        'keywords': ['risk', 'keyword'],
        'kw_norm': 2,
    }]
    clf = AIRiskClassifier(risks=my_risks)
    """

    def __init__(
        self,
        risks:            list,
        threshold_high:   float = 0.30,
        threshold_medium: float = 0.12,
        alpha:            float = 0.35,
        aggregation:      str = "max",
        keyword_top_k:    int = 20,
        keyword_max_df:   float = 0.5,
        keyword_min_word_len: int = 3,
    ):
        if not risks:
            raise ValueError(
                "risks parameter is required. Either:\n"
                "  1. Use AIRiskClassifier.from_nexus(taxonomy='ibm-risk-atlas')\n"
                "  2. Pass a custom risk taxonomy dict: AIRiskClassifier(risks=[...])"
            )

        self.threshold_high   = threshold_high
        self.threshold_medium = threshold_medium
        self.alpha            = alpha
        self.aggregation      = aggregation
        self.keyword_top_k    = keyword_top_k
        self.keyword_max_df   = keyword_max_df
        self.keyword_min_word_len = keyword_min_word_len

        # Normalize Risk objects → internal dict format
        raw_risks = risks

        # Normalize Risk objects → internal dict format
        self.taxonomy = [self._risk_to_dict(r) for r in raw_risks]

        # Extract keywords for auto-extracted risks (corpus-level, with stopword/max_df filtering)
        self._auto_extract_keywords()

        # Build TF-IDF over all anchor sentences
        corpus: list[str] = []
        self._anchor_slices: list[slice] = []
        idx = 0
        for risk in self.taxonomy:
            n = len(risk["anchors"])
            self._anchor_slices.append(slice(idx, idx + n))
            corpus.extend(risk["anchors"])
            idx += n

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1,
            analyzer="word", strip_accents="unicode",
        )
        self._anchor_matrix = self._vectorizer.fit_transform(corpus)

        # Pre-tokenize keyword sets for all risks
        self._precompute_keywords()

    def _auto_extract_keywords(self):
        """
        Extract keywords for all auto-extracted risks using corpus-level TF-IDF.
        Applies to risks without hand-curated keywords (Risk objects from nexus).
        Static taxonomy risks (already with keywords) are skipped.
        """
        pending = [r for r in self.taxonomy if "keywords" not in r]
        if not pending:
            return

        texts = [r["_extraction_text"] for r in pending]
        keyword_lists = _extract_keywords_corpus(
            texts,
            top_k=self.keyword_top_k,
            max_df=self.keyword_max_df,
            min_word_len=self.keyword_min_word_len,
        )

        for risk, kws in zip(pending, keyword_lists):
            risk["keywords"] = kws
            risk["kw_norm"] = max(3, len(kws) // 2) if kws else 1
            risk.pop("_extraction_text", None)

    def _precompute_keywords(self):
        """Pre-tokenize keyword sets once to speed up scoring."""
        for _risk in self.taxonomy:
            _kw_tokens: set[str] = set()
            for kw in _risk["keywords"]:
                _kw_tokens.update(_tokenize(kw))
            _risk["_kw_tokens"] = _kw_tokens

    @classmethod
    def from_nexus(
        cls,
        taxonomy: Optional[str] = "ibm-risk-atlas",
        threshold_high: float = 0.30,
        threshold_medium: float = 0.12,
        alpha: float = 0.35,
        aggregation: str = "max",
        keyword_top_k: int = 20,
        keyword_max_df: float = 0.5,
        keyword_min_word_len: int = 3,
    ):
        """
        Initialize classifier with risks from ai-atlas-nexus library.

        Parameters
        ----------
        taxonomy : str | None
            Taxonomy name to filter risks (e.g., 'ibm-risk-atlas', 'nist-ai-rmf').
            Pass None to load all risks.
        threshold_high, threshold_medium, alpha, aggregation : float, str
            Same as __init__.
        keyword_top_k : int
            Maximum keywords per risk after corpus-level TF-IDF extraction (default 20).
        keyword_max_df : float
            Drop terms appearing in >max_df fraction of risks (default 0.5).
            E.g. 0.5 drops terms in >50% of risks.
        keyword_min_word_len : int
            Ignore tokens shorter than this (default 3, filters "a", "ai", "or").

        Returns
        -------
        AIRiskClassifier
            Classifier initialized with nexus risks.

        Raises
        ------
        ImportError
            If ai_atlas_nexus is not installed.
        """
        try:
            from ai_atlas_nexus import AIAtlasNexus
        except ImportError:
            raise ImportError(
                "ai-atlas-nexus is required for from_nexus(). "
                "Install with: pip install ai-atlas-nexus"
            )

        nexus = AIAtlasNexus()
        risks = nexus.get_all_risks(taxonomy=taxonomy)

        return cls(
            threshold_high=threshold_high,
            threshold_medium=threshold_medium,
            alpha=alpha,
            risks=risks,
            aggregation=aggregation,
            keyword_top_k=keyword_top_k,
            keyword_max_df=keyword_max_df,
            keyword_min_word_len=keyword_min_word_len,
        )

    @staticmethod
    def _risk_to_dict(risk: Any) -> dict:
        """
        Convert a Risk object from ai-atlas-nexus to dict format for classifier.

        Parameters
        ----------
        risk : Risk or dict
            Risk object from ai-atlas-nexus or dict (passthrough).

        Returns
        -------
        dict
            Risk dict with keys: id, label, group, sources, anchors, keywords, kw_norm.
        """
        # If already a dict, return as-is
        if isinstance(risk, dict):
            return risk

        # Risk object from ai-atlas-nexus
        # Extract text anchors: description + concern + any related text
        anchors = []
        if hasattr(risk, "description") and risk.description:
            anchors.append(risk.description)
        if hasattr(risk, "concern") and risk.concern:
            anchors.append(risk.concern)
        if hasattr(risk, "name") and risk.name:
            anchors.append(risk.name)

        # Determine group/category from risk_type or taxonomy
        group = getattr(risk, "risk_type", "uncategorized")
        if not group:
            group = "uncategorized"

        # Get sources from related metadata
        sources = []
        if hasattr(risk, "isDefinedByTaxonomy") and risk.isDefinedByTaxonomy:
            sources.append(risk.isDefinedByTaxonomy)

        # For auto-extracted risks (no hand-curated keywords), defer extraction to corpus level
        # Store raw text and mark for batch processing
        return {
            "id": risk.tag or risk.id,
            "label": risk.name or risk.tag,
            "group": group,
            "sources": sources,
            "anchors": anchors if anchors else [risk.name or ""],
            "_extraction_text": _build_extraction_text(risk),
            "kw_norm": None,
        }

    # ------------------------------------------------------------------

    def _keyword_scores(
        self, text: str
    ) -> tuple[np.ndarray, list[list[str]]]:
        tokens = _tokenize(text)
        scores  = np.zeros(len(self.taxonomy))
        matches: list[list[str]] = []
        for j, risk in enumerate(self.taxonomy):
            kw_tokens = risk["_kw_tokens"]
            norm      = risk.get("kw_norm", 5)
            hits      = kw_tokens & tokens
            scores[j] = min(len(hits) / norm, 1.0)
            # Report original keyword strings that fired
            fired = [kw for kw in risk["keywords"]
                     if _tokenize(kw) & tokens]
            matches.append(fired)
        return scores, matches

    # ------------------------------------------------------------------

    def identify_risks_from_usecases(
        self, usecases: List[str]
    ) -> List[UsecaseRiskResult]:
        """
        Identify applicable risks for one or more AI use-case descriptions.

        Parameters
        ----------
        usecases : list of str
            Plain-text descriptions of AI system use cases.

        Returns
        -------
        list of UsecaseRiskResult
        """
        uc_mat   = self._vectorizer.transform(usecases)
        sim_full = cosine_similarity(uc_mat, self._anchor_matrix)

        results = []
        for i, uc_text in enumerate(usecases):
            tfidf = np.zeros(len(self.taxonomy))
            for j, sl in enumerate(self._anchor_slices):
                sims = sim_full[i, sl]
                tfidf[j] = sims.max() if self.aggregation == "max" else sims.mean()

            kw, kw_matches = self._keyword_scores(uc_text)
            combined = self.alpha * tfidf + (1 - self.alpha) * kw

            identified = []
            for j, risk in enumerate(self.taxonomy):
                s = float(combined[j])
                if s >= self.threshold_medium:
                    relevance = "high" if s >= self.threshold_high else "medium"
                    identified.append(IdentifiedRisk(
                        id=risk["id"], label=risk["label"],
                        group=risk["group"], sources=risk["sources"],
                        score=round(s, 4),
                        tfidf_score=round(float(tfidf[j]), 4),
                        keyword_score=round(float(kw[j]), 4),
                        relevance=relevance,
                        matched_keywords=kw_matches[j],
                    ))

            identified.sort(key=lambda r: r.score, reverse=True)
            results.append(UsecaseRiskResult(usecase=uc_text, risks=identified))

        return results

    # ------------------------------------------------------------------

    def score(self, usecases: List[str]) -> np.ndarray:
        """
        Return raw score matrix (n_usecases, n_risks).
        Compatible with sklearn pipelines.
        """
        uc_mat   = self._vectorizer.transform(usecases)
        sim_full = cosine_similarity(uc_mat, self._anchor_matrix)

        tfidf = np.zeros((len(usecases), len(self.taxonomy)))
        for j, sl in enumerate(self._anchor_slices):
            col = sim_full[:, sl]
            tfidf[:, j] = col.max(axis=1) if self.aggregation == "max" else col.mean(axis=1)

        kw = np.zeros_like(tfidf)
        for i, uc in enumerate(usecases):
            kw[i], _ = self._keyword_scores(uc)

        return self.alpha * tfidf + (1 - self.alpha) * kw

    def risk_ids(self) -> List[str]:
        return [r["id"] for r in self.taxonomy]

    def __repr__(self) -> str:
        return (
            f"AIRiskClassifier(n_risks={len(self.taxonomy)}, "
            f"vocab={len(self._vectorizer.vocabulary_)}, "
            f"alpha={self.alpha}, "
            f"thresholds=({self.threshold_medium}, {self.threshold_high}))"
        )
