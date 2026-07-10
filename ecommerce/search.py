"""Product search: full-text, faceted, typo-tolerant, relevance-ranked.

The engine builds an in-memory inverted index over product text once, then
answers queries by intersecting posting lists — O(matching documents), not
O(catalog) — which keeps latency well under the 200 ms budget even at
100,000 SKUs (see ``tests/ecommerce/test_search.py`` for the timing check).

Ranking blends a text-match score (how well the query terms hit the title vs
the body) with business signals (product popularity), so a strong seller
surfaces above an incidental keyword match without letting popularity
override a clearly better textual match.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .catalog import Catalog, Product

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def _trigrams(term: str) -> Set[str]:
    padded = f"  {term} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _edit_distance_le(a: str, b: str, limit: int) -> bool:
    """True if Levenshtein(a, b) <= limit, computed with an early-exit band."""
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            best = min(best, cur[j])
        if best > limit:
            return False
        prev = cur
    return prev[-1] <= limit


@dataclass
class SearchHit:
    product_id: str
    score: float


@dataclass
class Facet:
    """One facetable attribute and the counts for each of its values."""

    attribute: str
    values: Dict[str, int] = field(default_factory=dict)


@dataclass
class SearchResults:
    hits: List[SearchHit]
    facets: List[Facet]
    total: int
    took_ms: float
    corrected_terms: Dict[str, str] = field(default_factory=dict)


# Weight a title match far above a body/tag match, mirroring shopper intent.
_FIELD_WEIGHT = {"title": 6.0, "tags": 3.0, "attrs": 2.0, "body": 1.0}


class SearchIndex:
    """Inverted index with typo tolerance, synonyms, and relevance ranking."""

    def __init__(
        self,
        catalog: Catalog,
        *,
        synonyms: Optional[Dict[str, Sequence[str]]] = None,
        facet_attributes: Sequence[str] = (),
    ):
        self.catalog = catalog
        self.facet_attributes = list(facet_attributes)
        # term -> {product_id -> weighted term frequency}
        self._postings: Dict[str, Dict[str, float]] = defaultdict(dict)
        # trigram -> set(term) for typo-tolerant candidate lookup
        self._trigram: Dict[str, Set[str]] = defaultdict(set)
        self._vocab: Set[str] = set()
        # bidirectional synonym expansion (query "tv" -> {"tv","television"})
        self._synonyms: Dict[str, Set[str]] = defaultdict(set)
        for group in (synonyms or {}).values():
            words = [w.lower() for w in group]
            for w in words:
                self._synonyms[w].update(x for x in words if x != w)
        # Caches populated at build time so the query hot path stays tight.
        self._doc_count = 0
        self._active: Set[str] = set()  # product ids with sellable variants
        self._boost: Dict[str, float] = {}  # product id -> popularity multiplier
        self._idf_cache: Dict[str, float] = {}
        # Precomputed facet values: attribute -> {product_id -> value}, so
        # counting facets over a large result set is pure dict lookups.
        self._facet_index: Dict[str, Dict[str, str]] = {
            attr: {} for attr in self.facet_attributes
        }
        self._build()

    def _index_field(self, product_id: str, text: str, weight: float) -> None:
        for term in tokenize(text):
            self._postings[term][product_id] = (
                self._postings[term].get(product_id, 0.0) + weight
            )
            if term not in self._vocab:
                self._vocab.add(term)
                for tri in _trigrams(term):
                    self._trigram[tri].add(term)

    def _build(self) -> None:
        import math

        count = 0
        for product in self.catalog.products():
            count += 1
            if not product.active_variants():
                continue  # nothing sellable -> keep it out of results
            pid = product.id
            self._active.add(pid)
            # Bounded popularity boost, precomputed once per product.
            self._boost[pid] = 1.0 + min(0.5, math.log1p(product.popularity) / 10.0)
            for attr, index in self._facet_index.items():
                value = product.attributes.get(attr)
                if value is not None:
                    index[pid] = value
            self._index_field(pid, product.title, _FIELD_WEIGHT["title"])
            self._index_field(pid, " ".join(product.tags), _FIELD_WEIGHT["tags"])
            self._index_field(
                pid, " ".join(product.attributes.values()), _FIELD_WEIGHT["attrs"]
            )
            self._index_field(pid, product.description, _FIELD_WEIGHT["body"])
        self._doc_count = max(1, count)

    # ---- typo tolerance --------------------------------------------------
    def _correct(self, term: str) -> Optional[str]:
        """Return the closest in-vocabulary term within an edit budget.

        Candidates are gathered by shared trigrams (cheap) and then filtered
        by a bounded edit distance. The budget scales with word length so
        short words demand near-exact matches.
        """
        if term in self._vocab:
            return term
        budget = 1 if len(term) <= 5 else 2
        candidates: Set[str] = set()
        for tri in _trigrams(term):
            candidates |= self._trigram.get(tri, set())
        best: Optional[str] = None
        best_key: Tuple[int, int] = (budget + 1, 0)
        for cand in candidates:
            if not _edit_distance_le(term, cand, budget):
                continue
            # Prefer smaller edit distance, then the more common term.
            dist = 0 if term == cand else _levenshtein(term, cand)
            popularity = len(self._postings.get(cand, {}))
            key = (dist, -popularity)
            if key < best_key:
                best_key, best = key, cand
        return best

    def _expand(self, term: str) -> Set[str]:
        return {term} | self._synonyms.get(term, set())

    # ---- query -----------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, str]] = None,
        category: Optional[str] = None,
        limit: int = 24,
    ) -> SearchResults:
        start = time.perf_counter()
        raw_terms = tokenize(query)
        corrected: Dict[str, str] = {}
        scores: Dict[str, float] = defaultdict(float)

        # Restrict to a category subtree if requested.
        allowed: Optional[Set[str]] = None
        if category:
            subtree = set(self.catalog.descendants(category))
            allowed = {
                p.id for p in self.catalog.products() if p.category in subtree
            }

        matched_any = not raw_terms  # empty query -> browse mode (match all)
        for term in raw_terms:
            fixed = self._correct(term)
            if fixed and fixed != term:
                corrected[term] = fixed
            search_term = fixed or term
            # Union the postings of the term and all its synonyms.
            for variant in self._expand(search_term):
                postings = self._postings.get(variant)
                if not postings:
                    continue
                matched_any = True
                idf = self._idf(variant)
                for pid, tf in postings.items():
                    scores[pid] += tf * idf

        if not matched_any:
            return SearchResults([], [], 0, self._ms(start), corrected)

        # Browse mode with no terms: seed every sellable product.
        if not raw_terms:
            for product in self.catalog.products():
                if product.active_variants():
                    scores[product.id] = 0.0

        # ---- filtering (attribute facets + category) --------------------
        # Hot path: rely on precomputed active set and popularity boost, and
        # only touch the Product object when a filter or category needs it.
        results: List[Tuple[str, float]] = []
        active = self._active
        boost = self._boost
        for pid, text_score in scores.items():
            if pid not in active:
                continue
            if allowed is not None and pid not in allowed:
                continue
            if filters:
                product = self.catalog.product(pid)
                if product is None or not self._matches(product, filters):
                    continue
            results.append((pid, text_score * boost.get(pid, 1.0)))

        total = len(results)
        results.sort(key=lambda x: (-x[1], x[0]))
        hits = [SearchHit(pid, round(score, 4)) for pid, score in results[:limit]]
        facets = self._facets(pid for pid, _ in results)
        return SearchResults(hits, facets, total, self._ms(start), corrected)

    # ---- scoring helpers -------------------------------------------------
    def _idf(self, term: str) -> float:
        """Inverse document frequency, memoised per term.

        Rare terms weigh more than common ones. Popularity boost (applied in
        the scoring loop) multiplies the text score by at most ~1.5x, so it
        breaks ties and lifts strong sellers but never lets a popular-yet-
        irrelevant product outrank a clearly better textual match.
        """
        cached = self._idf_cache.get(term)
        if cached is not None:
            return cached
        import math

        df = len(self._postings.get(term, {})) or 1
        value = math.log(1 + self._doc_count / df)
        self._idf_cache[term] = value
        return value

    def _matches(self, product: Product, filters: Dict[str, str]) -> bool:
        return all(product.attributes.get(k) == v for k, v in filters.items())

    def _facets(self, product_ids: Iterable[str]) -> List[Facet]:
        # Count from the precomputed facet index — no Product access per hit.
        ids = list(product_ids)
        out: List[Facet] = []
        for attr in self.facet_attributes:
            index = self._facet_index.get(attr, {})
            values: Dict[str, int] = {}
            for pid in ids:
                value = index.get(pid)
                if value is not None:
                    values[value] = values.get(value, 0) + 1
            if values:
                out.append(Facet(attr, values))
        return out

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            )
        prev = cur
    return prev[-1]
