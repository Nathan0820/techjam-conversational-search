"""Latent semantic index over the product catalog.

BM25 matches words. Two products can describe the same thing and share almost no
tokens -- "leather belt with a buckle" against "full grain cowhide strap, pin
closure" -- and lexical scoring rates that pair near zero. This module supplies the
complementary signal: a dense vector per product, where closeness reflects shared
context rather than shared characters.

The vectors come from truncated SVD over a TF-IDF matrix, i.e. latent semantic
analysis. With only a few hundred dimensions available, the factorisation cannot give
"leather" and "cowhide" separate axes; because they co-occur with the same
neighbours -- belt, strap, hide, tanned -- it is forced to place them on shared ones.
That compression is where the semantics come from. Nothing labels the dimensions and
they are not individually interpretable.

Chosen over a pretrained sentence encoder deliberately. `docs/submission_rules.md`
warns that final scoring may run with network access disabled, and a transformer
would fetch ~90MB of weights at construction time; if that fetch failed the agent
would raise before answering a single turn. This trains on the catalog already on
disk, needs no download, and keeps the "in-memory, light execution" constraint. The
cost is that it only knows vocabulary appearing in these 50,000 products: a paraphrase
using words the catalog never contains cannot be bridged, where a pretrained model
would manage it. That limitation is real and documented in the README.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Latent dimensions kept from the decomposition. 300 is the conventional choice for a
# corpus this size; too few collapses real distinctions, too many preserves noise.
# Left unswept on purpose -- every other parameter tuned today moved the score by less
# than the run-to-run noise band, and this one has no reason to behave differently.
DEFAULT_COMPONENTS = 300

# Terms appearing in only one product carry no co-occurrence information, and terms in
# more than half of them carry no discriminative power. Trimming both ends keeps the
# decomposition focused on vocabulary that actually relates products to each other.
MIN_DOCUMENT_FREQUENCY = 2
MAX_DOCUMENT_FREQUENCY = 0.5

# Truncated SVD is randomised; fixing the seed keeps successive evaluation runs
# comparable, which matters when the effects being measured are small.
RANDOM_SEED = 20260831


class LatentSemanticIndex:
    """Dense semantic vectors for a fixed catalog, queryable by free text."""

    def __init__(
        self,
        documents: Mapping[str, str],
        n_components: int = DEFAULT_COMPONENTS,
        random_state: int = RANDOM_SEED,
    ) -> None:
        """Fit the index over `documents`, a mapping of product id to searchable text."""

        self.ids: list[str] = list(documents)
        corpus = [documents[identifier] for identifier in self.ids]

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            min_df=MIN_DOCUMENT_FREQUENCY,
            max_df=MAX_DOCUMENT_FREQUENCY,
            sublinear_tf=True,
        )
        term_document = self._vectorizer.fit_transform(corpus)

        # Cannot ask for more components than the matrix has dimensions.
        components = min(n_components, min(term_document.shape) - 1)
        self._svd = TruncatedSVD(n_components=components, random_state=random_state)
        document_vectors = self._svd.fit_transform(term_document)

        # Pre-normalising to unit length turns cosine similarity into a plain dot
        # product at query time, which is the operation on the hot path.
        self._document_vectors = _unit_rows(document_vectors).astype(np.float32)
        self._position = {identifier: index for index, identifier in enumerate(self.ids)}

    @property
    def n_components(self) -> int:
        """Number of latent dimensions actually retained."""

        return int(self._svd.n_components)

    def _embed_query(self, query: str) -> np.ndarray | None:
        """Project free text into the latent space, or None if it shares no vocabulary."""

        sparse = self._vectorizer.transform([query])
        if sparse.nnz == 0:
            # Every term was unseen or trimmed; there is no position for this query.
            return None
        vector = self._svd.transform(sparse)[0]
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return None
        return (vector / norm).astype(np.float32)

    def score(self, query: str, parent_asins: Sequence[str]) -> dict[str, float]:
        """Return cosine similarity in [-1, 1] for each requested id.

        Scores only the ids given rather than the whole catalog: lexical retrieval has
        already chosen who competes, and this decides their order. Ids absent from the
        index, and queries sharing no vocabulary with it, yield an empty mapping so the
        caller can fall back cleanly.
        """

        embedded = self._embed_query(query)
        if embedded is None or not parent_asins:
            return {}

        rows: list[int] = []
        present: list[str] = []
        for parent_asin in parent_asins:
            index = self._position.get(parent_asin)
            if index is not None:
                rows.append(index)
                present.append(parent_asin)
        if not rows:
            return {}

        similarities = self._document_vectors[rows] @ embedded
        return {
            parent_asin: float(similarity)
            for parent_asin, similarity in zip(present, similarities)
        }

    def nearest(self, query: str, n: int = 500) -> list[tuple[str, float]]:
        """Return the `n` most similar products across the whole catalog, best first.

        Not used on the response path -- lexical recall@500 is already 1.000, so a
        second candidate source adds nothing -- but it is what makes this a retriever
        rather than only a scorer, and it is how the two routes get compared.
        """

        embedded = self._embed_query(query)
        if embedded is None or n <= 0:
            return []
        similarities = self._document_vectors @ embedded
        count = min(n, similarities.shape[0])
        top = np.argpartition(-similarities, count - 1)[:count]
        top = top[np.argsort(-similarities[top])]
        return [(self.ids[index], float(similarities[index])) for index in top]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows untouched."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_index(
    documents: Mapping[str, str],
    n_components: int = DEFAULT_COMPONENTS,
) -> LatentSemanticIndex | None:
    """Fit an index, returning None if that is not possible for any reason.

    The dense route is an enhancement, never a dependency. Anything that goes wrong
    here -- a missing optional package, an unusable catalog, a memory limit on the
    grading machine -- must leave the agent answering exactly as it does without it,
    rather than raising before the first turn.
    """

    try:
        if not documents:
            return None
        return LatentSemanticIndex(documents, n_components=n_components)
    except Exception:
        return None


def flatten_documents(products: Iterable[Mapping[str, object]]) -> dict[str, str]:
    """Build id -> searchable text, mirroring the fields the FTS5 index covers."""

    fields = ("title", "categories", "features", "details", "store", "description")
    documents: dict[str, str] = {}
    for product in products:
        parent_asin = str(product.get("parent_asin", "")).strip()
        if not parent_asin:
            continue
        parts: list[str] = []
        for field in fields:
            value = product.get(field)
            if value is None:
                continue
            if isinstance(value, dict):
                parts.extend(f"{key} {item}" for key, item in value.items())
            elif isinstance(value, (list, tuple)):
                parts.extend(str(item) for item in value)
            else:
                parts.append(str(value))
        documents[parent_asin] = " ".join(parts)
    return documents
