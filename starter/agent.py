"""Local BM25 shopping agent with persistent dialogue state plumbing."""

from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from pathlib import Path

from dialogue.accumulator import accumulate_information
from dialogue.intent_detector import detect_intent
from dialogue.slot_extractor import extract_slots
from dialogue.state import SessionState
from src.ranking.reranker import Reranker, to_evaluator_recommendations


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    """Flatten a catalog field into searchable text."""

    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    """Extract unique-query candidates after basic stop-word filtering."""

    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


# STUB (owned by role B, replace with the real ask policy). Cycling through a few
# attributes is enough to stop the simulated customer returning content-free replies,
# which is what the retrieval work needs in order to be measurable at all.
STUB_ASK_CYCLE = ("feature", "material", "color")

# Per-field BM25 weights, in the column order declared in _build_index():
# parent_asin, title, categories, features, details, store, description.
# These are still TechJam's defaults; tuning them is retrieval work (role A).
FIELD_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# Cap on how many distinct terms are sent to FTS5 in one query.
MAX_QUERY_TERMS = 40

# Default candidate pool size handed downstream to reranking (role C).
# recall@500 is 1.000 on the public dev set, so 500 loses nothing.
DEFAULT_CANDIDATES = 500


class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        """Build the catalog index and initialize the session registry."""

        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.sessions: dict[str, SessionState] = {}
        self.catalog_by_asin: dict[str, dict] = {}
        self.reranker = Reranker()
        self._build_index()

    def _build_index(self) -> None:
        """Load catalog records into the in-memory FTS5 index."""

        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self.catalog_by_asin[parent_asin] = product
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Replace a session with fresh state and a copied user profile."""

        # The profile is anonymized and may be used for personalization.
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            user_profile=deepcopy(user_profile),
        )

    def retrieve(self, query: str, n: int = DEFAULT_CANDIDATES) -> list[tuple[str, float]]:
        """Return up to `n` (parent_asin, score) candidates for `query`, best first.

        Scores are negated SQLite bm25() values so that higher is better, which is
        the convention downstream reranking expects. Returns [] for an empty query.
        """
        terms = list(dict.fromkeys(_terms(query)))[:MAX_QUERY_TERMS]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        weights = ", ".join(str(weight) for weight in FIELD_WEIGHTS)
        rows = self.connection.execute(
            f"SELECT parent_asin, -bm25(products, {weights}) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score DESC LIMIT ?",
            (expression, n),
        ).fetchall()
        return [(str(row[0]), float(row[1])) for row in rows]

    def ranking_candidates(
        self,
        retrieved: list[tuple[str, float]],
    ) -> list[dict]:
        """Hydrate retrieval IDs with catalog metadata for downstream ranking."""

        candidates = []
        for parent_asin, score in retrieved:
            product = self.catalog_by_asin.get(parent_asin)
            if product is None:
                continue
            candidates.append({
                **product,
                "parent_asin": parent_asin,
                "bm25_score": score,
            })
        return candidates

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        """Return baseline recommendations and commit state after success."""

        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        extraction = extract_slots(user_message)
        detected_intent = detect_intent(user_message, state, extraction)
        ranking_state = deepcopy(state)
        accumulate_information(ranking_state, extraction)
        ranking_state.intent = detected_intent
        # Step 7 - build the query from every customer turn so far, including this one,
        # so constraints revealed earlier in the session still influence retrieval.
        # Nothing is written to state yet: if retrieval raises, the turn must leave the
        # session untouched. Switching this to state.revealed_text is a separate change.
        prior_messages = [
            entry["content"] for entry in state.message_history if entry["role"] == "user"
        ]
        query = " ".join([*prior_messages, user_message])
        # Step 8 - retrieve a recall-oriented pool, hydrate its catalog metadata,
        # and let ranking choose the final evaluator-facing top_k.
        retrieved = self.retrieve(query)
        candidates = self.ranking_candidates(retrieved)
        ranked = self.reranker.rerank(candidates, ranking_state, top_k=top_k)
        recommendations = to_evaluator_recommendations(ranked)
        response = {
            "message": "Here are the closest matches I found.",
            "ask_attribute": STUB_ASK_CYCLE[(turn - 1) % len(STUB_ASK_CYCLE)],
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        state.turn = turn
        state.message_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response["message"]},
        ])
        accumulate_information(state, extraction)
        state.intent = detected_intent
        return response
