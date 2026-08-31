"""Local console for driving the agent by hand.

Not part of scoring, and the evaluator never touches it. It exists because aggregate
metrics hide how a conversation actually unfolds: this shows what the agent knows
after each turn, which question it chose and why the ranking changed, which is how
several defects in the dialogue layer were found.

Standard library only, single-threaded, and bound to localhost.
"""

from __future__ import annotations

import json
import mimetypes
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import evaluate, load_jsonl
from starter.agent import Agent


STATIC_DIR = Path(__file__).resolve().parent / "static"
CATALOG_PATH = ROOT / "data" / "catalog.jsonl"
RESULTS_PATH = ROOT / "results.json"
PUBLIC_SET_PATH = ROOT / "data" / "public_set.jsonl"
HOST = "127.0.0.1"
PORT = 8000


def metrics_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full evaluator result to the fields the page displays."""

    return {
        "available": True,
        "sample_count": data.get("sample_count"),
        "hit_rate_at_10": data.get("hit_rate_at_10"),
        "mrr": data.get("mrr"),
        "mttc": data.get("mttc"),
        "technical_score": data.get("recommended_technical_score"),
        "scenario_metrics": data.get("scenario_metrics", {}),
    }


def load_metrics() -> dict[str, Any]:
    """Read the last saved evaluation, or report that none exists yet."""

    if not RESULTS_PATH.exists():
        return {"available": False}
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return metrics_payload(data)


class App:
    """Holds the one live Agent and the catalog lookups the page needs."""

    def __init__(self) -> None:
        """Build the agent and index the catalog once, at startup."""

        self.agent = Agent(CATALOG_PATH)
        self.catalog_by_asin = {}
        self.categories_by_asin = {}
        with CATALOG_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self.catalog_by_asin[parent_asin] = product
                self.categories_by_asin[parent_asin] = [
                    str(value) for value in product.get("categories") or []
                ]
        self.catalog_ids = set(self.catalog_by_asin)
        self.public_samples = load_jsonl(PUBLIC_SET_PATH)

    def reset(self, session_id: str | None = None) -> str:
        """Start a fresh conversation and return its id."""

        identifier = session_id or uuid.uuid4().hex
        self.agent.reset(identifier, {})
        return identifier

    def evaluate_public_set(self) -> dict[str, Any]:
        """Evaluate the current agent and atomically publish frontend metrics."""

        preserved_sessions = set(self.agent.sessions)
        try:
            result = evaluate(
                self.agent,
                self.public_samples,
                self.catalog_ids,
                self.categories_by_asin,
                self.catalog_by_asin,
            )
        finally:
            for session_id in set(self.agent.sessions) - preserved_sessions:
                self.agent.sessions.pop(session_id, None)

        temporary_path = RESULTS_PATH.with_name(f".{RESULTS_PATH.name}.tmp")
        temporary_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(RESULTS_PATH)
        return metrics_payload(result)

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one turn and return the agent's reply plus its current state.

        Raises ValueError with a message meant for the page when the session is
        unknown, the message is empty, or the turn is outside the ten-turn limit.
        """

        session_id = str(payload.get("session_id", "")).strip()
        message = str(payload.get("message", "")).strip()
        turn = payload.get("turn", 1)
        if not session_id or session_id not in self.agent.sessions:
            raise ValueError("Start a new session before sending a message.")
        if not message:
            raise ValueError("Message cannot be empty.")
        if not isinstance(turn, int) or not 1 <= turn <= 10:
            raise ValueError("Turn must be an integer from 1 to 10.")

        response = self.agent.respond(session_id, message, turn, top_k=10)
        state = self.agent.sessions[session_id]
        recommendations = []
        for recommendation in response["recommendations"]:
            parent_asin = recommendation["parent_asin"]
            product = self.catalog_by_asin.get(parent_asin, {})
            categories = product.get("categories") or []
            raw_features = product.get("features") or []
            feature_values = raw_features if isinstance(raw_features, list) else [raw_features]
            features = [str(value).strip() for value in feature_values if str(value).strip()]
            raw_details = product.get("details") or {}
            details = (
                [
                    f"{key}: {value}"
                    for key, value in raw_details.items()
                    if value not in (None, "", [])
                ]
                if isinstance(raw_details, dict)
                else [str(value).strip() for value in (
                    raw_details if isinstance(raw_details, list) else [raw_details]
                ) if str(value).strip()]
            )
            highlights = [*features[:2], *details[:1]][:3]
            recommendations.append({
                **recommendation,
                "title": product.get("title") or "Untitled product",
                "store": product.get("store") or "Unknown store",
                "category": categories[-1] if categories else "Uncategorized",
                "price": product.get("price"),
                "highlights": highlights,
            })
        active_slots = {
            name: [str(value) for value in values]
            for name, values in state.slots.items()
            if values
        }
        return {
            **response,
            "recommendations": recommendations,
            "state": {
                "intent": state.intent,
                "turn": state.turn,
                "slots": active_slots,
                "hard_constraints": sorted(state.hard_constraints),
                "soft_preferences": sorted(state.soft_preferences),
            },
        }


APP: App


class Handler(BaseHTTPRequestHandler):
    """Serves the static page and the small JSON API behind it."""

    def log_message(self, format: str, *args: object) -> None:
        """Prefix request logs so they are distinguishable from agent output."""

        print(f"[frontend] {format % args}")

    def send_json(self, status: int, value: Any) -> None:
        """Write a JSON response, marked no-store so the page never shows stale metrics."""

        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve the last saved metrics, or a static file from the page directory."""

        path = urlparse(self.path).path
        if path == "/api/metrics":
            self.send_json(200, load_metrics())
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in target.parents or not target.is_file():
            self.send_error(404)
            return
        content = target.read_bytes()
        mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        """Handle session reset, a chat turn, or a full re-evaluation of the agent."""

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/reset":
                self.send_json(200, {"session_id": APP.reset()})
            elif self.path == "/api/evaluate":
                self.send_json(200, APP.evaluate_public_set())
            elif self.path == "/api/chat":
                self.send_json(200, APP.chat(payload))
            else:
                self.send_error(404)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": f"Agent error: {error}"})


def main() -> None:
    """Build the agent, then serve until interrupted."""

    global APP
    if not CATALOG_PATH.exists():
        raise SystemExit(f"Catalog not found: {CATALOG_PATH}")
    print("Loading the product catalog and building the search index...")
    APP = App()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Frontend ready at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        APP.agent.connection.close()
        server.server_close()


if __name__ == "__main__":
    main()
