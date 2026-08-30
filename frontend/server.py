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

from starter.agent import Agent


STATIC_DIR = Path(__file__).resolve().parent / "static"
CATALOG_PATH = ROOT / "data" / "catalog.jsonl"
RESULTS_PATH = ROOT / "results.json"
HOST = "127.0.0.1"
PORT = 8000


def load_metrics() -> dict[str, Any]:
    if not RESULTS_PATH.exists():
        return {"available": False}
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {
        "available": True,
        "sample_count": data.get("sample_count"),
        "hit_rate_at_10": data.get("hit_rate_at_10"),
        "mrr": data.get("mrr"),
        "mttc": data.get("mttc"),
        "technical_score": data.get("recommended_technical_score"),
        "scenario_metrics": data.get("scenario_metrics", {}),
    }


class App:
    def __init__(self) -> None:
        self.agent = Agent(CATALOG_PATH)

    def reset(self, session_id: str | None = None) -> str:
        identifier = session_id or uuid.uuid4().hex
        self.agent.reset(identifier, {})
        return identifier

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
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
            product = self.agent.catalog_by_asin.get(parent_asin, {})
            categories = product.get("categories") or []
            recommendations.append({
                **recommendation,
                "title": product.get("title") or "Untitled product",
                "store": product.get("store") or "Unknown store",
                "category": categories[-1] if categories else "Uncategorized",
                "price": product.get("price"),
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
    def log_message(self, format: str, *args: object) -> None:
        print(f"[frontend] {format % args}")

    def send_json(self, status: int, value: Any) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
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
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/reset":
                self.send_json(200, {"session_id": APP.reset()})
            elif self.path == "/api/chat":
                self.send_json(200, APP.chat(payload))
            else:
                self.send_error(404)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": f"Agent error: {error}"})


def main() -> None:
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
