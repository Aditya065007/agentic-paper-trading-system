"""Local-only API for the Render-hosted paper-trading dashboard.

The Render site is static. Trading runs on this machine so broker/news/LLM keys never
need to be sent to Render. Bind only to loopback; use the printed pairing token.
"""
from __future__ import annotations

import hmac
import json
import math
import os
import re
import secrets
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Order and stop switches must be inherited from the process environment, not loaded
# from .env. Capture them before dotenv so a local .env can never accidentally enable trading.
_ORDER_GATE_AT_START = os.environ.get("ALPACA_ORDERS_ENABLED")
_STOPS_GATE_AT_START = os.environ.get("ALPACA_STOPS_ENABLED")
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
for _gate_name, _initial_value in (
    ("ALPACA_ORDERS_ENABLED", _ORDER_GATE_AT_START),
    ("ALPACA_STOPS_ENABLED", _STOPS_GATE_AT_START),
):
    if _initial_value is None:
        os.environ.pop(_gate_name, None)
    else:
        os.environ[_gate_name] = _initial_value
del _gate_name, _initial_value

from trading_system.crew import guarded_plan, journal, run_cycle
from trading_system.servers.tools import live_orders

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = PACKAGE_DIR / "run_reports"
STATE_DIR = Path(__file__).resolve().parent / ".local_state"
DEFAULT_PORT = 8765
KEY_NAMES = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "NEWSAPI_KEY", "GROQ_API_KEY")
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
RUN_LOCK = threading.RLock()
PAIRING_TOKEN = secrets.token_urlsafe(32)
KEYS_SET_IN_UI: set[str] = set()


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _origins() -> set[str]:
    raw = os.environ.get("TRADING_UI_ORIGINS", "")
    configured = {part.strip().rstrip("/") for part in raw.split(",") if part.strip()}
    configured.update({
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
        "http://localhost:8765", "http://127.0.0.1:8765",
    })
    return configured


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()
                if not any(word in str(k).lower() for word in ("api_key", "secret_key", "password", "token"))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _redact(value):
    if isinstance(value, str):
        result = value
        for name in KEY_NAMES:
            secret = os.environ.get(name, "").strip()
            if secret and len(secret) >= 4:
                result = result.replace(secret, "[REDACTED]")
        return result
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def safe_snapshot(result: dict) -> dict:
    """Return JSON-safe, secret-redacted cycle output."""
    return _redact(_json_safe(result if isinstance(result, dict) else {}))


def validate_symbols(raw) -> list[str]:
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, list):
        values = raw
    else:
        raise RequestError("Enter symbols as a comma-separated list.")
    symbols, seen = [], set()
    for item in values:
        if not isinstance(item, str):
            raise RequestError("Every stock symbol must be text.")
        symbol = item.strip().upper()
        if not symbol:
            continue
        if not SYMBOL_RE.fullmatch(symbol):
            raise RequestError(f"Invalid ticker symbol: {symbol[:16]}")
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    if not symbols:
        raise RequestError("Enter at least one stock symbol.")
    if len(symbols) > 15:
        raise RequestError("Use at most 15 symbols per cycle.")
    return symbols


def validate_cycle_request(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RequestError("Request body must be a JSON object.")
    symbols = validate_symbols(payload.get("symbols"))
    timeframe = payload.get("timeframe", "1Day")
    if timeframe not in ("1Day", "1Hour", "BOTH"):
        raise RequestError("Timeframe must be 1Day, 1Hour, or BOTH.")
    lookback = payload.get("lookback_days", 180)
    if isinstance(lookback, bool) or not isinstance(lookback, int) or not 30 <= lookback <= 730:
        raise RequestError("Lookback must be between 30 and 730 days.")
    return {
        "symbols": symbols,
        "timeframe": timeframe,
        "lookback_days": lookback,
        "advisory": bool(payload.get("advisory", False)),
        "debate_band": payload.get("debate_band", "spec"),
        "reflect": bool(payload.get("reflect", False)),
        "reflect_dry_run": bool(payload.get("reflect_dry_run", False)),
        # Approval is intentionally omitted: every cycle stops for human review.
    }


def key_status() -> dict:
    return {
        "keys": {name: {"configured": bool(os.environ.get(name, "").strip()),
                        "updated_in_ui": name in KEYS_SET_IN_UI}
                 for name in KEY_NAMES},
        "broker_mode": (os.environ.get("BROKER_MODE") or "offline").strip().lower(),
        "live_orders_enabled": live_orders.orders_enabled(),
        "paper_keys_ready": bool(os.environ.get("ALPACA_API_KEY", "").strip()
                                 and os.environ.get("ALPACA_SECRET_KEY", "").strip()),
    }


def update_local_settings(payload: dict) -> dict:
    """Update process memory only. No key value is returned or written to disk."""
    if not isinstance(payload, dict):
        raise RequestError("Settings must be a JSON object.")
    mode = payload.get("broker_mode")
    if mode is not None:
        if mode not in ("offline", "alpaca_paper"):
            raise RequestError("Broker mode must be offline or alpaca_paper.")
    with RUN_LOCK:
        for name in KEY_NAMES:
            if name not in payload:
                continue
            value = payload[name]
            if not isinstance(value, str) or len(value) > 4096:
                raise RequestError(f"{name} must be text under 4096 characters.")
            value = value.strip()
            if value:
                os.environ[name] = value
                KEYS_SET_IN_UI.add(name)
            else:
                os.environ.pop(name, None)
                KEYS_SET_IN_UI.discard(name)
        if mode is not None:
            os.environ["BROKER_MODE"] = mode
    return key_status()


def _runs_dir() -> Path:
    return STATE_DIR / "runs"


def _save_snapshot(snapshot: dict) -> None:
    _runs_dir().mkdir(parents=True, exist_ok=True)
    path = _runs_dir() / f"{snapshot['snapshot_id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(safe_snapshot(snapshot), indent=2, ensure_ascii=False),
                         encoding="utf-8")
    temporary.replace(path)


def _load_snapshot(snapshot_id: str) -> dict:
    if not re.fullmatch(r"[a-f0-9-]{36}", str(snapshot_id)):
        raise RequestError("Invalid run snapshot id.", 404)
    path = _runs_dir() / f"{snapshot_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RequestError("Run snapshot was not found on this machine.", 404) from None
    if not isinstance(value, dict) or value.get("snapshot_id") != snapshot_id:
        raise RequestError("Run snapshot was not found on this machine.", 404)
    return value


def execute_cycle(payload: dict) -> dict:
    config = validate_cycle_request(payload)
    with RUN_LOCK:
        result = run_cycle.run_cycle_sync(config)
        snapshot = safe_snapshot(result)
        snapshot.update({
            "snapshot_id": str(uuid.uuid4()),
            "created_at": _now(),
            "approval_state": "pending" if snapshot.get("approved") else "none",
            "approval_note": "Review these exact saved orders. Approving will not rerun analysis.",
        })
        _save_snapshot(snapshot)
        return snapshot


def approve_saved_orders(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RequestError("Approval request must be a JSON object.")
    snapshot_id = payload.get("snapshot_id")
    selected_raw = payload.get("symbols")
    mode = payload.get("mode", "dry_run")
    if mode not in ("dry_run", "alpaca_paper"):
        raise RequestError("Approval mode must be dry_run or alpaca_paper.")
    selected = validate_symbols(selected_raw)
    with RUN_LOCK:
        snapshot = _load_snapshot(str(snapshot_id))
        if snapshot.get("approval_state") != "pending":
            raise RequestError("This run is no longer awaiting approval. Run analysis again if needed.", 409)
        orders = snapshot.get("approved")
        if not isinstance(orders, list) or not orders:
            raise RequestError("This run has no guardian-approved orders.", 409)
        approved_symbols = {str(order.get("symbol", "")).upper()
                            for order in orders if isinstance(order, dict)}
        if not set(selected) <= approved_symbols:
            raise RequestError("You can approve only symbols in this run's saved approved-order list.")
        exact_orders = [order for order in orders
                        if isinstance(order, dict) and str(order.get("symbol", "")).upper() in set(selected)]
        if len(exact_orders) != len(selected):
            raise RequestError("Each selected symbol must map to exactly one saved order.")
        if mode == "alpaca_paper":
            if not live_orders.orders_enabled():
                raise RequestError("Live paper submission is disabled. Start the local agent with ALPACA_ORDERS_ENABLED=true to enable the existing safety gate.", 409)
            if (os.environ.get("BROKER_MODE") or "offline").strip().lower() != "alpaca_paper":
                raise RequestError("Switch Broker mode to alpaca_paper before approving live paper orders.", 409)
            if not (os.environ.get("ALPACA_API_KEY", "").strip()
                    and os.environ.get("ALPACA_SECRET_KEY", "").strip()):
                raise RequestError("Set both Alpaca PAPER keys in Local key management first.", 409)
            try:
                placed = live_orders.submit_orders(exact_orders, only_symbols=selected)
            except live_orders.OrderSubmissionBlocked as exc:
                raise RequestError(str(exc), 409) from None
        else:
            placed = guarded_plan.place_approved(exact_orders, only_symbols=selected)

        # Preserve the existing M9 audit trail; live orders also get the existing M6 memory entry.
        journal.append_order_results(snapshot.get("run_id"), snapshot.get("timeframe"), placed,
                                     live=mode == "alpaca_paper")
        if mode == "alpaca_paper":
            run_cycle._journal_live_orders({"run_id": snapshot.get("run_id"),
                                            "timeframe": snapshot.get("timeframe")}, placed)
        snapshot["placed"] = list(snapshot.get("placed") or []) + placed
        snapshot["approval_state"] = "completed"
        snapshot["approval_mode"] = mode
        snapshot["approved_symbols"] = selected
        snapshot["approval_completed_at"] = _now()
        snapshot["status"] = "ok" if mode == "alpaca_paper" else "dry_run_approved"
        _save_snapshot(snapshot)
        return safe_snapshot(snapshot)


def history() -> list[dict]:
    paths = sorted(_runs_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
    items = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            items.append({
                "snapshot_id": data.get("snapshot_id"),
                "created_at": data.get("created_at"),
                "status": data.get("status"),
                "approval_state": data.get("approval_state"),
                "symbols": data.get("symbols") or [],
                "timeframe": data.get("timeframe"),
                "approved_count": len(data.get("approved") or []),
            })
    return items


class Handler(BaseHTTPRequestHandler):
    server_version = "PaperTradingLocalAgent/1.0"

    def log_message(self, fmt, *args):
        # Avoid logging request bodies, keys, or pairing headers.
        print("Local agent:", fmt % args)

    def _origin(self):
        origin = self.headers.get("Origin", "").rstrip("/")
        if origin and origin not in _origins():
            self._send(403, {"error": "This dashboard origin is not allowed. Set TRADING_UI_ORIGINS locally."})
            return None
        return origin

    def _send(self, status: int, body: dict, origin: str = ""):
        encoded = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _auth(self, origin):
        if not hmac.compare_digest(self.headers.get("X-Local-Agent-Token", ""), PAIRING_TOKEN):
            self._send(401, {"error": "Pairing token is missing or incorrect."}, origin or "")
            return False
        return True

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise RequestError("Invalid content length.")
        if length <= 0 or length > 1_000_000:
            raise RequestError("Request body must be between 1 byte and 1 MB.")
        try:
            value = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            raise RequestError("Request body must be valid JSON.")
        if not isinstance(value, dict):
            raise RequestError("Request body must be a JSON object.")
        return value

    def do_OPTIONS(self):
        origin = self.headers.get("Origin", "").rstrip("/")
        if not origin or origin not in _origins():
            self._send(403, {"error": "Dashboard origin is not allowed."})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Local-Agent-Token")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        origin = self._origin()
        if origin is None or not self._auth(origin):
            return
        path = urlparse(self.path).path
        if path not in ("/api/health", "/api/keys", "/api/history") and not path.startswith("/api/history/"):
            self._send(404, {"error": "Endpoint not found."}, origin)
            return
        if path == "/api/health":
            self._send(200, {"ok": True, **key_status()}, origin)
        elif path == "/api/keys":
            self._send(200, key_status(), origin)
        elif path == "/api/history":
            self._send(200, {"items": history()}, origin)
        else:
            snapshot_id = path.rsplit("/", 1)[-1]
            try:
                self._send(200, _load_snapshot(snapshot_id), origin)
            except RequestError as exc:
                self._send(exc.status, {"error": str(exc)}, origin)

    def do_POST(self):
        origin = self._origin()
        if origin is None or not self._auth(origin):
            return
        path = urlparse(self.path).path
        try:
            payload = self._body()
            if path == "/api/keys":
                self._send(200, update_local_settings(payload), origin)
            elif path == "/api/run":
                self._send(200, execute_cycle(payload), origin)
            elif path == "/api/approve":
                self._send(200, approve_saved_orders(payload), origin)
            else:
                self._send(404, {"error": "Endpoint not found."}, origin)
        except RequestError as exc:
            self._send(exc.status, {"error": str(exc)}, origin)
        except run_cycle.CycleConfigError as exc:
            self._send(400, {"error": str(exc)}, origin)
        except Exception as exc:
            # Error type only: never expose traceback, request contents, or credential-bearing messages.
            self._send(500, {"error": f"Local agent failed safely ({type(exc).__name__})."}, origin)


def main():
    try:
        port = int(os.environ.get("TRADING_UI_PORT", str(DEFAULT_PORT)))
    except ValueError:
        raise SystemExit("TRADING_UI_PORT must be an integer.")
    if not 1024 <= port <= 65535:
        raise SystemExit("TRADING_UI_PORT must be between 1024 and 65535.")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    print(f"Paper-trading local agent listening only on http://127.0.0.1:{port}")
    print(f"Pairing token (enter this in the dashboard; never share it): {PAIRING_TOKEN}")
    print("Keys entered in the dashboard stay in this local process and are not saved to disk.")
    print("Live paper orders remain disabled unless ALPACA_ORDERS_ENABLED=true was set outside the UI.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local agent.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
