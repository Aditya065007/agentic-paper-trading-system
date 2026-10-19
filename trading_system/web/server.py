"""Loopback-only web UI for the existing run_cycle pipeline.

Start with: python -m trading_system.web.server
Live submission requires PAPER_WEB_LIVE_MODE=true at process launch plus the existing
ALPACA_ORDERS_ENABLED=true and BROKER_MODE=alpaca_paper gates. The UI cannot set order gates.
"""
import hmac, os, re, secrets, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route
from trading_system.web import keys, store

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Capture live mode BEFORE loading .env: a value in .env must never enable live mode.
LIVE_MODE = os.environ.get("PAPER_WEB_LIVE_MODE", "").strip().lower() == "true"
LAUNCH_ORDERS_GATE = os.environ.get("ALPACA_ORDERS_ENABLED", "").strip().lower() == "true"
LAUNCH_STOPS_GATE = os.environ.get("ALPACA_STOPS_ENABLED", "").strip().lower() == "true"
PORT = int(os.environ.get("PAPER_WEB_PORT", "8765"))
load_dotenv(PROJECT_ROOT / ".env", override=False)
os.environ["ALPACA_ORDERS_ENABLED"] = "true" if LAUNCH_ORDERS_GATE else "false"
os.environ["ALPACA_STOPS_ENABLED"] = "true" if LAUNCH_STOPS_GATE else "false"
STATIC_DIR = Path(__file__).resolve().parent / "static"
LAUNCH_TOKEN = secrets.token_urlsafe(32)
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
MAX_SYMBOLS, SNAPSHOT_TTL = 25, 15 * 60
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-cycle")
_RUN_LOCK = threading.Lock()
_APPROVAL_LOCK = threading.Lock()

def _now():
    return datetime.now(timezone.utc).isoformat()

def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()
                if not any(x in str(k).lower() for x in ("api_key", "secret_key", "password", "token"))}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, str):
        for name in keys.SECRET_KEYS:
            secret = os.environ.get(name, "")
            if secret and len(secret) >= 4: value = value.replace(secret, "[REDACTED]")
        return value[:10000]
    if value is None or isinstance(value, (int, float, bool)): return value
    return str(value)[:1000]

def _public(record):
    return _safe({k: v for k, v in record.items() if k not in {"secret_env", "raw_exception"}})

def _host_ok(request):
    return request.headers.get("host", "").lower() in {
        "127.0.0.1:" + str(PORT), "localhost:" + str(PORT), "[::1]:" + str(PORT)}

def _origin_ok(request):
    origin = request.headers.get("origin")
    return not origin or origin in {
        "http://127.0.0.1:" + str(PORT), "http://localhost:" + str(PORT), "http://[::1]:" + str(PORT)}

async def _guard(request, call_next):
    if not _host_ok(request): return JSONResponse({"error": "Invalid Host header."}, status_code=400)
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not _origin_ok(request):
        return JSONResponse({"error": "Invalid request origin."}, status_code=403)
    if request.url.path.startswith("/api/"):
        token = request.headers.get("x-launch-token", "")
        if not token or not hmac.compare_digest(token, LAUNCH_TOKEN):
            return JSONResponse({"error": "Launch token missing or invalid. Use the URL printed by the app."}, status_code=403)
    return await call_next(request)

def normalize_symbols(value):
    if not isinstance(value, list) or not value or len(value) > MAX_SYMBOLS:
        raise ValueError("Enter between 1 and " + str(MAX_SYMBOLS) + " stock symbols.")
    result = []
    for raw in value:
        if not isinstance(raw, str): raise ValueError("Every symbol must be text.")
        symbol = raw.strip().upper()
        if not SYMBOL_RE.fullmatch(symbol): raise ValueError("Invalid stock symbol: " + symbol[:20])
        if symbol not in result: result.append(symbol)
    if not result: raise ValueError("Enter at least one stock symbol.")
    return result

def _run_job(run_id, config):
    record = store.get_run(run_id)
    if not record: return
    record.update({"status": "running", "started_at": _now()}); store.save_run(record)
    try:
        with _RUN_LOCK:
            from trading_system.crew.run_cycle import run_cycle_sync
            result = run_cycle_sync(config)
        record = store.get_run(run_id) or record
        record.update({"status": "completed", "finished_at": _now(), "result": _safe(result),
                       "approval_expires_at": time.time() + SNAPSHOT_TTL,
                       "approved_orders": _safe(result.get("approved") or []),
                       "pipeline_status": result.get("status")})
        store.save_run(record)
    except Exception as exc:
        record = store.get_run(run_id) or record
        record.update({"status": "failed", "finished_at": _now(),
                       "error": "Pipeline failed (" + type(exc).__name__ + "). Check the local terminal."})
        store.save_run(record)

async def home(request):
    token = request.query_params.get("token", "")
    if not token or not hmac.compare_digest(token, LAUNCH_TOKEN):
        return PlainTextResponse("Open the exact local URL printed in the terminal.", status_code=403)
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

async def status(request):
    return JSONResponse({"ok": True, "live_mode": LIVE_MODE,
        "orders_gate_open": LAUNCH_ORDERS_GATE,
        "broker_mode": os.environ.get("BROKER_MODE", "offline").strip().lower(),
        "keys": keys.public_settings(), "max_symbols": MAX_SYMBOLS, "snapshot_ttl_seconds": SNAPSHOT_TTL})

async def create_run(request):
    try:
        body = await request.json()
        symbols = normalize_symbols(body.get("symbols"))
        timeframe = body.get("timeframe", "1Day")
        if timeframe not in {"1Day", "1Hour", "BOTH"}: raise ValueError("Timeframe must be 1Day, 1Hour or BOTH.")
        advisory = body.get("advisory", False)
        if not isinstance(advisory, bool): raise ValueError("Advisory must be true or false.")
        lookback = body.get("lookback_days", 180)
        if isinstance(lookback, bool) or not isinstance(lookback, int) or not 1 <= lookback <= 2000:
            raise ValueError("Lookback must be between 1 and 2000 days.")
        run_id = str(uuid.uuid4())
        config = {"symbols": symbols, "timeframe": timeframe, "advisory": advisory, "lookback_days": lookback}
        record = {"id": run_id, "created_at": _now(), "status": "queued", "config": config, "approved_orders": []}
        store.save_run(record)
        _EXECUTOR.submit(_run_job, run_id, dict(config, approval=None))
        return JSONResponse({"run": _public(record)}, status_code=202)
    except ValueError as exc: return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception: return JSONResponse({"error": "Request body must be valid JSON."}, status_code=400)

async def get_run(request):
    record = store.get_run(request.path_params["run_id"])
    if not record: return JSONResponse({"error": "Run not found."}, status_code=404)
    if record.get("approval_expires_at") and time.time() > record["approval_expires_at"]:
        record["approval_expired"] = True; store.save_run(record)
    return JSONResponse({"run": _public(record)})

async def list_runs(request):
    runs = sorted(store.read_runs(), key=lambda r: r.get("created_at", ""), reverse=True)
    for record in runs:
        if record.get("approval_expires_at") and time.time() > record["approval_expires_at"]:
            record["approval_expired"] = True
    return JSONResponse({"runs": [_public(r) for r in runs[:100]]})

def _claim_approval(run_id, body):
    """Validate and claim a saved proposal under one lock, preventing concurrent duplicate submissions."""
    with _APPROVAL_LOCK:
        record = store.get_run(run_id)
        if not record: return None, JSONResponse({"error": "Run not found."}, status_code=404)
        if body.get("confirm") is not True:
            return None, JSONResponse({"error": "Explicit confirmation is required."}, status_code=400)
        if record.get("status") != "completed" or record.get("pipeline_status") != "awaiting_approval":
            return None, JSONResponse({"error": "This run has no pending approval."}, status_code=409)
        expiry = record.get("approval_expires_at", 0)
        if not isinstance(expiry, (int, float)) or time.time() > expiry:
            return None, JSONResponse({"error": "Approval snapshot expired. Run the pipeline again."}, status_code=410)
        orders = record.get("approved_orders") or []
        if not orders: return None, JSONResponse({"error": "Guardian approved no orders in this run."}, status_code=409)
        try: symbols = normalize_symbols(body.get("symbols"))
        except ValueError as exc: return None, JSONResponse({"error": str(exc)}, status_code=400)
        eligible = {str(o.get("symbol", "")).upper() for o in orders if isinstance(o, dict)}
        if not set(symbols) <= eligible:
            return None, JSONResponse({"error": "Only guardian-approved symbols from this run can be selected."}, status_code=400)
        record["status"], record["approved_symbols"] = "submitting", symbols
        store.save_run(record)
        return (record, symbols, orders), None

async def approve_run(request):
    try: body = await request.json()
    except Exception: return JSONResponse({"error": "Request body must be valid JSON."}, status_code=400)
    claimed, error = _claim_approval(request.path_params["run_id"], body)
    if error: return error
    record, symbols, orders = claimed
    try:
        if LIVE_MODE:
            if not LAUNCH_ORDERS_GATE:
                raise PermissionError("Live mode requires ALPACA_ORDERS_ENABLED=true at launch.")
            if os.environ.get("BROKER_MODE", "").strip().lower() != "alpaca_paper":
                raise PermissionError("Live mode requires BROKER_MODE=alpaca_paper.")
            from trading_system.servers.tools import live_orders
            with _RUN_LOCK: placed = live_orders.submit_orders(orders, only_symbols=symbols)
            try:
                from trading_system.crew.run_cycle import _journal_live_orders
                _journal_live_orders({"run_id": record.get("result", {}).get("run_id"),
                                      "timeframe": record.get("result", {}).get("timeframe")}, placed)
            except Exception: pass
            mode = "alpaca_paper"
        else:
            from trading_system.crew import guarded_plan
            placed = guarded_plan.place_approved(orders, only_symbols=symbols)
            mode = "dry_run"
        record.update({"status": "approved", "approval_mode": mode, "placement": _safe(placed),
                       "approved_at": _now(), "approved_orders": []})
        store.save_run(record)
        return JSONResponse({"run": _public(record)})
    except PermissionError as exc:
        record.update({"status": "completed", "approved_symbols": []}); store.save_run(record)
        return JSONResponse({"error": str(exc)}, status_code=403)
    except Exception as exc:
        record.update({"status": "approval_error", "error": "Approval failed (" + type(exc).__name__ +
                       "). Check Alpaca paper dashboard before retrying."}); store.save_run(record)
        return JSONResponse({"error": record["error"]}, status_code=502)

async def save_keys(request):
    try:
        body = await request.json()
        return JSONResponse({"keys": keys.save_settings(body.get("values", {}))})
    except ValueError as exc: return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception: return JSONResponse({"error": "Could not save local settings."}, status_code=500)

async def test_key(request):
    try: return JSONResponse(keys.test_connection((await request.json()).get("provider", "")))
    except Exception: return JSONResponse({"ok": False, "message": "Connection test failed."}, status_code=400)

app = Starlette(debug=False, routes=[
    Route("/", home, methods=["GET"]),
    Route("/api/status", status, methods=["GET"]),
    Route("/api/runs", list_runs, methods=["GET"]),
    Route("/api/runs", create_run, methods=["POST"]),
    Route("/api/runs/{run_id}", get_run, methods=["GET"]),
    Route("/api/runs/{run_id}/approve", approve_run, methods=["POST"]),
    Route("/api/keys", save_keys, methods=["POST"]),
    Route("/api/keys/test", test_key, methods=["POST"]),
])
app.middleware("http")(_guard)

def main():
    if not STATIC_DIR.joinpath("index.html").is_file(): raise SystemExit("Dashboard static/index.html is missing.")
    url = "http://127.0.0.1:" + str(PORT) + "/?token=" + LAUNCH_TOKEN
    print("\nLocal paper-trading dashboard\nOpen this exact URL in your browser:\n  " + url)
    print("Live paper submission mode: " + ("ON" if LIVE_MODE else "OFF (dry-run only)"))
    if LIVE_MODE: print("Requires ALPACA_ORDERS_ENABLED=true and BROKER_MODE=alpaca_paper in the launch environment.")
    print("Listening only on 127.0.0.1. Keep this terminal open; Ctrl+C stops the dashboard.\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning", access_log=False)

if __name__ == "__main__": main()
