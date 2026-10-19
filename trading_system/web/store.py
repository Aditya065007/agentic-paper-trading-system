"""Local-only JSON store for dashboard run history."""
import json, os, tempfile, threading
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent / ".local_state"
RUNS_FILE = STATE_DIR / "runs.json"
_LOCK = threading.RLock()

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def _secure_dir():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try: STATE_DIR.chmod(0o700)
    except OSError: pass

def read_runs():
    with _LOCK:
        _secure_dir()
        try:
            value = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []

def write_runs(runs):
    with _LOCK:
        _secure_dir()
        fd, name = tempfile.mkstemp(prefix=".runs-", suffix=".tmp", dir=str(STATE_DIR))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(runs[-100:], stream, ensure_ascii=False, indent=2, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            try: os.chmod(name, 0o600)
            except OSError: pass
            os.replace(name, RUNS_FILE)
        finally:
            try: os.unlink(name)
            except OSError: pass

def get_run(run_id):
    return next((r for r in read_runs() if r.get("id") == run_id), None)

def save_run(record):
    runs = read_runs()
    for i, old in enumerate(runs):
        if old.get("id") == record.get("id"):
            runs[i] = record
            break
    else:
        runs.append(record)
    write_runs(runs)
