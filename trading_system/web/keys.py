"""Local credential settings. Order gates and broker URL cannot be changed here."""
import os, re, tempfile
from pathlib import Path
import requests

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
EDITABLE = {"ALPACA_API_KEY", "ALPACA_SECRET_KEY", "GROQ_API_KEY", "NEWSAPI_KEY", "BROKER_MODE"}
SECRET_KEYS = {"ALPACA_API_KEY", "ALPACA_SECRET_KEY", "GROQ_API_KEY", "NEWSAPI_KEY"}
KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

def _read_env_lines():
    try: return ENV_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError: return []

def save_settings(values):
    if not isinstance(values, dict): raise ValueError("Settings must be an object.")
    if set(values) - EDITABLE: raise ValueError("One or more settings cannot be changed here.")
    normalized = {}
    for name, value in values.items():
        if not isinstance(name, str) or not KEY_PATTERN.fullmatch(name) or name not in EDITABLE:
            raise ValueError("Unsupported setting.")
        if not isinstance(value, str): raise ValueError(name + " must be text.")
        value = value.strip()
        if any(c in value for c in ("\n", "\r", "\x00")): raise ValueError(name + " contains an invalid character.")
        if name == "BROKER_MODE" and value not in {"offline", "alpaca_paper"}:
            raise ValueError("BROKER_MODE must be offline or alpaca_paper.")
        if name in SECRET_KEYS and value and (len(value) > 500 or value.startswith("http")):
            raise ValueError(name + " does not look like a valid key value.")
        normalized[name] = value
    lines, replaced, output = _read_env_lines(), set(), []
    for line in lines:
        match = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s*=", line)
        name = match.group(1) if match else None
        if name in normalized:
            if name not in replaced:
                output.append(name + "=" + normalized[name]); replaced.add(name)
        else: output.append(line)
    for name, value in normalized.items():
        if name not in replaced: output.append(name + "=" + value)
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".env-", suffix=".tmp", dir=str(ENV_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(output).rstrip() + "\n"); stream.flush(); os.fsync(stream.fileno())
        try: os.chmod(temp_path, 0o600)
        except OSError: pass
        os.replace(temp_path, ENV_PATH)
        try: ENV_PATH.chmod(0o600)
        except OSError: pass
    finally:
        try: os.unlink(temp_path)
        except OSError: pass
    for name, value in normalized.items():
        if value: os.environ[name] = value
        else: os.environ.pop(name, None)
    return public_settings()

def public_settings():
    result = {}
    for name in sorted(EDITABLE):
        value = os.environ.get(name, "")
        if name in SECRET_KEYS:
            result[name] = {"configured": bool(value), "masked": ("••••" + value[-4:]) if value else ""}
        else:
            result[name] = value or ("offline" if name == "BROKER_MODE" else "")
    return result

def test_connection(provider):
    """Never return response bodies, keys, or exception messages."""
    try:
        if provider == "alpaca":
            key, secret = os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_SECRET_KEY", "")
            if not key or not secret: return {"ok": False, "message": "Add both Alpaca paper keys first."}
            r = requests.get("https://paper-api.alpaca.markets/v2/account",
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}, timeout=8)
            ok = r.status_code == 200
            return {"ok": ok, "message": "Alpaca paper connection successful." if ok else
                    "Alpaca returned HTTP " + str(r.status_code) + "; check paper keys."}
        if provider == "groq":
            key = os.environ.get("GROQ_API_KEY", "")
            if not key: return {"ok": False, "message": "Add a Groq API key first."}
            r = requests.get("https://api.groq.com/openai/v1/models",
                headers={"Authorization": "Bearer " + key}, timeout=8)
            ok = r.status_code == 200
            return {"ok": ok, "message": "Groq connection successful." if ok else
                    "Groq returned HTTP " + str(r.status_code) + "; check the key."}
        if provider == "newsapi":
            key = os.environ.get("NEWSAPI_KEY", "")
            if not key: return {"ok": False, "message": "Add a NewsAPI key first."}
            r = requests.get("https://newsapi.org/v2/top-headlines",
                params={"country": "us", "pageSize": 1, "apiKey": key}, timeout=8)
            ok = r.status_code == 200
            return {"ok": ok, "message": "NewsAPI connection successful." if ok else
                    "NewsAPI returned HTTP " + str(r.status_code) + "; check key or quota."}
        return {"ok": False, "message": "Unknown provider."}
    except requests.RequestException:
        return {"ok": False, "message": "Connection failed or timed out. Check internet access."}
