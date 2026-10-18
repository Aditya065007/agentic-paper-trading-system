# servers/tools/data.py
# M1 - pure data-layer helpers (no network, no MCP). The server file does the HTTP request and calls these.
#
#   bars_to_frame      Alpaca bars -> clean OHLCV frame (UTC index; keeps VWAP and trade count)
#   regular_hours_only keep only bars that overlap the 09:30-16:00 New York session (hourly only)
#   drop_incomplete    drop bars that are still forming (daily until 16:00 ET, hourly until the hour ends)
#   read_cache/write_cache  small JSON disk cache of the RAW Alpaca rows, with a time-to-live
#   process_rows       raw rows -> frames plus human-readable notes
#
# Everything that can go wrong with the cache or the settings file falls back to safe behaviour
# (fetch again / built-in defaults) and says so in a note. A cache problem never stops a run.

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
NEW_YORK = "America/New_York"
TIMEFRAMES = ("1Day", "1Hour")
# Session in New York time. Early-close days (13:00) are not special-cased: the daily bar is then
# treated as in progress until 16:00, which only ever waits too long, never uses a partial bar.
SESSION_OPEN_MINUTES = 9 * 60 + 30
SESSION_CLOSE_MINUTES = 16 * 60

DEFAULT_SETTINGS = {
    "adjustment": "split",
    "drop_in_progress_bar": True,
    "regular_hours_only": True,
    "cache_enabled": True,
    "cache_ttl_minutes": {"1Day": 15, "1Hour": 5},
}
_ADJUSTMENTS = ("raw", "split", "dividend", "all")


# ------------------------------------------------------------------ settings

def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_data_settings(path=None):
    """Return (settings, warning). A missing or invalid file gives the defaults plus a warning."""
    try:
        import yaml
        with open(Path(path) if path is not None else CONFIG_DIR / "data.yaml", "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError("file is empty or not a mapping")
        settings = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_SETTINGS.items()}
        for key in ("drop_in_progress_bar", "regular_hours_only", "cache_enabled"):
            if key in data:
                if not isinstance(data[key], bool):
                    raise ValueError(f"{key} must be true or false")
                settings[key] = data[key]
        if "adjustment" in data:
            if data["adjustment"] not in _ADJUSTMENTS:
                raise ValueError(f"adjustment must be one of {', '.join(_ADJUSTMENTS)}")
            settings["adjustment"] = data["adjustment"]
        if "cache_ttl_minutes" in data:
            ttl = data["cache_ttl_minutes"]
            if not isinstance(ttl, dict) or set(ttl) != set(TIMEFRAMES):
                raise ValueError("cache_ttl_minutes must list exactly 1Day and 1Hour")
            for name, minutes in ttl.items():
                if not _number(minutes) or not 0 <= minutes <= 24 * 60:
                    raise ValueError(f"cache_ttl_minutes.{name} must be a number of minutes from 0 to 1440")
            settings["cache_ttl_minutes"] = {k: float(v) for k, v in ttl.items()}
        return settings, None
    except Exception as exc:
        defaults = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_SETTINGS.items()}
        return defaults, f"config/data.yaml could not be used ({exc}); built-in data settings are in use."


# ------------------------------------------------------------------ frames

def bars_to_frame(bars):
    """List of Alpaca bar dicts -> (frame, dropped_invalid_count).

    Columns: open high low close volume vwap trade_count. The index is the bar start time in UTC.
    Rows with a missing or non-numeric OHLCV value are dropped (and counted), duplicates keep the last."""
    frame = pd.DataFrame(bars)
    if frame.empty or "t" not in frame.columns:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "trade_count"]), 0
    frame["t"] = pd.to_datetime(frame["t"], utc=True, errors="coerce")
    frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
                                  "vw": "vwap", "n": "trade_count"})
    for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if column not in frame.columns:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = ["t", "open", "high", "low", "close", "volume"]
    valid = frame[required].notna().all(axis=1)
    dropped = int((~valid).sum())
    frame = frame.loc[valid]
    frame = frame[~frame["t"].duplicated(keep="last")]
    frame = frame.set_index("t").sort_index()
    return frame[["open", "high", "low", "close", "volume", "vwap", "trade_count"]], dropped


def _new_york_minutes(index):
    local = index.tz_convert(NEW_YORK)
    return local, local.hour * 60 + local.minute


def regular_hours_only(frame, timeframe):
    """Hourly bars: keep those that overlap 09:30-16:00 New York time. Daily bars are returned as they are.

    An hourly bar starting at 09:00 overlaps the open, so it is kept (it may include up to 30 minutes
    of pre-market trading: [VERIFY] against real Alpaca hourly bars)."""
    if timeframe != "1Hour" or frame.empty:
        return frame, 0
    _, minutes = _new_york_minutes(frame.index)
    keep = (minutes + 60 > SESSION_OPEN_MINUTES) & (minutes < SESSION_CLOSE_MINUTES)
    return frame.loc[keep], int((~keep).sum())


def drop_incomplete(frame, timeframe, now):
    """Remove bars that are still forming at `now` (UTC). Returns (frame, list of dropped bar start times).

    Daily: complete only after 16:00 New York time on the bar's own date.
    Hourly: complete only once start + 1 hour has passed."""
    if frame.empty:
        return frame, []
    if timeframe == "1Hour":
        complete = (frame.index + timedelta(hours=1)) <= now
    else:
        local = frame.index.tz_convert(NEW_YORK)
        close = pd.DatetimeIndex([
            pd.Timestamp(d.year, d.month, d.day, SESSION_CLOSE_MINUTES // 60, SESSION_CLOSE_MINUTES % 60,
                         tz=NEW_YORK).tz_convert("UTC")
            for d in local
        ])
        complete = close <= now
    dropped = [str(ts) for ts in frame.index[~complete]]
    return frame.loc[complete], dropped


def process_rows(rows, timeframe, settings, now=None):
    """Raw Alpaca rows {symbol: [bars]} -> ({symbol: frame}, notes). `now` defaults to the current time."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    frames, notes = {}, []
    for symbol, bars in (rows or {}).items():
        frame, invalid = bars_to_frame(bars)
        if invalid:
            notes.append(f"{symbol}: dropped {invalid} bar(s) with missing or non-numeric values.")
        if settings.get("regular_hours_only", True):
            frame, outside = regular_hours_only(frame, timeframe)
            if outside:
                notes.append(f"{symbol}: dropped {outside} bar(s) outside regular trading hours.")
        if settings.get("drop_in_progress_bar", True):
            frame, dropped = drop_incomplete(frame, timeframe, now)
            if dropped:
                notes.append(f"{symbol}: dropped the in-progress bar ({', '.join(dropped)}).")
        frames[symbol] = frame
    return frames, notes


# ------------------------------------------------------------------ disk cache (raw rows, JSON)

def cache_key(symbols, timeframe, lookback_days, adjustment, feed):
    text = json.dumps([sorted(symbols), timeframe, int(lookback_days), adjustment, feed])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _cache_file(cache_dir, key):
    return Path(cache_dir) / f"bars_{key}.json"


def read_cache(cache_dir, key, ttl_minutes, now=None):
    """Return the cached raw rows if they are younger than the TTL, else None. Never raises."""
    try:
        if not ttl_minutes or ttl_minutes <= 0:
            return None
        now = now or datetime.now(timezone.utc)
        payload = json.loads(_cache_file(cache_dir, key).read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        age = now - fetched_at
        if age < timedelta(0) or age > timedelta(minutes=float(ttl_minutes)):
            return None
        rows = payload["rows"]
        return rows if isinstance(rows, dict) else None
    except Exception:
        return None


def write_cache(cache_dir, key, rows, now=None) -> bool:
    """Write the raw rows atomically. Returns False (and does nothing else) if the write fails."""
    try:
        now = now or datetime.now(timezone.utc)
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"fetched_at": now.isoformat(), "rows": rows}
        handle, temp_name = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            os.replace(temp_name, _cache_file(directory, key))
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return True
    except Exception:
        return False
