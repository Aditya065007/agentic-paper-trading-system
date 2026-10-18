"""G4: M1 data layer. Split adjustment, in-progress bar, regular hours, disk cache, timeframe check.

Pure helpers are tested directly; the server path uses a fake `requests.get` and a frozen clock.
Nothing touches the network or the real cache folder."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from trading_system.servers import technical_indicators_mcp_server_multiStocks as server
from trading_system.servers.tools import data as m1

UTC = timezone.utc
SETTINGS = m1.DEFAULT_SETTINGS


def bar(t, o=10.0, h=11.0, l=9.0, c=10.5, v=1000, vw=10.4, n=50):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "vw": vw, "n": n}


def ts(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


# ------------------------------------------------------------------ frames

def test_frame_keeps_vwap_and_trade_count_and_a_utc_index():
    frame, dropped = m1.bars_to_frame([bar("2026-10-15T04:00:00Z"), bar("2026-10-16T04:00:00Z", vw=11.0, n=70)])
    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
    assert frame["vwap"].iloc[1] == 11.0 and frame["trade_count"].iloc[1] == 70
    assert str(frame.index.tz) == "UTC" and dropped == 0


def test_frame_without_vwap_or_count_still_works():
    frame, _ = m1.bars_to_frame([{"t": "2026-10-16T04:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 5}])
    assert len(frame) == 1 and pd.isna(frame["vwap"].iloc[0])


def test_invalid_rows_are_dropped_and_counted_duplicates_keep_the_last():
    bars = [bar("2026-10-14T04:00:00Z"), bar("2026-10-15T04:00:00Z", c=None), bar("2026-10-16T04:00:00Z", o="x"),
            bar("2026-10-17T04:00:00Z", c=1.0), bar("2026-10-17T04:00:00Z", c=2.0)]
    frame, dropped = m1.bars_to_frame(bars)
    assert dropped == 2 and len(frame) == 2 and frame["close"].iloc[-1] == 2.0


def test_empty_input_gives_an_empty_frame():
    frame, dropped = m1.bars_to_frame([])
    assert frame.empty and dropped == 0 and "vwap" in frame.columns


# ------------------------------------------------------------------ in-progress bar

def daily(*days):
    frame, _ = m1.bars_to_frame([bar(f"{d}T04:00:00Z") for d in days])   # 00:00 New York (summer time)
    return frame


def test_todays_daily_bar_is_dropped_before_the_close_and_kept_after():
    frame = daily("2026-10-15", "2026-10-16")
    kept, dropped = m1.drop_incomplete(frame, "1Day", ts("2026-10-16T19:59:00"))   # 15:59 New York
    assert len(kept) == 1 and len(dropped) == 1
    kept, dropped = m1.drop_incomplete(frame, "1Day", ts("2026-10-16T20:00:00"))   # 16:00 New York
    assert len(kept) == 2 and dropped == []


def test_daily_close_uses_winter_time_too():
    frame, _ = m1.bars_to_frame([bar("2026-12-15T05:00:00Z")])                      # 00:00 New York (winter)
    assert len(m1.drop_incomplete(frame, "1Day", ts("2026-12-15T20:59:00"))[0]) == 0
    assert len(m1.drop_incomplete(frame, "1Day", ts("2026-12-15T21:00:00"))[0]) == 1


def test_weekend_run_keeps_fridays_complete_bar():
    frame = daily("2026-10-15", "2026-10-16")
    kept, dropped = m1.drop_incomplete(frame, "1Day", ts("2026-10-18T12:00:00"))   # Sunday
    assert len(kept) == 2 and dropped == []


def test_hourly_bar_is_complete_only_when_its_hour_has_passed():
    frame, _ = m1.bars_to_frame([bar("2026-10-16T14:00:00Z"), bar("2026-10-16T15:00:00Z")])
    kept, dropped = m1.drop_incomplete(frame, "1Hour", ts("2026-10-16T15:59:59"))
    assert len(kept) == 1 and len(dropped) == 1
    kept, _ = m1.drop_incomplete(frame, "1Hour", ts("2026-10-16T16:00:00"))
    assert len(kept) == 2


def test_empty_frame_is_fine():
    frame, _ = m1.bars_to_frame([])
    assert m1.drop_incomplete(frame, "1Day", ts("2026-10-16T20:00:00"))[1] == []


# ------------------------------------------------------------------ regular hours

def hourly_day():
    # 04:00 .. 19:00 New York (summer time = UTC-4)
    return m1.bars_to_frame([bar(f"2026-10-16T{h + 4:02d}:00:00Z") for h in range(4, 20) if h + 4 < 24])[0]


def test_hourly_keeps_only_bars_overlapping_the_regular_session():
    frame = hourly_day()
    kept, outside = m1.regular_hours_only(frame, "1Hour")
    hours = sorted(kept.index.tz_convert("America/New_York").hour)
    assert hours == [9, 10, 11, 12, 13, 14, 15]          # 09:00 overlaps the 09:30 open; 16:00 is after the close
    assert outside == len(frame) - len(kept)


def test_daily_bars_are_never_filtered_by_hours():
    frame = daily("2026-10-15", "2026-10-16")
    kept, outside = m1.regular_hours_only(frame, "1Day")
    assert len(kept) == 2 and outside == 0


def test_hourly_filter_follows_the_clock_change_in_winter():
    frame = m1.bars_to_frame([bar(f"2026-12-15T{h:02d}:00:00Z") for h in range(12, 23)])[0]   # UTC-5
    kept, _ = m1.regular_hours_only(frame, "1Hour")
    assert sorted(kept.index.tz_convert("America/New_York").hour) == [9, 10, 11, 12, 13, 14, 15]


def test_process_rows_reports_what_it_dropped_and_respects_the_switches():
    rows = {"AAPL": [bar("2026-10-15T04:00:00Z"), bar("2026-10-16T04:00:00Z")]}
    frames, notes = m1.process_rows(rows, "1Day", SETTINGS, now=ts("2026-10-16T15:00:00"))
    assert len(frames["AAPL"]) == 1 and any("in-progress" in n for n in notes)
    off = dict(SETTINGS, drop_in_progress_bar=False)
    frames, notes = m1.process_rows(rows, "1Day", off, now=ts("2026-10-16T15:00:00"))
    assert len(frames["AAPL"]) == 2 and notes == []


# ------------------------------------------------------------------ settings

def test_shipped_settings_file_loads_cleanly_and_equals_the_defaults():
    settings, warning = m1.load_data_settings()
    assert warning is None and settings == m1.DEFAULT_SETTINGS


@pytest.mark.parametrize("text", [
    "", "adjustment: nonsense\n", "drop_in_progress_bar: yes please\n", "regular_hours_only: 1\n",
    "cache_ttl_minutes: 10\n", "cache_ttl_minutes: {1Day: 5}\n", "cache_ttl_minutes: {1Day: -1, 1Hour: 5}\n",
    "cache_ttl_minutes: {1Day: abc, 1Hour: 5}\n", "cache_enabled: [\n"])
def test_bad_settings_fall_back_to_defaults_with_a_warning(tmp_path, text):
    path = tmp_path / "data.yaml"
    path.write_text(text, encoding="utf-8")
    settings, warning = m1.load_data_settings(path)
    assert settings == m1.DEFAULT_SETTINGS and "data.yaml" in warning


def test_missing_settings_file_falls_back(tmp_path):
    settings, warning = m1.load_data_settings(tmp_path / "nope.yaml")
    assert settings == m1.DEFAULT_SETTINGS and warning


def test_a_valid_edit_is_used(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("adjustment: all\ncache_enabled: false\ncache_ttl_minutes: {1Day: 0, 1Hour: 2}\n", encoding="utf-8")
    settings, warning = m1.load_data_settings(path)
    assert warning is None and settings["adjustment"] == "all" and settings["cache_enabled"] is False
    assert settings["cache_ttl_minutes"] == {"1Day": 0.0, "1Hour": 2.0}
    assert settings["drop_in_progress_bar"] is True                      # untouched keys keep defaults


# ------------------------------------------------------------------ disk cache

NOW = ts("2026-10-16T21:00:00")
ROWS = {"AAPL": [bar("2026-10-16T04:00:00Z")]}


def test_cache_round_trip_and_ttl(tmp_path):
    key = m1.cache_key(["AAPL"], "1Day", 180, "split", "iex")
    assert m1.write_cache(tmp_path, key, ROWS, NOW) is True
    assert m1.read_cache(tmp_path, key, 15, NOW + timedelta(minutes=14)) == ROWS
    assert m1.read_cache(tmp_path, key, 15, NOW + timedelta(minutes=16)) is None


def test_ttl_zero_or_a_clock_that_went_backwards_never_uses_the_cache(tmp_path):
    key = m1.cache_key(["AAPL"], "1Day", 180, "split", "iex")
    m1.write_cache(tmp_path, key, ROWS, NOW)
    assert m1.read_cache(tmp_path, key, 0, NOW) is None
    assert m1.read_cache(tmp_path, key, 15, NOW - timedelta(minutes=1)) is None


def test_cache_key_changes_with_every_input_and_ignores_symbol_order():
    base = m1.cache_key(["AAPL", "MSFT"], "1Day", 180, "split", "iex")
    assert base == m1.cache_key(["MSFT", "AAPL"], "1Day", 180, "split", "iex")
    for other in (m1.cache_key(["AAPL"], "1Day", 180, "split", "iex"),
                  m1.cache_key(["AAPL", "MSFT"], "1Hour", 180, "split", "iex"),
                  m1.cache_key(["AAPL", "MSFT"], "1Day", 90, "split", "iex"),
                  m1.cache_key(["AAPL", "MSFT"], "1Day", 180, "raw", "iex"),
                  m1.cache_key(["AAPL", "MSFT"], "1Day", 180, "split", "sip")):
        assert other != base


@pytest.mark.parametrize("content", ["", "not json", "[]", '{"fetched_at": "x", "rows": {}}',
                                     '{"fetched_at": "2026-10-16T21:00:00+00:00", "rows": [1]}'])
def test_a_damaged_cache_file_is_ignored_not_fatal(tmp_path, content):
    key = m1.cache_key(["AAPL"], "1Day", 180, "split", "iex")
    (tmp_path / f"bars_{key}.json").write_text(content, encoding="utf-8")
    assert m1.read_cache(tmp_path, key, 15, NOW) is None


def test_a_cache_that_cannot_be_written_is_reported_false_not_raised(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert m1.write_cache(blocker / "sub", "k", ROWS, NOW) is False


def test_cache_file_holds_bars_only(tmp_path):
    key = m1.cache_key(["AAPL"], "1Day", 180, "split", "iex")
    m1.write_cache(tmp_path, key, ROWS, NOW)
    text = (tmp_path / f"bars_{key}.json").read_text(encoding="utf-8")
    assert set(json.loads(text)) == {"fetched_at", "rows"}       # bars and a timestamp, nothing else
    assert not list(tmp_path.glob("*.tmp"))                       # no temp file left behind


# ------------------------------------------------------------------ server path

class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Server with a fake Alpaca, a frozen clock, a private cache folder and clean module state."""
    calls = []
    pages = [{"bars": {"AAPL": [bar("2026-10-15T04:00:00Z"), bar("2026-10-16T04:00:00Z")]}, "next_page_token": "p2"},
             {"bars": {"AAPL": [bar("2026-10-17T04:00:00Z")], "MSFT": [bar("2026-10-16T04:00:00Z")]}}]

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(dict(params))
        return FakeResponse(pages[(len(calls) - 1) % 2])      # page 1, page 2, page 1, ...

    clock = {"now": ts("2026-10-18T12:00:00")}                                    # Sunday
    monkeypatch.setattr(server.requests, "get", fake_get)
    monkeypatch.setattr(server, "_utcnow", lambda: clock["now"])
    monkeypatch.setattr(server, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(server, "DATA_SETTINGS", json.loads(json.dumps(m1.DEFAULT_SETTINGS)))
    monkeypatch.setattr(server, "DATA_SETTINGS_WARNING", None)
    monkeypatch.setattr(server, "_cache", {})
    monkeypatch.setattr(server, "_active_timeframe", None)
    return calls, clock


def test_request_asks_alpaca_for_split_adjusted_bars_and_follows_pages(wired):
    calls, _ = wired
    out = server.fetch_ohlcv_batch(["AAPL", "MSFT"], 180, "1Day")
    assert [c["adjustment"] for c in calls] == ["split", "split"] and calls[1]["page_token"] == "p2"
    assert [f["symbol"] for f in out["fetched"]] == ["AAPL", "MSFT"] and out["status"] == "ok"


def test_the_cached_frames_keep_vwap_and_trade_count(wired):
    server.fetch_ohlcv_batch(["AAPL"], 180, "1Day")
    frame = server._get_df("AAPL")
    assert {"vwap", "trade_count"} <= set(frame.columns) and frame["vwap"].iloc[-1] == 10.4


def test_a_sunday_run_keeps_the_complete_friday_bar(wired):
    out = server.fetch_ohlcv_batch(["AAPL"], 180, "1Day")
    assert out["fetched"][0]["bars_fetched"] == 3        # 15th, 16th and a (fake) 17th: all closed by Sunday noon


def test_during_the_session_todays_bar_is_dropped_and_a_note_says_so(wired):
    _, clock = wired
    clock["now"] = ts("2026-10-17T15:00:00")                                        # 11:00 New York, bar of the 17th forming
    out = server.fetch_ohlcv_batch(["AAPL"], 180, "1Day")
    assert out["fetched"][0]["bars_fetched"] == 2
    assert any("in-progress" in n for n in out["data_notes"])


def test_second_call_inside_the_ttl_is_served_from_disk_without_a_request(wired):
    calls, clock = wired
    server.fetch_ohlcv_batch(["AAPL", "MSFT"], 180, "1Day")
    n = len(calls)
    clock["now"] += timedelta(minutes=5)
    out = server.fetch_ohlcv_batch(["AAPL", "MSFT"], 180, "1Day")
    assert len(calls) == n and any("disk cache" in note for note in out["data_notes"])
    clock["now"] += timedelta(minutes=30)
    server.fetch_ohlcv_batch(["AAPL", "MSFT"], 180, "1Day")
    assert len(calls) > n                                                           # expired, so it asked again


def test_cached_rows_are_re_filtered_at_the_current_time(wired):
    _, clock = wired
    clock["now"] = ts("2026-10-17T19:55:00")                                        # 15:55 New York: bar of the 17th forming
    assert server.fetch_ohlcv_batch(["AAPL"], 180, "1Day")["fetched"][0]["bars_fetched"] == 2
    clock["now"] = ts("2026-10-17T20:05:00")                                        # 16:05, 10 minutes later: same cached rows, now complete
    out = server.fetch_ohlcv_batch(["AAPL"], 180, "1Day")
    assert out["fetched"][0]["bars_fetched"] == 3 and any("disk cache" in n for n in out["data_notes"])


def test_cache_switched_off_always_asks_alpaca(wired):
    calls, _ = wired
    server.DATA_SETTINGS["cache_enabled"] = False
    server.fetch_ohlcv_batch(["AAPL", "MSFT"], 180, "1Day")
    n = len(calls)
    server.fetch_ohlcv_batch(["AAPL", "MSFT"], 180, "1Day")
    assert len(calls) == 2 * n and not (server.CACHE_DIR).exists()


def test_a_settings_warning_reaches_the_caller(wired):
    server.DATA_SETTINGS_WARNING = "config/data.yaml could not be used (test)"
    assert "could not be used" in server.fetch_ohlcv_batch(["AAPL"], 180, "1Day")["data_notes"][0]


@pytest.mark.parametrize("bad", ["1Min", "", "1day", None])
def test_unknown_timeframes_are_rejected_before_any_request(wired, bad):
    calls, _ = wired
    with pytest.raises(ValueError):
        server.fetch_ohlcv_batch(["AAPL"], 180, bad)
    with pytest.raises(ValueError):
        server.fetch_ohlcv("AAPL", 180, bad)
    assert calls == []


def test_single_symbol_fetch_uses_the_same_pipeline(wired):
    summary = server.fetch_ohlcv("AAPL", 180, "1Day")
    assert summary["bars_fetched"] == 3 and "vwap" in server._get_df("AAPL").columns
