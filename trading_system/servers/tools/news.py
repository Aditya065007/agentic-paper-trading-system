"""Slow, optional news and financial-sentiment tools for the MCP server."""

import copy
import json
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FINBERT_MODEL = "ProsusAI/finbert"
_news_cache: dict[tuple, dict] = {}
_finbert_pipeline = None

# A symbol's news score is shrunk toward 0 when it rests on fewer scored headlines than this:
# one cautionary quote must not swing a composite (seen live: a single JPM headline scored -0.93).
MIN_HEADLINES_FOR_FULL_CONFIDENCE = 3
# Per-symbol fetch: one NewsAPI request per symbol, cached on disk, with a daily request budget.
# The free plan allows 100 requests a day (counting day assumed to be UTC; [VERIFY]); the default
# leaves headroom and can be changed with the NEWSAPI_DAILY_LIMIT environment variable.
DEFAULT_DAILY_REQUEST_LIMIT = 80
NEWS_CACHE_TTL_MINUTES = 120
_NEWS_PREFIX = "news"
_BUDGET_FILE = "news_budget.json"


def _clean_symbols(symbols) -> list[str]:
    if isinstance(symbols, str):
        symbols = symbols.replace(",", " ").split()
    if not isinstance(symbols, (list, tuple)):
        return []
    return sorted({str(symbol).strip().upper() for symbol in symbols
                   if str(symbol).strip()})


def _window(lookback_days: int) -> tuple[str, str]:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=lookback_days - 1)
    return start.isoformat(), end.isoformat()


def _article_key(article: dict) -> tuple:
    url = article.get("url")
    if isinstance(url, str) and url.strip():
        return ("url", url.strip().casefold())
    source = article.get("source")
    source_name = source.get("name") if isinstance(source, dict) else None
    return ("text", article.get("title", "").strip().casefold(),
            article.get("publishedAt"), str(source_name or "").casefold())


def _response(headlines=None, warnings=None, errors=None, as_of=None) -> dict:
    articles = headlines if isinstance(headlines, list) else []
    warning_list = list(warnings or [])
    error_list = list(errors or [])
    return {
        "as_of": as_of or datetime.now(timezone.utc).isoformat(),
        "headlines": articles,
        "article_count": len(articles),
        "available": bool(articles),
        "warnings": warning_list,
        "errors": error_list,
        "degraded": bool(warning_list or error_list),
    }


def _parse_articles(payload: dict, symbols: list[str]) -> tuple[list[dict], list[str]]:
    warnings = []
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return [], ["NewsAPI returned an unsuccessful or malformed response."]

    articles = payload.get("articles")
    if not isinstance(articles, list):
        return [], ["NewsAPI response did not contain a valid articles list."]

    parsed = []
    seen = set()
    for article in articles:
        if not isinstance(article, dict):
            warnings.append("A malformed NewsAPI article was skipped.")
            continue
        title = article.get("title")
        if not isinstance(title, str) or not title.strip():
            warnings.append("A NewsAPI article without a title was skipped.")
            continue
        key = _article_key(article)
        if key in seen:
            continue
        seen.add(key)

        source = article.get("source")
        source_name = source.get("name") if isinstance(source, dict) else None
        published_at = article.get("publishedAt")
        description = article.get("description")
        searchable = " ".join((title, description if isinstance(description, str) else ""))
        for symbol in symbols:
            if re.search(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])",
                         searchable.upper()):
                parsed.append({
                    "symbol": symbol,
                    "title": title.strip(),
                    "source": source_name if isinstance(source_name, str) else None,
                    "published_at": published_at if isinstance(published_at, str) else None,
                    "url": article.get("url") if isinstance(article.get("url"), str) else None,
                })
    return parsed, warnings


def get_news(symbols: list[str], lookback_days: int = 3) -> dict:
    """Fetch a batched, recent NewsAPI result set; this is not a trading signal."""
    cleaned = _clean_symbols(symbols)
    if not cleaned:
        return _response(warnings=["Provide at least one symbol."],
                         errors=["No valid symbols were provided."])
    if not isinstance(lookback_days, int) or not 1 <= lookback_days <= 29:
        return _response(warnings=["lookback_days must be between 1 and 29."],
                         errors=["Invalid lookback window."])

    api_key = os.environ.get("NEWSAPI_KEY", "").strip()
    if not api_key:
        return _response(
            warnings=["NEWSAPI_KEY is not configured; no headlines were fetched."],
            errors=["NewsAPI credentials are unavailable."],
        )

    from_date, to_date = _window(lookback_days)
    query = " OR ".join(f'"{symbol}"' for symbol in cleaned)
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "sortBy": "publishedAt",
        "pageSize": 100,
        "language": "en",
    }
    cache_key = (tuple(cleaned), query, from_date, to_date, params["pageSize"], params["language"])
    if cache_key in _news_cache:
        return copy.deepcopy(_news_cache[cache_key])

    try:
        response = requests.get(NEWSAPI_URL, params=params,
                               headers={"X-Api-Key": api_key}, timeout=10)
        response.raise_for_status()
        payload = response.json()
        headlines, warnings = _parse_articles(payload, cleaned)
        result = _response(headlines, warnings)
        if not headlines and not warnings:
            result["warnings"].append("NewsAPI returned no matching headlines.")
            result["degraded"] = True
    except requests.Timeout:
        result = _response(
            warnings=["NewsAPI request timed out; no headlines were fetched."],
            errors=["NewsAPI timeout."],
        )
    except requests.RequestException:
        result = _response(
            warnings=["NewsAPI request failed; no headlines were fetched."],
            errors=["NewsAPI request failure."],
        )
    except (TypeError, ValueError):
        result = _response(
            warnings=["NewsAPI response could not be parsed."],
            errors=["Malformed NewsAPI response."],
        )

    _news_cache[cache_key] = copy.deepcopy(result)
    return result


def _cache_dir() -> Path:
    return Path(os.environ.get("TRADING_DATA_CACHE_DIR")
                or Path(__file__).resolve().parents[2] / "data_cache")


def _daily_limit() -> int:
    try:
        value = int(os.environ.get("NEWSAPI_DAILY_LIMIT", DEFAULT_DAILY_REQUEST_LIMIT))
    except ValueError:
        return DEFAULT_DAILY_REQUEST_LIMIT
    return value if value >= 0 else DEFAULT_DAILY_REQUEST_LIMIT


def _budget_used(directory: Path, today: str) -> int:
    try:
        payload = json.loads((directory / _BUDGET_FILE).read_text(encoding="utf-8"))
        if payload.get("date") == today and isinstance(payload.get("count"), int):
            return max(0, payload["count"])
    except Exception:
        pass
    return 0


def _budget_save(directory: Path, today: str, count: int) -> None:
    """Best effort: a failed write never blocks news, it only loses the tally."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump({"date": today, "count": count}, stream)
            os.replace(temp_name, directory / _BUDGET_FILE)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    except Exception:
        pass


def get_news_by_symbol(symbols: list[str], lookback_days: int = 3) -> dict:
    """Fetch recent headlines with ONE NewsAPI request per symbol, so a busy ticker cannot crowd out a
    quiet one (the batched get_news shares one 100-article page across every symbol).

    Results are cached on disk for NEWS_CACHE_TTL_MINUTES (cache hits cost no request). Requests are
    counted against a daily budget; once it is used up the remaining symbols are skipped with a warning.
    The result has the same shape as get_news, so score_headlines accepts it unchanged. This is not a
    trading signal."""
    from . import data as m1data

    cleaned = _clean_symbols(symbols)
    if not cleaned:
        return _response(warnings=["Provide at least one symbol."],
                         errors=["No valid symbols were provided."])
    if (isinstance(lookback_days, bool) or not isinstance(lookback_days, int)
            or not 1 <= lookback_days <= 29):
        return _response(warnings=["lookback_days must be between 1 and 29."],
                         errors=["Invalid lookback window."])
    api_key = os.environ.get("NEWSAPI_KEY", "").strip()
    if not api_key:
        return _response(
            warnings=["NEWSAPI_KEY is not configured; no headlines were fetched."],
            errors=["NewsAPI credentials are unavailable."],
        )

    from_date, to_date = _window(lookback_days)
    directory = _cache_dir()
    today = datetime.now(timezone.utc).date().isoformat()
    limit = _daily_limit()
    used = _budget_used(directory, today)
    headlines, warnings, errors = [], [], []
    requests_made = cache_hits = 0
    skipped, empty, stop_reason = [], [], None

    for symbol in cleaned:
        key = m1data.cache_key([symbol], _NEWS_PREFIX, lookback_days, "newsapi", f"{from_date}:{to_date}")
        cached = m1data.read_cache(directory, key, NEWS_CACHE_TTL_MINUTES, prefix=_NEWS_PREFIX)
        if cached is not None and isinstance(cached.get("headlines"), list):
            cache_hits += 1
            found = cached["headlines"]
            warnings.extend(value for value in cached.get("warnings", []) if isinstance(value, str))
        else:
            if stop_reason is None and used >= limit:
                stop_reason = f"daily request budget of {limit} reached"
            if stop_reason is not None:
                skipped.append(symbol)
                continue
            params = {"q": f'"{symbol}"', "from": from_date, "to": to_date,
                      "sortBy": "publishedAt", "pageSize": 100, "language": "en"}
            used += 1
            requests_made += 1
            _budget_save(directory, today, used)
            try:
                response = requests.get(NEWSAPI_URL, params=params,
                                        headers={"X-Api-Key": api_key}, timeout=10)
                if response.status_code == 429:
                    used = max(used, limit)
                    _budget_save(directory, today, used)
                    stop_reason = "NewsAPI rate limit reached (HTTP 429)"
                    skipped.append(symbol)
                    continue
                response.raise_for_status()
                found, parse_warnings = _parse_articles(response.json(), [symbol])
                if any("unsuccessful or malformed" in value for value in parse_warnings):
                    errors.append(f"NewsAPI returned an unsuccessful response for {symbol}.")
                    warnings.extend(parse_warnings)
                    continue
                warnings.extend(parse_warnings)
                m1data.write_cache(directory, key, {"headlines": found, "warnings": parse_warnings},
                                   prefix=_NEWS_PREFIX)
            except requests.Timeout:
                errors.append(f"NewsAPI timeout for {symbol}.")
                warnings.append(f"NewsAPI request for {symbol} timed out; no headlines were fetched.")
                continue
            except requests.RequestException:
                errors.append(f"NewsAPI request failure for {symbol}.")
                warnings.append(f"NewsAPI request for {symbol} failed; no headlines were fetched.")
                continue
            except (TypeError, ValueError):
                errors.append(f"Malformed NewsAPI response for {symbol}.")
                warnings.append(f"NewsAPI response for {symbol} could not be parsed.")
                continue
        if not found:
            empty.append(symbol)
        headlines.extend(found)

    if skipped:
        warnings.append(f"News skipped for {', '.join(skipped)}: {stop_reason}.")
    if empty:
        warnings.append(f"No matching headlines for {', '.join(empty)}.")
    result = _response(headlines, list(dict.fromkeys(warnings)), errors)
    result["budget"] = {"limit": limit, "used_today": used,
                        "requests_made": requests_made, "cache_hits": cache_hits}
    return result


def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_MODEL, local_files_only=True)
        _finbert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer,
                                     top_k=None)
    return _finbert_pipeline


def _probabilities(text: str) -> dict[str, float]:
    predictions = _load_finbert()(text, truncation=True)
    if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    if not isinstance(predictions, list):
        raise ValueError("Unexpected FinBERT output")
    values = {}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        label = str(prediction.get("label", "")).casefold()
        score = prediction.get("score")
        if (label in {"positive", "negative", "neutral"}
                and isinstance(score, (int, float)) and math.isfinite(score)
                and 0.0 <= score <= 1.0):
            values[label] = float(score)
    if not all(label in values for label in ("positive", "negative", "neutral")):
        raise ValueError("FinBERT did not return all sentiment classes")
    return values


def _aggregate(symbol, articles) -> dict:
    available_scores = [article["sentiment_score"] for article in articles
                        if isinstance(article.get("sentiment_score"), (int, float))]
    if not available_scores:
        return {
            "symbol": symbol,
            "news_score": None,
            "sentiment": "unavailable",
            "sentiment_available": False,
            "article_count": len(articles),
            "scored_article_count": 0,
        }
    raw = max(-1.0, min(1.0, sum(available_scores) / len(available_scores)))
    confidence = min(1.0, len(available_scores) / MIN_HEADLINES_FOR_FULL_CONFIDENCE)
    score = raw * confidence
    sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return {
        "symbol": symbol,
        "news_score": round(score, 6),
        "raw_news_score": round(raw, 6),
        "confidence": round(confidence, 4),
        "sentiment": sentiment,
        "sentiment_available": True,
        "article_count": len(articles),
        "scored_article_count": len(available_scores),
    }


def score_headlines(headlines: list[dict] | dict) -> dict:
    """Return per-headline sentiment and an available per-symbol aggregate score."""
    if isinstance(headlines, dict):
        source = headlines.get("headlines")
        inherited_warnings = [value for value in headlines.get("warnings", [])
                              if isinstance(value, str)] \
            if isinstance(headlines.get("warnings", []), (list, tuple)) else []
        inherited_errors = [value for value in headlines.get("errors", [])
                            if isinstance(value, str)] \
            if isinstance(headlines.get("errors", []), (list, tuple)) else []
        as_of = headlines.get("as_of")
    else:
        source = headlines
        inherited_warnings, inherited_errors, as_of = [], [], None
    warnings = list(inherited_warnings)
    errors = list(inherited_errors)
    if not isinstance(source, list):
        warnings.append("Headline input must be a list or get_news result.")
        errors.append("Invalid headline input.")
        source = []

    normalized = []
    seen = set()
    for item in source:
        if isinstance(item, str):
            item = {"title": item}
        if not isinstance(item, dict):
            warnings.append("A malformed headline was skipped.")
            continue
        title = item.get("title", item.get("headline"))
        if not isinstance(title, str) or not title.strip():
            warnings.append("A headline without valid title text was skipped.")
            continue
        symbol = item.get("symbol")
        symbol = symbol.strip().upper() if isinstance(symbol, str) and symbol.strip() else None
        article = {
            "symbol": symbol,
            "title": title.strip(),
            "source": item.get("source") if isinstance(item.get("source"), str) else None,
            "published_at": item.get("published_at", item.get("publishedAt"))
            if isinstance(item.get("published_at", item.get("publishedAt")), str) else None,
            "url": item.get("url") if isinstance(item.get("url"), str) else None,
        }
        identity = (symbol, _article_key(article))
        if identity in seen:
            warnings.append("Duplicate headline was removed before sentiment scoring.")
            continue
        seen.add(identity)
        normalized.append(article)

    if not normalized:
        if not warnings:
            warnings.append("No headlines are available for sentiment scoring.")
        return {
            "as_of": as_of or datetime.now(timezone.utc).isoformat(),
            "scores": [],
            "by_symbol": {},
            "news_score": None,
            "sentiment": "unavailable",
            "sentiment_available": False,
            "article_count": 0,
            "scored_article_count": 0,
            "warnings": warnings,
            "errors": errors,
            "degraded": True,
        }

    scored = []
    try:
        _load_finbert()
    except Exception:
        errors.append("FinBERT is unavailable locally.")
        warnings.append("FinBERT is unavailable; sentiment scores were not produced.")
        for article in normalized:
            scored.append({**article, "sentiment": "unavailable", "label": "unavailable",
                           "score": None, "sentiment_score": None, "probabilities": None,
                           "model": FINBERT_MODEL})
    else:
        for article in normalized:
            try:
                probabilities = _probabilities(article["title"])
                value = max(-1.0, min(1.0,
                                      probabilities["positive"] - probabilities["negative"]))
                sentiment = "positive" if value > 0 else "negative" if value < 0 else "neutral"
                probabilities = {key: round(score, 6) for key, score in probabilities.items()}
                scored.append({**article, "sentiment": sentiment, "label": sentiment,
                               "score": round(value, 6),
                               "sentiment_score": round(value, 6),
                               "probabilities": probabilities, "model": FINBERT_MODEL})
            except Exception:
                errors.append("FinBERT failed to score one or more headlines.")
                warnings.append("A headline sentiment is unavailable after model inference failure.")
                scored.append({**article, "sentiment": "unavailable", "label": "unavailable",
                               "score": None, "sentiment_score": None, "probabilities": None,
                               "model": FINBERT_MODEL})

    grouped = {}
    for article in scored:
        grouped.setdefault(article["symbol"], []).append(article)
    by_symbol = {symbol: _aggregate(symbol, articles)
                 for symbol, articles in grouped.items() if symbol is not None}
    if len(grouped) == 1:
        aggregate = next(iter(grouped.values()))
        root_signal = _aggregate(aggregate[0]["symbol"], aggregate)
    else:
        root_signal = _aggregate(None, [])
        warnings.append("Multiple symbols supplied; use by_symbol aggregates for downstream scoring.")
    scored_count = sum(article["sentiment_score"] is not None for article in scored)
    if scored_count and scored_count < len(scored):
        warnings.append("Some headlines were unavailable; aggregate uses successfully scored headlines only.")
    if not scored_count:
        warnings.append("No sentiment score is available; news_score remains null.")
    return {
        "as_of": as_of or datetime.now(timezone.utc).isoformat(),
        "scores": scored,
        "by_symbol": by_symbol,
        "news_score": root_signal["news_score"],
        "sentiment": root_signal["sentiment"],
        "sentiment_available": root_signal["sentiment_available"],
        "article_count": len(scored),
        "scored_article_count": scored_count,
        "warnings": warnings,
        "errors": errors,
        "degraded": bool(warnings or errors),
    }


def register_news_tools(mcp) -> None:
    mcp.tool()(get_news)
    mcp.tool()(score_headlines)