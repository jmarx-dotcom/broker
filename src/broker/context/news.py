"""Nachrichtenbeschaffung für die Kandidaten.

Zwei keyfreie Quellen: die Meldungen des Marktdaten-Providers und der
Google-News-RSS-Feed. Beides ist Rohmaterial für die LLM-Einordnung, keine
eigenständige Analyse — Schlagzeilen allein sagen nichts über Bewertung.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote_plus

import requests

from broker.models import NewsItem
from broker.providers.base import MarketDataProvider
from broker.providers.cache import DayCache, cache_key

log = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}&hl=de&gl=DE&ceid=DE:de"
)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


def fetch_google_news(
    query: str, limit: int = 5, cache: DayCache | None = None, timeout: float = 15.0
) -> list[NewsItem]:
    """Holt Schlagzeilen über den Google-News-RSS-Feed. Kein Key nötig."""

    def fetch() -> list[NewsItem]:
        url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))
        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": "broker-screener/0.1"}
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception as exc:
            log.debug("Google-News für %r nicht abrufbar: %s", query, exc)
            return []

        items: list[NewsItem] = []
        for node in root.findall(".//item")[:limit]:
            title = (node.findtext("title") or "").strip()
            if not title:
                continue
            published = None
            raw_date = node.findtext("pubDate")
            if raw_date:
                try:
                    published = datetime.strptime(
                        raw_date, "%a, %d %b %Y %H:%M:%S %Z"
                    ).date()
                except ValueError:
                    pass
            items.append(
                NewsItem(
                    title=title,
                    publisher=node.findtext("source"),
                    published=published,
                    url=node.findtext("link"),
                    summary=_strip_html(node.findtext("description") or "") or None,
                )
            )
        return items

    if cache is None:
        return fetch()
    return cache.get_or_compute("gnews", cache_key(query, limit), fetch)


def collect_news(
    provider: MarketDataProvider,
    ticker: str,
    company_name: str | None = None,
    limit: int = 6,
    cache: DayCache | None = None,
    use_google: bool = True,
) -> list[NewsItem]:
    """Sammelt Meldungen aus beiden Quellen und dedupliziert nach Titel."""
    items: list[NewsItem] = []
    try:
        items.extend(provider.news(ticker, limit=limit))
    except Exception as exc:
        log.debug("Provider-News für %s fehlgeschlagen: %s", ticker, exc)

    if use_google and company_name:
        items.extend(fetch_google_news(f"{company_name} Aktie", limit=limit, cache=cache))

    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        key = item.title.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:limit]
