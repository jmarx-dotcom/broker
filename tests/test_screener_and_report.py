"""Integrationstest der gesamten Pipeline gegen einen Fake-Provider."""

from __future__ import annotations

import json

import pytest

from broker.config import Config, Thresholds
from broker.macro.regime import build_regime, neutral_regime
from broker.models import Fundamentals, NewsItem
from broker.providers.base import ProviderError
from broker.report.html import render_html, write_json, write_report
from broker.screener import Screener
from broker.universe import UniverseEntry
from tests.conftest import make_history, make_quarterly_eps


class FakeProvider:
    """Provider mit vorgegebenen Daten — kein Netzwerk im Spiel."""

    name = "fake"

    def __init__(self, fundamentals: dict[str, Fundamentals], histories: dict):
        self._fundamentals = fundamentals
        self._histories = histories
        self.history_calls: list[str] = []

    def fundamentals(self, ticker: str) -> Fundamentals:
        if ticker not in self._fundamentals:
            raise ProviderError(f"kein Eintrag für {ticker}")
        return self._fundamentals[ticker]

    def history(self, ticker: str, period: str = "3y"):
        self.history_calls.append(ticker)
        if ticker not in self._histories:
            raise ProviderError(f"keine Historie für {ticker}")
        return self._histories[ticker]

    def news(self, ticker: str, limit: int = 5) -> list[NewsItem]:
        return [NewsItem(title=f"Meldung zu {ticker}")]


def build_universe() -> tuple[list[UniverseEntry], FakeProvider]:
    entries = [
        UniverseEntry("CHEAP.DE", "DAX", "DE"),
        UniverseEntry("RICH.DE", "DAX", "DE"),
        UniverseEntry("TRAP.DE", "DAX", "DE"),
        UniverseEntry("LOSS.DE", "DAX", "DE"),
        UniverseEntry("TINY.DE", "DAX", "DE"),
        UniverseEntry("BROKEN.DE", "DAX", "DE"),
        UniverseEntry("PEER1.DE", "DAX", "DE"),
        UniverseEntry("PEER2.DE", "DAX", "DE"),
    ]

    def base(ticker: str, **kwargs) -> Fundamentals:
        defaults = dict(
            ticker=ticker,
            name=ticker.split(".")[0].title() + " AG",
            sector="Industrials",
            country="Germany",
            currency="EUR",
            market_cap=4.0e9,
            total_debt=8.0e8,
            total_cash=5.0e8,
            ebitda=7.0e8,
            free_cashflow=3.0e8,
            revenue_growth=0.05,
            earnings_growth=0.08,
            return_on_equity=0.16,
            profit_margin=0.10,
            quarterly_eps=make_quarterly_eps(growth=0.02),
        )
        defaults.update(kwargs)
        return Fundamentals(**defaults)  # type: ignore[arg-type]

    fundamentals = {
        "CHEAP.DE": base("CHEAP.DE", trailing_pe=9.0, forward_pe=8.0),
        "RICH.DE": base("RICH.DE", trailing_pe=48.0, forward_pe=52.0),
        "TRAP.DE": base(
            "TRAP.DE",
            trailing_pe=6.0,
            forward_pe=15.0,
            revenue_growth=-0.20,
            earnings_growth=-0.50,
            profit_margin=-0.03,
            return_on_equity=-0.06,
            free_cashflow=-2.0e8,
            payout_ratio=1.5,
            total_debt=4.0e9,
            total_cash=5.0e7,
        ),
        "LOSS.DE": base("LOSS.DE", trailing_pe=-12.0),
        "TINY.DE": base("TINY.DE", trailing_pe=8.0, market_cap=1.0e7),
        "BROKEN.DE": base("BROKEN.DE", trailing_pe=11.0),
        "PEER1.DE": base("PEER1.DE", trailing_pe=19.0),
        "PEER2.DE": base("PEER2.DE", trailing_pe=21.0),
    }

    histories = {
        t: make_history(ticker=t, days=600, trend=-0.0004, noise=0.008, seed=i)
        for i, t in enumerate(fundamentals)
        if t != "BROKEN.DE"  # BROKEN liefert absichtlich keine Historie
    }
    histories["^GDAXI"] = make_history(ticker="^GDAXI", days=600, trend=0.0003)
    histories["^GSPC"] = make_history(ticker="^GSPC", days=600, trend=0.0004)
    histories["^STOXX50E"] = make_history(ticker="^STOXX50E", days=600, trend=0.0002)

    return entries, FakeProvider(fundamentals, histories)


@pytest.fixture
def screening_result():
    entries, provider = build_universe()
    config = Config(
        thresholds=Thresholds(min_score=0.0, min_history_days=200), max_candidates=20
    )
    return Screener(config, provider, workers=2).run(entries, neutral_regime()), provider


class TestScreener:
    def test_filters_remove_the_obvious_rejects(self, screening_result):
        result, _ = screening_result
        tickers = {c.ticker for c in result.candidates}

        assert "LOSS.DE" not in tickers, "negatives KGV muss rausfliegen"
        assert "TINY.DE" not in tickers, "zu kleine Marktkapitalisierung"
        assert "RICH.DE" not in tickers, "KGV über der Obergrenze"
        assert "CHEAP.DE" in tickers

    def test_missing_history_is_recorded_not_fatal(self, screening_result):
        result, _ = screening_result
        assert "BROKEN.DE" in result.stats.errors
        assert result.stats.scored > 0

    def test_cheap_and_solid_ranks_above_value_trap(self, screening_result):
        result, _ = screening_result
        by_ticker = {c.ticker: c for c in result.candidates}

        assert by_ticker["CHEAP.DE"].total_score > by_ticker["TRAP.DE"].total_score

    def test_value_trap_carries_red_flags(self, screening_result):
        result, _ = screening_result
        trap = next(c for c in result.candidates if c.ticker == "TRAP.DE")

        assert len(trap.quality.red_flags) >= 3

    def test_stats_are_consistent(self, screening_result):
        result, _ = screening_result
        stats = result.stats

        assert stats.universe_size == 8
        assert stats.fundamentals_ok == 8
        assert stats.passed_filters <= stats.fundamentals_ok
        assert stats.scored <= stats.passed_filters
        assert "8 Titel im Universum" in stats.summary()

    def test_candidates_are_sorted_by_score(self, screening_result):
        result, _ = screening_result
        scores = [c.total_score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_threshold_is_applied(self):
        entries, provider = build_universe()
        config = Config(thresholds=Thresholds(min_score=99.9, min_history_days=200))
        result = Screener(config, provider, workers=2).run(entries, neutral_regime())
        assert result.candidates == []

    def test_max_candidates_caps_the_output(self):
        entries, provider = build_universe()
        config = Config(
            thresholds=Thresholds(min_score=0.0, min_history_days=200), max_candidates=1
        )
        result = Screener(config, provider, workers=2).run(entries, neutral_regime())
        assert len(result.candidates) == 1

    def test_benchmark_is_fetched_for_relative_strength(self, screening_result):
        _, provider = screening_result
        assert "^GDAXI" in provider.history_calls


class TestReport:
    def test_html_contains_candidates_and_disclaimer(self, screening_result):
        result, _ = screening_result
        html = render_html(result, universe_label="dax")

        assert "Aktien-Screening" in html
        assert "CHEAP.DE" in html
        assert "keine Anlageberatung" in html
        assert "dax" in html

    def test_html_handles_empty_result(self):
        entries, provider = build_universe()
        config = Config(thresholds=Thresholds(min_score=99.9, min_history_days=200))
        result = Screener(config, provider, workers=2).run(entries, neutral_regime())

        html = render_html(result)
        assert "Kein Titel hat die Score-Schwelle erreicht" in html

    def test_html_escapes_untrusted_news_titles(self, screening_result):
        result, _ = screening_result
        result.candidates[0].news = [
            NewsItem(title="<script>alert(1)</script>", url="https://example.com")
        ]
        html = render_html(result)

        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_writes_html_file(self, screening_result, tmp_path):
        result, _ = screening_result
        path = write_report(result, tmp_path, universe_label="dax")

        assert path.is_file()
        assert path.suffix == ".html"
        assert "CHEAP.DE" in path.read_text(encoding="utf-8")

    def test_writes_valid_json(self, screening_result, tmp_path):
        result, _ = screening_result
        path = write_json(result, tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["stats"]["universe_size"] == 8
        assert len(payload["candidates"]) == len(result.candidates)
        assert "valuation" in payload["candidates"][0]
        assert "macro" in payload
