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


class Throttling(FakeProvider):
    """Weist die ersten `n` Abrufe je Art ab — wie Yahoo unter Last.

    Nicht "dieser Titel ist weg", sondern "gerade nicht, versuch es später":
    Dieselben Titel antworten in der zweiten Runde.
    """

    def __init__(self, *args, reject_first: int = 0, **kw):
        super().__init__(*args, **kw)
        self.remaining = reject_first
        self.rounds: list[str] = []

    def _throttle(self, ticker: str) -> None:
        if self.remaining > 0:
            self.remaining -= 1
            raise ProviderError(f"Too Many Requests für {ticker}")

    def fundamentals(self, ticker: str):
        self._throttle(ticker)
        return super().fundamentals(ticker)

    def history(self, ticker: str, period: str = "3y"):
        self._throttle(ticker)
        return super().history(ticker, period)


class TestDegradedRun:
    """Der Lauf vom 4. und 5. August: gemeldet als 'keine Treffer'.

    Tatsächlich waren von 674 Titeln 342 mit Daten angekommen, von 217
    gefilterten wurde genau einer bewertet, 548 Abrufe schlugen fehl. Die
    Nachricht war beruhigend und falsch. Ein Lauf, der fast nichts gesehen
    hat, muss das sagen — sonst ist eine Störung von einem ruhigen Markt
    nicht zu unterscheiden.
    """

    def test_intact_run_reports_no_trouble(self, screening_result):
        result, _ = screening_result
        assert result.trouble is None
        assert result.degraded is False

    def test_missing_scores_are_flagged_as_trouble(self):
        from broker.screener import ScreeningStats

        # Die realen Zahlen vom 5. August.
        stats = ScreeningStats(
            universe_size=674, fundamentals_ok=342, passed_filters=217, scored=1,
            errors={f"T{i}": "Kurshistorie" for i in range(548)},
        )
        assert stats.degraded
        assert "217" in stats.trouble and "1" in stats.trouble
        assert "nie geprüft" in stats.trouble

    def test_missing_data_is_flagged_even_without_filter_stage(self):
        from broker.screener import ScreeningStats

        stats = ScreeningStats(universe_size=674, fundamentals_ok=100)
        assert stats.degraded
        assert "100 von 674" in stats.trouble

    def test_empty_hit_list_alone_is_not_trouble(self):
        """Ein wirklich ruhiger Markt bleibt eine gültige Aussage."""
        from broker.screener import ScreeningStats

        stats = ScreeningStats(
            universe_size=100, fundamentals_ok=98, passed_filters=40, scored=40
        )
        assert stats.trouble is None

    def test_message_says_unvollstaendig_not_keine_treffer(self):
        from broker.notify import build_summary

        entries, provider = build_universe()
        config = Config(thresholds=Thresholds(min_score=0.0, min_history_days=200))
        result = Screener(config, provider, workers=2).run(entries, neutral_regime())
        result.stats.fundamentals_ok = 1  # Lauf nachträglich als gestört markieren

        summary = build_summary(result)
        assert "unvollständig" in summary
        assert "keine Treffer über der Score-Schwelle" not in summary
        assert "Journal" in summary


class TestRetryOnThrottling:
    """Yahoo weist unter Last ab — dieselben Titel antworten kurz darauf.

    Der DAX-Lauf über 40 Titel hatte null Fehler, der Lauf über 674 hatte 548.
    Nicht die Titel waren das Problem, sondern ihre Zahl.
    """

    def test_throttled_calls_are_recovered_in_the_second_round(self):
        entries, base = build_universe()
        provider = Throttling(
            base._fundamentals, base._histories, reject_first=6
        )
        config = Config(
            thresholds=Thresholds(min_score=0.0, min_history_days=200),
            max_candidates=20,
        )
        result = Screener(
            config, provider, workers=2, retry_pause=0
        ).run(entries, neutral_regime())

        # Ohne zweite Runde wären sechs Titel ohne Fundamentaldaten geblieben.
        assert result.stats.fundamentals_ok == 8
        assert result.trouble is None

    def test_few_failures_do_not_trigger_the_slow_round(self, monkeypatch):
        """Zwanzig Sekunden Pause wegen eines delisteten Titels wären verschwendet."""
        from broker import screener as screener_module

        slept: list[float] = []
        monkeypatch.setattr(screener_module.time, "sleep", lambda s: slept.append(s))

        entries, provider = build_universe()  # BROKEN.DE fällt aus, sonst nichts
        config = Config(
            thresholds=Thresholds(min_score=0.0, min_history_days=200),
            max_candidates=20,
        )
        Screener(config, provider, workers=2).run(entries, neutral_regime())

        assert slept == []

    def test_broad_failure_waits_before_retrying(self, monkeypatch):
        from broker import screener as screener_module

        slept: list[float] = []
        monkeypatch.setattr(screener_module.time, "sleep", lambda s: slept.append(s))

        entries, base = build_universe()
        provider = Throttling(base._fundamentals, base._histories, reject_first=6)
        config = Config(thresholds=Thresholds(min_score=0.0, min_history_days=200))
        Screener(config, provider, workers=2, retry_pause=7.5).run(
            entries, neutral_regime()
        )

        assert slept == [7.5]

    def test_several_windows_are_waited_out(self, monkeypatch):
        """Ein Fenster von ~340 Abrufen reicht für ~900 nicht — mehrere schon.

        Am 5. und 6. August kamen 342 und 334 Titel durch, dann sperrte Yahoo.
        Ein Lauf über das ganze Universum braucht rund 900 Abrufe. Der einzige
        Weg dahin ist das nächste Zeitfenster.
        """
        from broker import screener as screener_module

        slept: list[float] = []
        monkeypatch.setattr(screener_module.time, "sleep", lambda s: slept.append(s))

        from broker.screener import ScreeningStats

        # Ein Fenster lässt 40 Abrufe durch, dann sperrt es bis zur nächsten
        # Runde — dasselbe Verhältnis wie 340 von 900 in echt.
        window = {"left": 40}

        def call(key: str) -> str:
            if window["left"] <= 0:
                raise ProviderError("Too Many Requests. Rate limited.")
            window["left"] -= 1
            return key

        def open_window(_seconds):
            slept.append(_seconds)
            window["left"] = 40

        monkeypatch.setattr(screener_module.time, "sleep", open_window)

        config = Config(thresholds=Thresholds())
        screener = Screener(config, provider=None, workers=4, retry_pause=5)  # type: ignore[arg-type]
        stats = ScreeningStats()
        found = screener._gather(
            [f"T{i}" for i in range(100)], call, "Fundamentaldaten", stats
        )

        assert len(found) == 100, "alle Titel über mehrere Fenster geholt"
        assert stats.errors == {}
        assert len(slept) == 2, "zwei Fenster abgewartet"

    def test_waiting_stops_when_a_round_brings_nothing_back(self, monkeypatch):
        """Die eigentliche Abbruchregel: Erholung entscheidet, nicht die Rundenzahl.

        Bringt eine Pause keinen einzigen Titel zurück, ist es keine
        Drosselung — dann ist weiteres Warten nur ein Lauf, der ins Zeitlimit
        kriecht, statt das Problem zu melden.
        """
        from broker import screener as screener_module

        slept: list[float] = []
        monkeypatch.setattr(screener_module.time, "sleep", lambda s: slept.append(s))

        entries, base = build_universe()
        # Vier Titel sind dauerhaft weg — keine Runde holt sie zurück.
        gone = dict(base._fundamentals)
        for ticker in ("PEER1.DE", "PEER2.DE", "TINY.DE", "LOSS.DE"):
            del gone[ticker]
        provider = FakeProvider(gone, base._histories)
        config = Config(thresholds=Thresholds(min_score=0.0, min_history_days=200))
        Screener(config, provider, workers=2, retry_pause=5).run(
            entries, neutral_regime()
        )

        assert len(slept) == 1, "nach der ersten ergebnislosen Runde Schluss"


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

    def test_html_distinguishes_a_broken_run_from_a_quiet_market(self):
        entries, provider = build_universe()
        config = Config(thresholds=Thresholds(min_score=99.9, min_history_days=200))
        result = Screener(config, provider, workers=2).run(entries, neutral_regime())
        result.stats.fundamentals_ok = 1

        html = render_html(result)
        assert "Lauf unvollständig" in html
        assert "Das ist ein gültiges Ergebnis" not in html

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
