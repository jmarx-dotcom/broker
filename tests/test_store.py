"""Tests des Fundamentaldaten-Bestands.

Der Maßstab ist der Fall, aus dem er entstanden ist: Yahoo deckelt den
`.info`-Endpunkt bei rund 330 Abrufen (342, 334 und 327 an drei Tagen), das
Universum hat 674. Warten half nicht — der Deckel erholt sich nicht in Minuten.
Der Bestand muss deshalb über *mehrere Läufe* zusammentragen, was ein einzelner
nicht schafft, ohne dabei alte Zahlen als frische auszugeben.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from broker.config import Config, Thresholds
from broker.macro.regime import neutral_regime
from broker.models import Fundamentals
from broker.providers.base import ProviderError
from broker.providers.store import (
    FundamentalsStore,
    from_json,
    to_json,
)
from broker.screener import Screener
from tests.test_screener_and_report import FakeProvider, build_universe


def make(ticker: str, **kw) -> Fundamentals:
    defaults = dict(
        name=f"{ticker} AG", sector="Industrials", currency="EUR",
        market_cap=4.0e9, trailing_pe=12.0,
        quarterly_eps=pd.Series(
            [1.0, 1.1, 1.2], index=pd.to_datetime(["2025-09-30", "2025-12-31", "2026-03-31"])
        ),
    )
    defaults.update(kw)
    return Fundamentals(ticker=ticker, **defaults)  # type: ignore[arg-type]


class TestSerialisation:
    def test_roundtrip_keeps_the_numbers(self):
        original = make("SAP.DE", trailing_eps=5.5, total_debt=1.2e9)
        restored = from_json(to_json(original))

        assert restored.ticker == "SAP.DE"
        assert restored.trailing_eps == 5.5
        assert restored.total_debt == 1.2e9
        assert restored.sector == "Industrials"

    def test_roundtrip_keeps_the_time_series(self):
        """Ohne die Quartalsreihen fehlte die halbe Qualitätsanalyse."""
        restored = from_json(to_json(make("BMW.DE")))

        assert restored.quarterly_eps is not None
        assert list(restored.quarterly_eps) == [1.0, 1.1, 1.2]
        assert str(restored.quarterly_eps.index[-1].date()) == "2026-03-31"

    def test_missing_series_stays_missing(self):
        restored = from_json(to_json(make("X.DE", quarterly_eps=None)))
        assert restored.quarterly_eps is None


class TestStoreFile:
    def test_saves_and_loads(self, tmp_path):
        store = FundamentalsStore(tmp_path / "f.jsonl")
        store.put(make("A.DE"), date(2026, 8, 11))
        store.put(make("B.DE"), date(2026, 8, 11))
        assert store.save() == 2

        fresh = FundamentalsStore(tmp_path / "f.jsonl")
        assert fresh.load() == 2
        assert fresh.get("A.DE", date(2026, 8, 11)).name == "A.DE AG"

    def test_a_broken_line_does_not_cost_the_whole_store(self, tmp_path):
        """Ein kaputtes Feld darf nicht den Vorrat aller Titel kosten."""
        path = tmp_path / "f.jsonl"
        store = FundamentalsStore(path)
        store.put(make("A.DE"), date(2026, 8, 11))
        store.put(make("B.DE"), date(2026, 8, 11))
        store.save()

        path.write_text(
            path.read_text(encoding="utf-8") + "{kaputt\n", encoding="utf-8"
        )
        fresh = FundamentalsStore(path)
        assert fresh.load() == 2

    def test_empty_store_is_not_an_error(self, tmp_path):
        assert FundamentalsStore(tmp_path / "fehlt.jsonl").load() == 0


class TestAgeing:
    def test_old_entries_are_withheld(self):
        """Der Bestand überbrückt einen Ausfall, er ersetzt ihn nicht.

        Fällt der Abruf über Tage aus, altert der Bestand aus, die Abdeckung
        sinkt, und der Lauf meldet sich als unvollständig — statt mit Zahlen
        aus der Vorwoche einen vollständigen vorzutäuschen.
        """
        store = FundamentalsStore("egal", max_age_days=7)
        store.put(make("A.DE"), date(2026, 8, 1))

        assert store.get("A.DE", date(2026, 8, 8)) is not None
        assert store.get("A.DE", date(2026, 8, 9)) is None

    def test_prune_removes_what_get_would_refuse(self, tmp_path):
        store = FundamentalsStore(tmp_path / "f.jsonl", max_age_days=7)
        store.put(make("ALT.DE"), date(2026, 8, 1))
        store.put(make("NEU.DE"), date(2026, 8, 10))

        assert store.prune(date(2026, 8, 11)) == 1
        assert set(store.entries) == {"NEU.DE"}

    def test_refresh_order_puts_the_unknown_first(self):
        store = FundamentalsStore("egal")
        store.put(make("GESTERN.DE"), date(2026, 8, 10))
        store.put(make("ALT.DE"), date(2026, 8, 4))

        order = store.refresh_order(
            ["GESTERN.DE", "NEU.DE", "ALT.DE"], date(2026, 8, 11)
        )
        assert order == ["NEU.DE", "ALT.DE", "GESTERN.DE"]

    def test_refresh_order_is_stable_within_a_day(self):
        """Ein wiederholter Lauf am selben Tag muss dieselbe Hälfte auffrischen.

        Sonst frischte jeder Versuch eine andere Teilmenge auf, und das
        Universum käme nie vollständig zusammen.
        """
        store = FundamentalsStore("egal")
        tickers = [f"T{i}.DE" for i in range(20)]
        first = store.refresh_order(tickers, date(2026, 8, 11))
        second = store.refresh_order(list(reversed(tickers)), date(2026, 8, 11))
        assert first == second

    def test_drop_unknown_forgets_titles_that_left_the_universe(self):
        store = FundamentalsStore("egal")
        store.put(make("BLEIBT.DE"), date(2026, 8, 11))
        store.put(make("RAUS.DE"), date(2026, 8, 11))

        assert store.drop_unknown(["BLEIBT.DE"]) == 1
        assert set(store.entries) == {"BLEIBT.DE"}


class Capped(FakeProvider):
    """Liefert nur die ersten `cap` Fundamentaldaten-Abrufe — wie Yahoo."""

    def __init__(self, *args, cap: int, **kw):
        super().__init__(*args, **kw)
        self.cap = cap
        self.served = 0
        self.asked: list[str] = []

    def fundamentals(self, ticker: str):
        self.asked.append(ticker)
        if self.served >= self.cap:
            raise ProviderError("Too Many Requests. Rate limited.")
        self.served += 1
        return super().fundamentals(ticker)


class TestScreenerUsesTheStore:
    def _config(self):
        return Config(
            thresholds=Thresholds(min_score=0.0, min_history_days=200),
            max_candidates=20,
        )

    def test_two_runs_cover_what_one_cannot(self, tmp_path):
        """Der Kern: Ein Deckel unter der Universumsgröße, zwei Läufe reichen."""
        entries, base = build_universe()  # 8 Titel
        store = FundamentalsStore(tmp_path / "f.jsonl")

        # Tag 1: Deckel bei 4, Budget 4 — die Hälfte kommt durch.
        day1 = Capped(base._fundamentals, base._histories, cap=4)
        result1 = Screener(
            self._config(), day1, workers=2, retry_pause=0, store=store,
            refresh_budget=4, run_date=date(2026, 8, 11),
        ).run(entries, neutral_regime())
        assert result1.stats.fundamentals_ok == 4
        assert result1.stats.fundamentals_stored == 0

        # Tag 2: derselbe Deckel — aber jetzt sind die anderen vier dran, und
        # die vier von gestern kommen aus dem Bestand.
        day2 = Capped(base._fundamentals, base._histories, cap=4)
        result2 = Screener(
            self._config(), day2, workers=2, retry_pause=0, store=store,
            refresh_budget=4, run_date=date(2026, 8, 12),
        ).run(entries, neutral_regime())

        assert result2.stats.fundamentals_ok == 8, "vollständig über zwei Läufe"
        assert result2.stats.fundamentals_fresh == 4
        assert result2.stats.fundamentals_stored == 4
        assert result2.stats.oldest_stored_days == 1
        assert result2.trouble is None

    def test_the_second_run_refreshes_the_other_half(self, tmp_path):
        entries, base = build_universe()
        store = FundamentalsStore(tmp_path / "f.jsonl")

        day1 = Capped(base._fundamentals, base._histories, cap=99)
        Screener(
            self._config(), day1, workers=2, retry_pause=0, store=store,
            refresh_budget=4, run_date=date(2026, 8, 11),
        ).run(entries, neutral_regime())

        day2 = Capped(base._fundamentals, base._histories, cap=99)
        Screener(
            self._config(), day2, workers=2, retry_pause=0, store=store,
            refresh_budget=4, run_date=date(2026, 8, 12),
        ).run(entries, neutral_regime())

        assert set(day1.asked).isdisjoint(day2.asked), "keine Doppelarbeit"

    def test_a_failed_refresh_falls_back_to_the_store(self, tmp_path):
        """Ein gescheiterter Abruf ist kein Loch mehr, solange der Bestand trägt."""
        entries, base = build_universe()
        store = FundamentalsStore(tmp_path / "f.jsonl")

        good = Capped(base._fundamentals, base._histories, cap=99)
        Screener(
            self._config(), good, workers=2, retry_pause=0, store=store,
            refresh_budget=99, run_date=date(2026, 8, 11),
        ).run(entries, neutral_regime())

        blocked = Capped(base._fundamentals, base._histories, cap=0)
        result = Screener(
            self._config(), blocked, workers=2, retry_pause=0, store=store,
            refresh_budget=99, run_date=date(2026, 8, 12),
        ).run(entries, neutral_regime())

        assert result.stats.fundamentals_ok == 8
        assert result.stats.fundamentals_fresh == 0
        assert result.trouble is None

    def test_an_aged_out_store_lets_the_run_fail_loudly(self, tmp_path):
        """Nach einer Woche ohne Abruf darf der Bestand nichts mehr vortäuschen."""
        entries, base = build_universe()
        store = FundamentalsStore(tmp_path / "f.jsonl", max_age_days=7)

        good = Capped(base._fundamentals, base._histories, cap=99)
        Screener(
            self._config(), good, workers=2, retry_pause=0, store=store,
            refresh_budget=99, run_date=date(2026, 8, 1),
        ).run(entries, neutral_regime())

        blocked = Capped(base._fundamentals, base._histories, cap=0)
        result = Screener(
            self._config(), blocked, workers=2, retry_pause=0, store=store,
            refresh_budget=99, run_date=date(2026, 8, 20),
        ).run(entries, neutral_regime())

        assert result.stats.fundamentals_ok == 0
        assert result.degraded
        assert "0 von 8" in result.trouble

    def test_small_universes_are_simply_all_fresh(self, tmp_path):
        """Der DAX-Lauf über 40 Titel hatte nie ein Drosselungsproblem.

        Passt das Universum ins Budget, wird nichts aus dem Bestand ergänzt —
        er wird aber gefüllt und bleibt Rückfallebene für Einzelausfälle.
        """
        entries, base = build_universe()
        store = FundamentalsStore(tmp_path / "f.jsonl")
        provider = Capped(base._fundamentals, base._histories, cap=99)

        result = Screener(
            self._config(), provider, workers=2, retry_pause=0, store=store,
            refresh_budget=280, run_date=date(2026, 8, 11),
        ).run(entries, neutral_regime())

        assert result.stats.fundamentals_fresh == 8
        assert result.stats.fundamentals_stored == 0
        # Trotzdem gefüllt — der nächste größere Lauf profitiert davon.
        assert len(store.entries) == 8

    def test_missing_titles_are_named_not_silent(self, tmp_path):
        entries, base = build_universe()
        store = FundamentalsStore(tmp_path / "f.jsonl")
        provider = Capped(base._fundamentals, base._histories, cap=99)

        result = Screener(
            self._config(), provider, workers=2, retry_pause=0, store=store,
            refresh_budget=3, run_date=date(2026, 8, 11),
        ).run(entries, neutral_regime())

        assert result.stats.fundamentals_ok == 3
        missing = [m for m in result.stats.errors.values() if "nicht im Bestand" in m]
        assert len(missing) == 5, "die übersprungenen Titel stehen im Bericht"

    def test_summary_names_the_stored_share(self, tmp_path):
        entries, base = build_universe()
        store = FundamentalsStore(tmp_path / "f.jsonl")

        Screener(
            self._config(), Capped(base._fundamentals, base._histories, cap=99),
            workers=2, retry_pause=0, store=store, refresh_budget=4,
            run_date=date(2026, 8, 11),
        ).run(entries, neutral_regime())
        result = Screener(
            self._config(), Capped(base._fundamentals, base._histories, cap=99),
            workers=2, retry_pause=0, store=store, refresh_budget=4,
            run_date=date(2026, 8, 12),
        ).run(entries, neutral_regime())

        assert "aus dem Bestand" in result.stats.summary()
        assert "ältester 1 Tage" in result.stats.summary()
