from __future__ import annotations

import pandas as pd

from broker.analysis.valuation import (
    analyze_valuation,
    build_pe_history,
    sector_median_pe,
)
from broker.models import Fundamentals
from tests.conftest import make_history, make_quarterly_eps


class TestPeHistory:
    def test_builds_series_from_price_and_eps(self):
        history = make_history(days=750, start=100.0)
        eps = make_quarterly_eps(quarters=16, base=1.0)

        pe = build_pe_history(history, eps)

        assert pe is not None
        # Flacher Kurs bei 100 und TTM-EPS von 4.0 ergibt ein KGV um 25.
        assert 24.0 < float(pe.median()) < 26.0

    def test_returns_none_without_enough_quarters(self):
        history = make_history(days=750)
        assert build_pe_history(history, make_quarterly_eps(quarters=4)) is None

    def test_returns_none_without_eps(self):
        assert build_pe_history(make_history(), None) is None

    def test_ignores_negative_eps_periods(self):
        history = make_history(days=750)
        eps = make_quarterly_eps(quarters=16, base=-1.0)

        # Durchgehend negatives EPS: kein sinnvolles KGV.
        assert build_pe_history(history, eps) is None

    def test_handles_timezone_aware_index(self):
        history = make_history(days=750)
        history.frame.index = history.frame.index.tz_localize("UTC")

        assert build_pe_history(history, make_quarterly_eps()) is not None


class TestSectorMedian:
    def test_computes_median_per_sector(self):
        fundamentals = [
            Fundamentals(ticker=f"T{i}", sector="Technology", trailing_pe=pe)
            for i, pe in enumerate([10.0, 20.0, 30.0, 40.0])
        ]
        assert sector_median_pe(fundamentals)["Technology"] == 25.0

    def test_skips_sectors_with_too_few_peers(self):
        fundamentals = [
            Fundamentals(ticker="A", sector="Energy", trailing_pe=10.0),
            Fundamentals(ticker="B", sector="Energy", trailing_pe=12.0),
        ]
        assert "Energy" not in sector_median_pe(fundamentals, min_peers=4)

    def test_ignores_negative_and_absurd_pe(self):
        fundamentals = [
            Fundamentals(ticker="A", sector="Energy", trailing_pe=-5.0),
            Fundamentals(ticker="B", sector="Energy", trailing_pe=500.0),
            *[
                Fundamentals(ticker=f"C{i}", sector="Energy", trailing_pe=10.0)
                for i in range(4)
            ],
        ]
        assert sector_median_pe(fundamentals)["Energy"] == 10.0


class TestAnalyzeValuation:
    def test_cheap_beats_expensive_against_same_sector(self, flat_history):
        medians = {"Industrials": 20.0}
        cheap = Fundamentals(
            ticker="CHEAP", sector="Industrials", trailing_pe=10.0, forward_pe=9.0
        )
        pricey = Fundamentals(
            ticker="RICH", sector="Industrials", trailing_pe=32.0, forward_pe=34.0
        )

        cheap_result = analyze_valuation(cheap, flat_history, medians)
        pricey_result = analyze_valuation(pricey, flat_history, medians)

        assert cheap_result.score > pricey_result.score
        assert cheap_result.pe_vs_sector_median == 0.5

    def test_negative_pe_is_discarded_with_note(self, flat_history):
        f = Fundamentals(ticker="LOSS", sector="Industrials", trailing_pe=-8.0)

        result = analyze_valuation(f, flat_history, {"Industrials": 15.0})

        assert result.trailing_pe is None
        assert any("Negatives KGV" in n for n in result.notes)

    def test_rising_forward_pe_is_flagged_as_value_trap(self, flat_history):
        f = Fundamentals(
            ticker="TRAP", sector="Industrials", trailing_pe=6.0, forward_pe=13.0
        )

        result = analyze_valuation(f, flat_history, {"Industrials": 15.0})

        assert any("Value-Falle" in n for n in result.notes)

    def test_higher_bond_yield_lowers_the_score(self, flat_history):
        # Gleicher Titel, unterschiedliches Zinsumfeld: je höher die risikolose
        # Rendite, desto weniger attraktiv ist dieselbe Gewinnrendite.
        f = Fundamentals(ticker="THIN", sector="Industrials", trailing_pe=18.0)

        cheap_money = analyze_valuation(
            f, flat_history, {"Industrials": 20.0}, bond_yield=0.01
        )
        tight_money = analyze_valuation(
            f, flat_history, {"Industrials": 20.0}, bond_yield=0.06
        )

        assert tight_money.score < cheap_money.score
        assert tight_money.excess_yield_vs_bond is not None
        assert tight_money.excess_yield_vs_bond < 0
        assert any("keine Prämie" in n for n in tight_money.notes)

    def test_own_history_percentile_rewards_historically_cheap(self):
        # Kurs halbiert sich bei konstantem Gewinn -> KGV am unteren Rand.
        history = make_history(days=750, start=200.0, trend=-0.0009)
        eps = make_quarterly_eps(quarters=16, base=1.0)
        last_price = float(history.close.iloc[-1])
        f = Fundamentals(
            ticker="DROP",
            sector="Industrials",
            trailing_pe=last_price / 4.0,
            quarterly_eps=eps,
        )

        result = analyze_valuation(f, history, {})

        assert result.pe_percentile_own_history is not None
        assert result.pe_percentile_own_history < 20.0

    def test_returns_zero_score_when_nothing_is_computable(self, flat_history):
        f = Fundamentals(ticker="EMPTY", sector=None, trailing_pe=None)

        result = analyze_valuation(f, flat_history, {})

        assert result.score == 0.0
        assert any("nicht bewertbar" in n for n in result.notes)

    def test_peg_uses_growth_when_available(self, flat_history):
        f = Fundamentals(
            ticker="GROW",
            sector="Industrials",
            trailing_pe=20.0,
            forward_pe=18.0,
            earnings_growth=0.15,
        )

        result = analyze_valuation(f, flat_history, {"Industrials": 20.0})

        assert result.peg == 1.2  # 18 / 15
