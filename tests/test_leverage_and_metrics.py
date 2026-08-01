"""Tests für das Hebelmodul, die erweiterten Bewertungsmaße und die
neuen technischen Indikatoren."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from broker.analysis.leverage import (
    _norm_cdf,
    _norm_ppf,
    assess_leverage,
    barrier_for_probability,
    financing_cost,
    knockout_probability,
    max_sensible_leverage,
    volatility_drag,
)
from broker.analysis.technical import analyze_technical, atr, bollinger, stochastic
from broker.analysis.valuation import analyze_valuation, sector_medians
from broker.models import Fundamentals
from tests.conftest import make_history


class TestNormalDistribution:
    def test_cdf_is_symmetric(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5)
        assert _norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
        assert _norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)

    def test_ppf_inverts_cdf(self):
        for p in (0.01, 0.1, 0.5, 0.9, 0.99):
            assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)

    def test_ppf_rejects_invalid_input(self):
        with pytest.raises(ValueError):
            _norm_ppf(0.0)
        with pytest.raises(ValueError):
            _norm_ppf(1.0)


class TestVolatilityDrag:
    def test_no_drag_without_leverage(self):
        assert volatility_drag(1.0, 0.30, 252) == 0.0

    def test_drag_is_negative_and_grows_with_factor(self):
        two = volatility_drag(2.0, 0.30, 252)
        four = volatility_drag(4.0, 0.30, 252)

        assert two < 0 and four < 0
        assert four < two  # höherer Faktor, größerer Verlust

    def test_drag_grows_quadratically_with_volatility(self):
        low = volatility_drag(3.0, 0.20, 252)
        high = volatility_drag(3.0, 0.40, 252)

        # Doppelte Volatilität -> etwa vierfacher Log-Drag.
        assert math.log1p(high) == pytest.approx(4 * math.log1p(low), rel=1e-9)

    def test_matches_closed_form(self):
        # -0.5 * sigma^2 * k * (k-1) * T
        expected = math.expm1(-0.5 * 0.30**2 * 4 * 3 * 1.0)
        assert volatility_drag(4.0, 0.30, 252) == pytest.approx(expected)

    def test_realistic_case_is_substantial(self):
        """Der Kernbefund: bei 35 % Vola frisst Faktor 4 in einem Jahr ~50 %."""
        drag = volatility_drag(4.0, 0.35, 252)
        assert drag < -0.45

    def test_zero_volatility_means_no_drag(self):
        assert volatility_drag(4.0, 0.0, 252) == 0.0


class TestKnockoutProbability:
    def test_closer_barrier_is_riskier(self):
        near = knockout_probability(0.10, 0.35, 60)
        far = knockout_probability(0.30, 0.35, 60)
        assert near > far

    def test_longer_holding_is_riskier(self):
        short = knockout_probability(0.20, 0.35, 20)
        long = knockout_probability(0.20, 0.35, 250)
        assert long > short

    def test_higher_volatility_is_riskier(self):
        calm = knockout_probability(0.20, 0.20, 60)
        wild = knockout_probability(0.20, 0.60, 60)
        assert wild > calm

    def test_probability_stays_in_bounds(self):
        for distance in (0.01, 0.1, 0.5, 0.9):
            for vol in (0.1, 0.5, 1.5):
                p = knockout_probability(distance, vol, 250)
                assert p is not None and 0.0 <= p <= 1.0

    def test_touching_is_likelier_than_ending_below(self):
        """Der eigentliche Punkt: Berühren ist wahrscheinlicher als Schlusskurs.

        Unter dem Spiegelungsprinzip ist die Berührungswahrscheinlichkeit
        genau doppelt so hoch wie die Wahrscheinlichkeit, am Ende darunter
        zu liegen.
        """
        distance, vol, days = 0.20, 0.35, 60
        touch = knockout_probability(distance, vol, days)
        years = days / 252
        end_below = _norm_cdf(math.log(1 - distance) / (vol * math.sqrt(years)))

        assert touch == pytest.approx(2 * end_below, rel=1e-9)

    def test_invalid_inputs_return_none(self):
        assert knockout_probability(0.0, 0.3, 60) is None
        assert knockout_probability(1.5, 0.3, 60) is None
        assert knockout_probability(0.2, 0.0, 60) is None
        assert knockout_probability(0.2, 0.3, 0) is None


class TestBarrierForProbability:
    def test_inverts_knockout_probability(self):
        vol, days = 0.35, 60
        distance = barrier_for_probability(0.10, vol, days)
        assert distance is not None
        assert knockout_probability(distance, vol, days) == pytest.approx(0.10, abs=1e-6)

    def test_lower_target_needs_more_distance(self):
        strict = barrier_for_probability(0.05, 0.35, 60)
        loose = barrier_for_probability(0.25, 0.35, 60)
        assert strict > loose


class TestMaxLeverage:
    def test_more_tolerance_allows_more_leverage(self):
        cautious = max_sensible_leverage(0.35, 60, 0.15)
        bold = max_sensible_leverage(0.35, 60, 0.40)
        assert bold > cautious

    def test_higher_volatility_allows_less_leverage(self):
        calm = max_sensible_leverage(0.20, 60, 0.30)
        wild = max_sensible_leverage(0.60, 60, 0.30)
        assert wild < calm

    def test_longer_horizon_allows_less_leverage(self):
        short = max_sensible_leverage(0.35, 20, 0.30)
        long = max_sensible_leverage(0.35, 250, 0.30)
        assert long < short

    def test_invalid_inputs_return_none(self):
        assert max_sensible_leverage(0.0, 60, 0.3) is None
        assert max_sensible_leverage(0.3, 60, 0.0) is None
        assert max_sensible_leverage(0.3, 60, 1.5) is None


class TestFinancingCost:
    def test_no_cost_without_leverage(self):
        assert financing_cost(1.0, 0.06, 252) == 0.0

    def test_cost_scales_with_borrowed_part(self):
        assert financing_cost(3.0, 0.06, 252) == pytest.approx(2 * 0.06)

    def test_cost_scales_with_time(self):
        half = financing_cost(3.0, 0.06, 126)
        full = financing_cost(3.0, 0.06, 252)
        assert half == pytest.approx(full / 2)


class TestAssessment:
    def test_returns_none_without_volatility(self):
        assert assess_leverage("X", None) is None
        assert assess_leverage("X", 0.0) is None

    def test_produces_all_parts(self):
        result = assess_leverage("SAP.DE", 0.35, days=60, factor=3.0)

        assert result is not None
        assert result.drag < 0
        assert result.financing > 0
        assert set(result.knockout_risk) == {"10 %", "20 %", "30 %"}
        assert result.max_leverage is not None
        assert result.safe_barrier_10pct is not None

    def test_warns_when_requested_factor_exceeds_sensible(self):
        result = assess_leverage("WILD.DE", 0.70, days=120, factor=10.0)
        assert any("höchstens Hebel" in note for note in result.notes)

    def test_warns_about_high_volatility(self):
        result = assess_leverage("WILD.DE", 0.80, days=60, factor=2.0)
        assert any("Wette auf den Zeitpunkt" in note for note in result.notes)

    def test_calm_low_factor_needs_no_warning(self):
        result = assess_leverage("CALM.DE", 0.15, days=20, factor=2.0)
        assert result.notes == []


class TestNewIndicators:
    def test_atr_is_positive_for_real_ranges(self):
        history = make_history(days=200, trend=0.0, noise=0.02)
        value = atr(history.frame)
        assert value is not None and value > 0

    def test_atr_needs_high_low_columns(self):
        frame = pd.DataFrame({"Close": [1.0] * 50})
        from broker.models import PriceHistory

        assert atr(PriceHistory("X", frame).frame) is None

    def test_bollinger_percent_b_locates_the_price(self):
        rising = pd.Series([100 + i for i in range(40)])
        percent_b, bandwidth = bollinger(rising)

        assert percent_b is not None and bandwidth is not None
        assert percent_b > 0.8  # stetiger Anstieg -> oberes Band

    def test_bollinger_flat_series_has_no_band(self):
        assert bollinger(pd.Series([100.0] * 40)) == (None, None)

    def test_stochastic_is_high_at_the_top_of_the_range(self):
        history = make_history(days=100, trend=0.004)
        percent_k, percent_d = stochastic(history.frame)

        assert percent_k is not None and percent_k > 80

    def test_stochastic_is_low_at_the_bottom(self):
        history = make_history(days=100, trend=-0.004)
        percent_k, _ = stochastic(history.frame)
        assert percent_k is not None and percent_k < 20

    def test_indicators_reach_the_result(self):
        result = analyze_technical(make_history(days=400, trend=0.001, noise=0.01))

        assert result.atr14 is not None
        assert result.atr_percent is not None
        assert result.bollinger_percent_b is not None
        assert result.stochastic_k is not None


class TestBroadValuation:
    @staticmethod
    def peer(ticker: str, pe: float, ev: float, pb: float, fcf: float) -> Fundamentals:
        """Baut einen Titel mit vorgegebenen Kennzahlen."""
        market_cap = 1.0e9
        return Fundamentals(
            ticker=ticker, sector="Industrials", currency="EUR",
            market_cap=market_cap, trailing_pe=pe, price_to_book=pb,
            # EV/EBITDA über EBITDA steuern, Nettoverschuldung null halten.
            total_debt=0.0, total_cash=0.0, ebitda=market_cap / ev,
            free_cashflow=fcf * market_cap,
        )

    def universe(self) -> list[Fundamentals]:
        return [
            self.peer(f"P{i}", pe=20.0, ev=12.0, pb=2.0, fcf=0.05) for i in range(6)
        ]

    def test_sector_medians_cover_all_metrics(self):
        medians = sector_medians(self.universe())["Industrials"]

        assert medians["pe"] == pytest.approx(20.0)
        assert medians["ev_ebitda"] == pytest.approx(12.0)
        assert medians["pb"] == pytest.approx(2.0)
        assert medians["fcf_yield"] == pytest.approx(0.05)

    def test_broadly_cheap_beats_narrowly_cheap(self, flat_history):
        medians = sector_medians(self.universe())

        broad = self.peer("BROAD", pe=12.0, ev=7.0, pb=1.1, fcf=0.09)
        narrow = self.peer("NARROW", pe=12.0, ev=14.0, pb=2.6, fcf=0.02)

        broad_result = analyze_valuation(broad, flat_history, medians)
        narrow_result = analyze_valuation(narrow, flat_history, medians)

        assert broad_result.score > narrow_result.score
        assert broad_result.cheap_measures == 4
        assert narrow_result.cheap_measures == 1

    def test_breadth_is_reported(self, flat_history):
        medians = sector_medians(self.universe())
        broad = self.peer("BROAD", pe=12.0, ev=7.0, pb=1.1, fcf=0.09)

        result = analyze_valuation(broad, flat_history, medians)

        assert result.breadth == 1.0
        assert any("Bewertungsmaße" in note for note in result.notes)

    def test_narrow_cheapness_is_called_out(self, flat_history):
        medians = sector_medians(self.universe())
        narrow = self.peer("NARROW", pe=12.0, ev=14.0, pb=2.6, fcf=0.02)

        result = analyze_valuation(narrow, flat_history, medians)

        assert any("Einmaleffekte" in note for note in result.notes)

    def test_narrow_median_form_still_works(self, flat_history):
        """Rückwärtskompatibel: das schmale {Sektor: KGV} bleibt gültig."""
        f = Fundamentals(ticker="X", sector="Industrials", trailing_pe=10.0)
        result = analyze_valuation(f, flat_history, {"Industrials": 20.0})

        assert result.pe_vs_sector_median == 0.5

    def test_breadth_is_none_without_comparisons(self, flat_history):
        f = Fundamentals(ticker="X", sector=None, trailing_pe=10.0)
        assert analyze_valuation(f, flat_history, {}).breadth is None


class TestDerivedFundamentals:
    def test_enterprise_value_adds_net_debt(self):
        f = Fundamentals(ticker="X", market_cap=1.0e9, total_debt=3.0e8, total_cash=1.0e8)
        assert f.enterprise_value == pytest.approx(1.2e9)

    def test_net_cash_lowers_enterprise_value(self):
        f = Fundamentals(ticker="X", market_cap=1.0e9, total_debt=0.0, total_cash=2.0e8)
        assert f.enterprise_value == pytest.approx(8.0e8)

    def test_ev_to_ebitda(self):
        f = Fundamentals(
            ticker="X", market_cap=1.0e9, total_debt=0.0, total_cash=0.0, ebitda=1.0e8
        )
        assert f.ev_to_ebitda == pytest.approx(10.0)

    def test_ev_to_ebitda_is_none_for_negative_ebitda(self):
        f = Fundamentals(ticker="X", market_cap=1.0e9, ebitda=-5.0e7)
        assert f.ev_to_ebitda is None

    def test_fcf_yield(self):
        f = Fundamentals(ticker="X", market_cap=1.0e9, free_cashflow=7.0e7)
        assert f.fcf_yield == pytest.approx(0.07)

    def test_fcf_yield_can_be_negative(self):
        f = Fundamentals(ticker="X", market_cap=1.0e9, free_cashflow=-5.0e7)
        assert f.fcf_yield == pytest.approx(-0.05)

    def test_missing_inputs_yield_none(self):
        assert Fundamentals(ticker="X").enterprise_value is None
        assert Fundamentals(ticker="X").ev_to_ebitda is None
        assert Fundamentals(ticker="X").fcf_yield is None


class TestCurrencyNormalisation:
    """Die Größenschwelle muss für Dollar- und Euro-Titel gleich streng sein."""

    @staticmethod
    def screener():
        from broker.config import Config, Thresholds
        from broker.screener import Screener

        config = Config(thresholds=Thresholds(min_market_cap=3.0e8))
        return Screener(config, provider=None)  # type: ignore[arg-type]

    def test_euro_values_pass_through(self):
        f = Fundamentals(ticker="X.DE", currency="EUR", market_cap=5.0e8)
        assert self.screener()._market_cap_eur(f, 1.10) == pytest.approx(5.0e8)

    def test_dollar_values_are_converted(self):
        f = Fundamentals(ticker="X", currency="USD", market_cap=5.5e8)
        # DEXUSEU ist USD je EUR: 550 Mio. USD bei 1,10 sind 500 Mio. EUR.
        assert self.screener()._market_cap_eur(f, 1.10) == pytest.approx(5.0e8)

    def test_conversion_can_flip_the_filter_decision(self):
        screener = self.screener()
        # 320 Mio. USD liegen über der Rohschwelle, in Euro aber darunter.
        f = Fundamentals(ticker="X", currency="USD", market_cap=3.2e8, trailing_pe=12.0)

        assert screener._passes_hard_filters(f, eur_usd=None) is None
        assert screener._passes_hard_filters(f, eur_usd=1.20) is not None

    def test_unknown_currency_is_not_discarded(self):
        f = Fundamentals(ticker="X.SW", currency="CHF", market_cap=5.0e8)
        assert self.screener()._market_cap_eur(f, 1.10) == pytest.approx(5.0e8)

    def test_missing_rate_leaves_value_unchanged(self):
        f = Fundamentals(ticker="X", currency="USD", market_cap=5.5e8)
        assert self.screener()._market_cap_eur(f, None) == pytest.approx(5.5e8)
