from __future__ import annotations

from datetime import date

import pytest

from broker.macro.regime import bond_yield_for, build_regime, neutral_regime
from broker.macro.sensitivity import SECTOR_SENSITIVITY, sector_label
from broker.models import MacroSeries
from broker.universe import INDEX_GROUPS, load_universe


def series(key: str, label: str, value: float, change_3m: float | None = None):
    return MacroSeries(
        key=key, label=label, value=value, change_3m=change_3m, as_of=date(2026, 6, 30)
    )


class TestRegime:
    def test_neutral_regime_scores_all_sectors_at_fifty(self):
        regime = neutral_regime()
        assert regime.live is False
        assert set(regime.sector_scores) == set(SECTOR_SENSITIVITY)
        assert all(score == 50.0 for score in regime.sector_scores.values())

    def test_empty_series_falls_back_to_neutral(self):
        assert build_regime({}).live is False

    def test_rising_rates_hurt_real_estate_and_help_banks(self):
        regime = build_regime(
            {
                "fed_funds": series("fed_funds", "Leitzins", 5.0, change_3m=1.0),
                "yield_curve": series("yield_curve", "Kurve", 1.2),
            }
        )
        assert regime.rate_direction == "steigend"
        assert regime.sector_scores["Real Estate"] < 50.0
        assert regime.sector_scores["Financial Services"] > 50.0

    def test_inverted_curve_is_detected(self):
        regime = build_regime({"yield_curve": series("yield_curve", "Kurve", -0.4)})
        assert regime.curve_shape == "invers"

    def test_steep_curve_is_detected(self):
        regime = build_regime({"yield_curve": series("yield_curve", "Kurve", 1.5)})
        assert regime.curve_shape == "steil"

    def test_high_vix_means_risk_off_and_favours_defensives(self):
        regime = build_regime({"vix": series("vix", "VIX", 34.0)})
        assert regime.risk_appetite == "risk-off"
        assert regime.sector_scores["Consumer Defensive"] > 50.0
        assert regime.sector_scores["Consumer Cyclical"] < 50.0

    def test_rising_oil_helps_energy(self):
        regime = build_regime(
            {"oil_brent": series("oil_brent", "Brent", 95.0, change_3m=0.30)}
        )
        assert regime.sector_scores["Energy"] > 50.0

    def test_scores_stay_in_bounds_under_extreme_input(self):
        regime = build_regime(
            {
                "fed_funds": series("fed_funds", "Leitzins", 20.0, change_3m=10.0),
                "yield_curve": series("yield_curve", "Kurve", -5.0),
                "us_cpi": series("us_cpi", "CPI", 400.0, change_3m=0.5),
                "us_unemployment": series("us_unemployment", "Quote", 15.0, change_3m=5.0),
                "oil_brent": series("oil_brent", "Brent", 200.0, change_3m=2.0),
                "vix": series("vix", "VIX", 80.0),
            }
        )
        assert all(0.0 <= s <= 100.0 for s in regime.sector_scores.values())

    def test_score_for_unknown_sector_is_neutral(self):
        regime = build_regime({"vix": series("vix", "VIX", 20.0)})
        assert regime.score_for("Nicht Existent") == 50.0
        assert regime.score_for(None) == 50.0

    def test_summary_mentions_all_dimensions(self):
        regime = build_regime({"fed_funds": series("fed_funds", "Leitzins", 4.0, 0.5)})
        for token in ("Zinsen", "Renditekurve", "Inflation", "Wachstum", "Risiko"):
            assert token in regime.summary


class TestBondYield:
    def test_picks_euro_yield_for_european_titles(self):
        regime = build_regime(
            {
                "us_10y": series("us_10y", "US 10J", 4.5),
                "ez_10y": series("ez_10y", "EZ 10J", 2.6),
            }
        )
        assert bond_yield_for(regime, "DE") == pytest.approx(0.026)
        assert bond_yield_for(regime, "US") == pytest.approx(0.045)

    def test_returns_none_without_data(self):
        assert bond_yield_for(neutral_regime(), "DE") is None


class TestUniverse:
    def test_known_groups_are_not_empty(self):
        for group in INDEX_GROUPS:
            assert load_universe(group), f"Gruppe {group} ist leer"

    def test_all_contains_both_regions(self):
        regions = {e.region for e in load_universe("all")}
        assert "US" in regions
        assert "DE" in regions

    def test_germany_excludes_us_titles(self):
        assert all(e.index != "SP500" for e in load_universe("germany"))

    def test_tickers_are_unique(self):
        entries = load_universe("all")
        assert len({e.ticker for e in entries}) == len(entries)

    def test_explicit_index_list_is_accepted(self):
        entries = load_universe("DAX,SP500")
        indices = {e.index for e in entries}
        assert indices == {"DAX", "SP500"}

    def test_extra_tickers_are_appended_and_deduplicated(self):
        entries = load_universe("dax", extra_tickers=["ROCK.CO", "SAP.DE"])
        tickers = [e.ticker for e in entries]
        assert "ROCK.CO" in tickers
        assert tickers.count("SAP.DE") == 1

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError):
            load_universe("")


class TestSectorLabels:
    def test_every_sensitivity_sector_has_a_german_label(self):
        for sector in SECTOR_SENSITIVITY:
            assert sector_label(sector) != sector or sector == "Unbekannt"

    def test_unknown_sector_passes_through(self):
        assert sector_label("Something Else") == "Something Else"
        assert sector_label(None) == "Unbekannt"
