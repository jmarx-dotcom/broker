"""Tests für die Abschluss-Kennzahlen und die europäischen Makroquellen.

Beide Bausteine wurden gegen synthetische Daten entwickelt, weil in der
Entwicklungsumgebung kein Netzzugang zu Yahoo, Eurostat oder EZB besteht.
Die Tests bilden deshalb die Antwortformate nach, statt sie abzurufen.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from broker.analysis.quality import analyze_quality
from broker.macro.europe import (
    ECB_SERIES,
    EUROSTAT_SERIES,
    EcbClient,
    EcbSpec,
    EurostatClient,
    EurostatSpec,
    _parse_period,
    _to_series,
    fetch_european_series,
)
from broker.macro.regime import bond_yield_for, build_regime
from broker.models import Fundamentals, MacroRegime, MacroSeries
from tests.conftest import make_history


# --- Abschluss-Kennzahlen ------------------------------------------------


def quarters(values: list[float], end: str = "2026-06-30") -> pd.Series:
    index = pd.date_range(end=pd.Timestamp(end), periods=len(values), freq="QE")
    return pd.Series(values, index=index)


def test_roic_uses_invested_capital_not_equity():
    """Zwei Firmen mit gleichem EBIT, aber unterschiedlicher Finanzierung.

    Die verschuldete hebt ihren ROE, der ROIC bleibt gleich — genau dafür
    ist die Kennzahl da.
    """
    unlevered = Fundamentals(
        ticker="A", ebit=200.0, tax_rate=0.25, total_equity=1000.0,
        total_debt=0.0, total_cash=0.0,
    )
    levered = Fundamentals(
        ticker="B", ebit=200.0, tax_rate=0.25, total_equity=400.0,
        total_debt=600.0, total_cash=0.0,
    )
    assert unlevered.roic == pytest.approx(0.15)
    assert levered.roic == pytest.approx(0.15)


def test_roic_subtracts_cash_and_defaults_tax_rate():
    f = Fundamentals(
        ticker="A", ebit=100.0, total_equity=500.0, total_debt=300.0, total_cash=300.0,
    )
    # Ohne tax_rate greift der Standardsatz von 25%: NOPAT 75 auf 500 investiert.
    assert f.roic == pytest.approx(0.15)


def test_roic_none_when_invested_capital_not_positive():
    f = Fundamentals(
        ticker="A", ebit=100.0, total_equity=100.0, total_debt=0.0, total_cash=500.0,
    )
    assert f.roic is None


def test_roic_none_without_statements():
    assert Fundamentals(ticker="A").roic is None
    assert Fundamentals(ticker="A", ebit=100.0).roic is None


def test_roic_caps_absurd_tax_rate():
    """Yahoo liefert bei Sonderposten gelegentlich Steuerquoten über 100%."""
    f = Fundamentals(
        ticker="A", ebit=100.0, tax_rate=5.0, total_equity=100.0,
    )
    # Auf 60% gedeckelt: NOPAT 40 auf 100 investiert.
    assert f.roic == pytest.approx(0.40)


def test_interest_coverage_handles_sign_conventions():
    """Yahoo führt den Zinsaufwand je nach Zeile mit oder ohne Vorzeichen."""
    positive = Fundamentals(ticker="A", ebit=500.0, interest_expense=100.0)
    negative = Fundamentals(ticker="A", ebit=500.0, interest_expense=-100.0)
    assert positive.interest_coverage == pytest.approx(5.0)
    assert negative.interest_coverage == pytest.approx(5.0)


def test_interest_coverage_none_without_debt_service():
    assert Fundamentals(ticker="A", ebit=500.0, interest_expense=0.0).interest_coverage is None
    assert Fundamentals(ticker="A", ebit=500.0).interest_coverage is None


def test_current_ratio():
    f = Fundamentals(ticker="A", current_assets=1500.0, current_liabilities=1000.0)
    assert f.current_ratio == pytest.approx(1.5)
    assert Fundamentals(ticker="A", current_assets=1500.0).current_ratio is None
    assert Fundamentals(
        ticker="A", current_assets=1500.0, current_liabilities=0.0
    ).current_ratio is None


def test_margin_trend_detects_falling_margins():
    revenue = quarters([1000.0] * 8)
    income = quarters([120.0, 120.0, 120.0, 120.0, 90.0, 90.0, 90.0, 90.0])
    f = Fundamentals(ticker="A", quarterly_revenue=revenue, quarterly_net_income=income)
    # Ältere Hälfte 12%, neuere 9% -> -3 Prozentpunkte.
    assert f.margin_trend == pytest.approx(-0.03)


def test_margin_trend_detects_rising_margins():
    revenue = quarters([1000.0] * 8)
    income = quarters([80.0] * 4 + [110.0] * 4)
    f = Fundamentals(ticker="A", quarterly_revenue=revenue, quarterly_net_income=income)
    assert f.margin_trend == pytest.approx(0.03)


def test_margin_trend_needs_enough_quarters():
    revenue = quarters([1000.0] * 4)
    income = quarters([100.0] * 4)
    f = Fundamentals(ticker="A", quarterly_revenue=revenue, quarterly_net_income=income)
    assert f.margin_trend is None


def test_margin_trend_ignores_zero_revenue_quarters():
    revenue = quarters([1000.0, 0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    income = quarters([100.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    f = Fundamentals(ticker="A", quarterly_revenue=revenue, quarterly_net_income=income)
    # Ein Quartal fällt weg, es bleiben sechs — genug, und keine Division durch null.
    assert f.margin_trend == pytest.approx(0.0)


def test_share_dilution_annualizes():
    history = pd.Series(
        [100.0, 121.0],
        index=pd.to_datetime(["2024-06-30", "2026-06-30"]),
    )
    f = Fundamentals(ticker="A", shares_history=history)
    # +21% über zwei Jahre -> rund 10% pro Jahr.
    assert f.share_dilution == pytest.approx(0.10, abs=0.005)


def test_share_dilution_negative_for_buybacks():
    history = pd.Series(
        [100.0, 90.0],
        index=pd.to_datetime(["2024-06-30", "2026-06-30"]),
    )
    assert Fundamentals(ticker="A", shares_history=history).share_dilution < 0


def test_share_dilution_needs_a_meaningful_span():
    history = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2026-05-30", "2026-06-30"]),
    )
    # Ein Monat hochgerechnet ergäbe absurde Jahresraten.
    assert Fundamentals(ticker="A", shares_history=history).share_dilution is None
    assert Fundamentals(ticker="A").share_dilution is None


# --- Qualitätsscore mit den neuen Kennzahlen ------------------------------


def test_quality_flags_thin_interest_coverage():
    f = Fundamentals(
        ticker="A", ebitda=1.0e8, total_debt=2.0e8, total_cash=1.0e7,
        return_on_equity=0.10, profit_margin=0.05,
        ebit=1.0e8, interest_expense=8.0e7,  # Deckung 1,25
    )
    result = analyze_quality(f)
    assert any("Zinsdeckung" in flag for flag in result.red_flags)
    assert result.interest_coverage == pytest.approx(1.2, abs=0.1)


def test_quality_notes_comfortable_coverage():
    f = Fundamentals(
        ticker="A", return_on_equity=0.10, profit_margin=0.05,
        ebit=1.0e8, interest_expense=2.0e6,  # Deckung 50
    )
    result = analyze_quality(f)
    assert any("Zinslast" in note for note in result.notes)


def test_quality_flags_illiquidity_and_dilution():
    f = Fundamentals(
        ticker="A", return_on_equity=0.10, profit_margin=0.05,
        current_assets=8.0e7, current_liabilities=1.0e8,
        shares_history=pd.Series(
            [100.0, 120.0], index=pd.to_datetime(["2024-06-30", "2026-06-30"])
        ),
    )
    result = analyze_quality(f)
    assert any("Liquiditätsgrad" in flag for flag in result.red_flags)
    assert any("Aktienzahl wächst" in flag for flag in result.red_flags)
    assert result.current_ratio == pytest.approx(0.8)


def test_quality_notes_buybacks():
    f = Fundamentals(
        ticker="A", return_on_equity=0.10, profit_margin=0.05,
        shares_history=pd.Series(
            [100.0, 85.0], index=pd.to_datetime(["2024-06-30", "2026-06-30"])
        ),
    )
    result = analyze_quality(f)
    assert any("Aktienrückkäufe" in note for note in result.notes)


def test_quality_flags_falling_margin_trend():
    f = Fundamentals(
        ticker="A", return_on_equity=0.10, profit_margin=0.05,
        quarterly_revenue=quarters([1000.0] * 8),
        quarterly_net_income=quarters([120.0] * 4 + [70.0] * 4),
    )
    result = analyze_quality(f)
    assert any("Nettomarge fällt" in flag for flag in result.red_flags)
    assert result.margin_trend is not None and result.margin_trend < 0


def test_quality_flags_weak_roic():
    f = Fundamentals(
        ticker="A", return_on_equity=0.10, profit_margin=0.05,
        ebit=1.0e7, tax_rate=0.25, total_equity=1.0e9,
    )
    result = analyze_quality(f)
    assert any("Kapitalrendite von nur" in flag for flag in result.red_flags)


def test_quality_rewards_strong_roic():
    """Gleiche Ausgangslage, nur die Kapitalrendite unterscheidet sich."""
    base = dict(return_on_equity=0.10, profit_margin=0.05, total_equity=1.0e9)
    weak = analyze_quality(Fundamentals(ticker="A", ebit=2.0e7, **base))
    strong = analyze_quality(Fundamentals(ticker="B", ebit=3.0e8, **base))
    assert strong.score > weak.score


def test_quality_unchanged_without_statement_data(solid_fundamentals):
    """Fehlende Abschlüsse dürfen den Score nicht drücken."""
    result = analyze_quality(solid_fundamentals)
    assert result.roic is None
    assert result.interest_coverage is None
    assert result.margin_trend is None
    assert result.score > 60


# --- Provider: Abschlüsse einlesen ---------------------------------------


class FakeTicker:
    """Minimaler Ersatz für yfinance.Ticker mit Abschlüssen als DataFrame."""

    def __init__(self, income=None, balance=None, quarterly_income=None) -> None:
        empty = pd.DataFrame()
        self.income_stmt = empty if income is None else income
        self.balance_sheet = empty if balance is None else balance
        self.quarterly_income_stmt = empty if quarterly_income is None else quarterly_income


def statement_frame(rows: dict[str, list[float]], periods: list[str]) -> pd.DataFrame:
    """Yahoo liefert Abschlüsse mit Kennzahlen als Zeilen, Perioden als Spalten —
    und zwar in absteigender Reihenfolge, neueste Periode zuerst."""
    return pd.DataFrame(rows, index=pd.to_datetime(periods)).T


def test_provider_reads_statements(monkeypatch):
    from broker.providers.yahoo import YahooProvider

    income = statement_frame(
        {
            "EBIT": [500.0, 450.0],
            "Interest Expense": [50.0, 55.0],
            "Pretax Income": [400.0, 360.0],
            "Tax Provision": [100.0, 90.0],
        },
        ["2025-12-31", "2024-12-31"],
    )
    balance = statement_frame(
        {
            "Current Assets": [1200.0, 1100.0],
            "Current Liabilities": [800.0, 900.0],
            "Stockholders Equity": [2000.0, 1900.0],
            "Ordinary Shares Number": [95.0, 100.0],
        },
        ["2025-12-31", "2023-12-31"],
    )
    quarterly = statement_frame(
        {"Total Revenue": [1000.0, 990.0], "Net Income": [100.0, 95.0]},
        ["2026-03-31", "2025-12-31"],
    )

    provider = YahooProvider()
    monkeypatch.setattr(
        YahooProvider, "_ticker",
        lambda self, ticker: FakeTicker(income, balance, quarterly),
    )
    data = provider._statements("TEST")

    assert data["ebit"] == pytest.approx(500.0)  # neueste Spalte, nicht die älteste
    assert data["interest_expense"] == pytest.approx(50.0)
    assert data["tax_rate"] == pytest.approx(0.25)
    assert data["current_assets"] == pytest.approx(1200.0)
    assert data["current_liabilities"] == pytest.approx(800.0)
    assert data["total_equity"] == pytest.approx(2000.0)
    # Zeitreihen kommen aufsteigend sortiert zurück.
    assert list(data["shares_history"]) == [100.0, 95.0]
    assert list(data["quarterly_revenue"]) == [990.0, 1000.0]


def test_provider_falls_back_to_alternative_row_labels(monkeypatch):
    from broker.providers.yahoo import YahooProvider

    income = statement_frame({"Operating Income": [300.0]}, ["2025-12-31"])
    balance = statement_frame(
        {"Total Current Assets": [500.0], "Common Stock Equity": [900.0]},
        ["2025-12-31"],
    )

    provider = YahooProvider()
    monkeypatch.setattr(
        YahooProvider, "_ticker", lambda self, ticker: FakeTicker(income, balance)
    )
    data = provider._statements("TEST")

    assert data["ebit"] == pytest.approx(300.0)
    assert data["current_assets"] == pytest.approx(500.0)
    assert data["total_equity"] == pytest.approx(900.0)
    assert data["current_liabilities"] is None
    assert data["tax_rate"] is None


def test_provider_returns_empty_dict_when_statements_unavailable(monkeypatch):
    from broker.providers.yahoo import YahooProvider

    def boom(self, ticker):
        raise RuntimeError("Yahoo antwortet nicht")

    provider = YahooProvider()
    monkeypatch.setattr(YahooProvider, "_ticker", boom)
    assert provider._statements("TEST") == {}
    # Und Fundamentals lassen sich damit weiterhin bauen.
    assert Fundamentals(ticker="TEST", **provider._statements("TEST")).roic is None


def test_provider_handles_empty_frames(monkeypatch):
    from broker.providers.yahoo import YahooProvider

    provider = YahooProvider()
    monkeypatch.setattr(YahooProvider, "_ticker", lambda self, ticker: FakeTicker())
    data = provider._statements("TEST")
    assert set(data) == {
        "ebit", "interest_expense", "tax_rate", "current_assets",
        "current_liabilities", "total_equity", "quarterly_revenue",
        "quarterly_net_income", "shares_history",
    }
    assert all(v is None for v in data.values())


# --- Aktienzahl bei mehreren Gattungen -----------------------------------


def test_share_equivalent_uses_company_wide_count_for_preference_shares():
    """VW: 206,2 Mio. Vorzüge, aber 37,43 Mrd. Gesamtwert bei 74,70 EUR."""
    from broker.providers.yahoo import share_equivalent

    result = share_equivalent(2.062e8, 3.743e10, 74.70)
    assert result == pytest.approx(3.743e10 / 74.70)
    assert result / 2.062e8 == pytest.approx(2.43, abs=0.02)


def test_share_equivalent_keeps_reported_count_when_consistent():
    from broker.providers.yahoo import share_equivalent

    # Eine Gattung: Kurs mal Aktienzahl ergibt die Marktkapitalisierung.
    assert share_equivalent(2.0e8, 1.0e10, 50.0) == pytest.approx(2.0e8)


def test_share_equivalent_tolerates_small_gaps():
    from broker.providers.yahoo import share_equivalent

    # 10% Zeitversatz zwischen Kurs und Marktkapitalisierung: unverändert.
    assert share_equivalent(2.0e8, 1.1e10, 50.0) == pytest.approx(2.0e8)


def test_share_equivalent_ignores_smaller_implied_count():
    """Ist die errechnete Zahl kleiner, liegt kein Gattungsproblem vor."""
    from broker.providers.yahoo import share_equivalent

    assert share_equivalent(2.0e8, 5.0e9, 50.0) == pytest.approx(2.0e8)


def test_share_equivalent_falls_back_without_market_cap():
    from broker.providers.yahoo import share_equivalent

    assert share_equivalent(2.0e8, None, 50.0) == pytest.approx(2.0e8)
    assert share_equivalent(2.0e8, 1.0e10, None) == pytest.approx(2.0e8)
    assert share_equivalent(2.0e8, 1.0e10, 0.0) == pytest.approx(2.0e8)
    assert share_equivalent(None, None, None) is None


def test_preference_share_pe_history_matches_quoted_price():
    """Der eigentliche Schaden: die KGV-Reihe der Gattung.

    Mit der Gattungs-Aktienzahl fällt das historische KGV um den Faktor 2,43
    zu niedrig aus — das aktuelle KGV läge dann über dem gesamten Verlauf und
    der Titel bekäme null Punkte, obwohl er günstig ist.
    """
    from broker.analysis.valuation import build_pe_history
    from broker.providers.yahoo import share_equivalent

    net_income = quarters([5.0e8] * 12, end="2026-06-30")
    history = make_history(days=750, start=74.70, trend=0.0)

    gattung = build_pe_history(history, net_income / 2.062e8)
    gesamt = build_pe_history(
        history, net_income / share_equivalent(2.062e8, 3.743e10, 74.70)
    )
    assert gattung is not None and gesamt is not None
    assert float(gesamt.median()) / float(gattung.median()) == pytest.approx(
        2.43, abs=0.02
    )


# --- Perioden- und Reihenaufbereitung ------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026-07-15", date(2026, 7, 15)),
        ("2026-07", date(2026, 7, 1)),
        ("2026", date(2026, 1, 1)),
        ("2026Q2", date(2026, 6, 1)),
        ("2026-Q4", date(2026, 12, 1)),
        (" 2026-07 ", date(2026, 7, 1)),
    ],
)
def test_parse_period(text, expected):
    assert _parse_period(text) == expected


@pytest.mark.parametrize("text", ["", "Sommer", "2026-XX", "20Q6QQ"])
def test_parse_period_rejects_nonsense(text):
    assert _parse_period(text) is None


def test_to_series_percent_uses_differences():
    """Bei Prozentreihen ist die Differenz die richtige Veränderung, nicht die Rate."""
    observations = [
        (date(2025, 7, 1), 6.5),
        (date(2026, 4, 1), 6.0),
        (date(2026, 7, 1), 6.2),
    ]
    series = _to_series("ez_unemployment", "Quote", "%", observations, percent=True)
    assert series is not None
    assert series.value == pytest.approx(6.2)
    assert series.as_of == date(2026, 7, 1)
    assert series.change_3m == pytest.approx(0.2)
    assert series.change_12m == pytest.approx(-0.3)


def test_to_series_index_uses_relative_change():
    observations = [
        (date(2026, 4, 1), 100.0),
        (date(2026, 7, 1), 102.0),
    ]
    series = _to_series("ez_industrial_production", "Index", "Index", observations, percent=False)
    assert series is not None
    assert series.change_3m == pytest.approx(0.02)
    assert series.change_12m is None  # kein Wert ein Jahr zurück


def test_to_series_sorts_unordered_observations():
    observations = [
        (date(2026, 7, 1), 3.0),
        (date(2026, 1, 1), 1.0),
        (date(2026, 4, 1), 2.0),
    ]
    series = _to_series("k", "L", "%", observations, percent=True)
    assert series is not None and series.value == pytest.approx(3.0)


def test_to_series_empty():
    assert _to_series("k", "L", "%", [], percent=True) is None


# --- Eurostat- und EZB-Clients -------------------------------------------


class FakeResponse:
    def __init__(self, payload: object = None, text: str = "", status: int = 200) -> None:
        self._payload = payload
        self.text = text if text else json.dumps(payload)
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self) -> object:
        return self._payload


EUROSTAT_PAYLOAD = {
    "value": {"0": 100.0, "1": 101.5, "2": 103.0},
    "dimension": {
        "time": {"category": {"index": {"2026-04": 0, "2026-05": 1, "2026-06": 2}}}
    },
}

SPEC = EurostatSpec("test_key", "sts_inpr_m", "Testreihe", "Index", {"geo": "EA20"})
ECB_SPEC = EcbSpec("test_rate", "FM", "D.U2.EUR.TEST", "Testzins", "%")


def test_eurostat_client_maps_values_to_periods(monkeypatch):
    captured: dict = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse(EUROSTAT_PAYLOAD)

    monkeypatch.setattr("broker.macro.europe.requests.get", fake_get)
    series = EurostatClient().fetch(SPEC)

    assert series is not None
    assert series.value == pytest.approx(103.0)
    assert series.as_of == date(2026, 6, 1)
    assert "sts_inpr_m" in captured["url"]
    assert captured["params"]["geo"] == "EA20"
    assert captured["params"]["format"] == "JSON"


def test_eurostat_refuses_to_blend_several_series(monkeypatch):
    """Zwei Länder in einer Antwort ergeben keine Zeitreihe, sondern ein Gemisch.

    Die alte Auswertung hätte hier stillschweigend die erste Hälfte der Werte
    genommen und den Rest verworfen — ein Ergebnis, das nach einer sauberen
    Reihe aussieht und keines ist.
    """
    payload = {
        # geo (2) × time (3), zeilenweise: EA20 zuerst, dann DE.
        "value": {"0": 6.0, "1": 6.1, "2": 6.2, "3": 3.0, "4": 3.1, "5": 3.2},
        "id": ["geo", "time"],
        "size": [2, 3],
        "dimension": {
            "geo": {"category": {"index": {"EA20": 0, "DE": 1}}},
            "time": {"category": {"index": {"2026-04": 0, "2026-05": 1, "2026-06": 2}}},
        },
    }
    monkeypatch.setattr(
        "broker.macro.europe.requests.get", lambda *a, **k: FakeResponse(payload)
    )
    assert EurostatClient().fetch(SPEC) is None


def test_eurostat_guard_is_independent_of_axis_order(monkeypatch):
    """Auch wenn die Zeitachse zuerst steht, bleibt die Antwort mehrdeutig."""
    from broker.macro.europe import _time_lookup

    payload = {
        "value": {"0": 6.0, "1": 5.0, "2": 6.1, "3": 5.1},
        "id": ["time", "sex"],
        "size": [2, 2],
        "dimension": {
            "time": {"category": {"index": {"2026-05": 0, "2026-06": 1}}},
            "sex": {"category": {"index": {"M": 0, "F": 1}}},
        },
    }
    assert _time_lookup(payload) is None


def test_eurostat_single_valued_axes_are_fine():
    """Sind alle übrigen Achsen auf einen Wert eingegrenzt, passt die Zuordnung."""
    from broker.macro.europe import _time_lookup

    payload = {
        "value": {"0": 6.0, "1": 6.1, "2": 6.2},
        "id": ["geo", "sex", "time"],
        "size": [1, 1, 3],
        "dimension": {
            "time": {"category": {"index": {"2026-04": 0, "2026-05": 1, "2026-06": 2}}},
        },
    }
    lookup = _time_lookup(payload)
    assert lookup is not None
    assert [lookup(i) for i in range(3)] == ["2026-04", "2026-05", "2026-06"]


def test_time_lookup_without_axis_description_falls_back():
    """Ohne id/size bleibt nur die Annahme, dass allein die Zeit variiert."""
    from broker.macro.europe import _time_lookup

    lookup = _time_lookup(EUROSTAT_PAYLOAD)
    assert lookup is not None
    assert lookup(0) == "2026-04"
    assert lookup(2) == "2026-06"
    assert lookup(9) is None


def test_time_lookup_without_time_axis():
    from broker.macro.europe import _time_lookup

    assert _time_lookup({"value": {"0": 1.0}}) is None


def test_eurostat_falls_back_to_alternative_filters(monkeypatch):
    """Ein unbekannter Code liefert bei Eurostat 200 mit leerer Antwort."""
    spec = EurostatSpec(
        "ez_unemployment", "une_rt_m", "Quote", "%",
        {"geo": "EA20", "age": "TOTAL"},
        percent=True,
        fallbacks=({"geo": "EA19", "age": "TOTAL"},),
    )
    versuche: list[str] = []

    def fake_get(url, params=None, timeout=None):
        versuche.append(params["geo"])
        if params["geo"] == "EA20":
            return FakeResponse({"value": {}, "dimension": {}})
        return FakeResponse(EUROSTAT_PAYLOAD)

    monkeypatch.setattr("broker.macro.europe.requests.get", fake_get)
    series = EurostatClient().fetch(spec)

    assert versuche == ["EA20", "EA19"]
    assert series is not None and series.value == pytest.approx(103.0)


def test_eurostat_stops_at_first_working_filter(monkeypatch):
    spec = EurostatSpec(
        "k", "ds", "L", "Index", {"geo": "EA20"},
        fallbacks=({"geo": "EA19"},),
    )
    versuche: list[str] = []

    def fake_get(url, params=None, timeout=None):
        versuche.append(params["geo"])
        return FakeResponse(EUROSTAT_PAYLOAD)

    monkeypatch.setattr("broker.macro.europe.requests.get", fake_get)
    assert EurostatClient().fetch(spec) is not None
    assert versuche == ["EA20"]  # der Ersatzfilter bleibt ungenutzt


def test_eurostat_returns_none_when_every_filter_fails(monkeypatch):
    spec = EurostatSpec(
        "k", "ds", "L", "Index", {"geo": "EA20"}, fallbacks=({"geo": "EA19"},)
    )
    monkeypatch.setattr(
        "broker.macro.europe.requests.get",
        lambda *a, **k: FakeResponse({"value": {}, "dimension": {}}),
    )
    assert EurostatClient().fetch(spec) is None


def test_eurostat_reports_valid_codes_after_failing(monkeypatch, caplog):
    """Statt 'prüfe die Codeliste' soll die Codeliste selbst im Log stehen."""
    spec = EurostatSpec("k", "une_rt_m", "L", "%", {"geo": "EA20", "age": "TOTAL"})

    def fake_get(url, params=None, timeout=None):
        if params.get("lastTimePeriod"):
            return FakeResponse(
                {
                    "dimension": {
                        "geo": {"category": {"index": {"EA19": 0, "EA20": 1, "DE": 2}}},
                        "age": {"category": {"index": {"Y_GE15": 0, "Y_LT25": 1}}},
                    }
                }
            )
        return FakeResponse({"value": {}, "dimension": {}})

    monkeypatch.setattr("broker.macro.europe.requests.get", fake_get)
    with caplog.at_level("WARNING"):
        assert EurostatClient().fetch(spec) is None

    meldung = caplog.text
    assert "Y_GE15" in meldung and "Y_LT25" in meldung
    assert "geo = DE, EA19, EA20" in meldung


def test_eurostat_survives_unavailable_code_list(monkeypatch, caplog):
    spec = EurostatSpec("k", "ds", "L", "%", {"geo": "EA20"})

    def fake_get(url, params=None, timeout=None):
        if params.get("lastTimePeriod"):
            raise OSError("Codeliste gerade nicht erreichbar")
        return FakeResponse({"value": {}, "dimension": {}})

    monkeypatch.setattr("broker.macro.europe.requests.get", fake_get)
    with caplog.at_level("WARNING"):
        assert EurostatClient().fetch(spec) is None
    assert "Codeliste war nicht abrufbar" in caplog.text


def test_code_list_is_only_fetched_on_failure(monkeypatch):
    """Der erklärende Zusatzabruf darf den Normalfall nicht verteuern."""
    spec = EurostatSpec("k", "ds", "L", "Index", {"geo": "EA20"})
    abrufe: list[dict] = []

    def fake_get(url, params=None, timeout=None):
        abrufe.append(params)
        return FakeResponse(EUROSTAT_PAYLOAD)

    monkeypatch.setattr("broker.macro.europe.requests.get", fake_get)
    assert EurostatClient().fetch(spec) is not None
    assert len(abrufe) == 1
    assert "lastTimePeriod" not in abrufe[0]


def test_unemployment_spec_has_fallbacks():
    """Die Reihe, die im Live-Lauf ausfiel, darf nicht ohne Ersatz bleiben."""
    spec = next(s for s in EUROSTAT_SERIES if s.key == "ez_unemployment")
    assert spec.fallbacks
    assert len(spec.filter_sets) == len(spec.fallbacks) + 1
    # Jeder Ersatzsatz belegt dieselben Achsen — ein fehlender Filter machte
    # die Antwort mehrdimensional.
    assert all(set(f) == set(spec.filters) for f in spec.fallbacks)


def test_eurostat_client_handles_gaps_and_bad_indices(monkeypatch):
    payload = {
        "value": {"0": 100.0, "1": None, "zwei": 99.0, "2": 103.0},
        "dimension": {
            "time": {"category": {"index": {"2026-04": 0, "2026-05": 1, "2026-06": 2}}}
        },
    }
    monkeypatch.setattr(
        "broker.macro.europe.requests.get",
        lambda *a, **k: FakeResponse(payload),
    )
    series = EurostatClient().fetch(SPEC)
    assert series is not None and series.value == pytest.approx(103.0)


def test_eurostat_client_returns_none_on_empty_payload(monkeypatch):
    monkeypatch.setattr(
        "broker.macro.europe.requests.get",
        lambda *a, **k: FakeResponse({"value": {}, "dimension": {}}),
    )
    assert EurostatClient().fetch(SPEC) is None


def test_eurostat_client_survives_http_error(monkeypatch):
    monkeypatch.setattr(
        "broker.macro.europe.requests.get",
        lambda *a, **k: FakeResponse({}, status=404),
    )
    assert EurostatClient().fetch(SPEC) is None


def test_eurostat_client_survives_connection_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("kein Netz")

    monkeypatch.setattr("broker.macro.europe.requests.get", boom)
    assert EurostatClient().fetch(SPEC) is None


ECB_CSV = """KEY,FREQ,TIME_PERIOD,OBS_VALUE
FM.D.U2.EUR.TEST,D,2026-03-01,4.00
FM.D.U2.EUR.TEST,D,2026-04-01,3.75
FM.D.U2.EUR.TEST,D,2026-05-01,3.50
FM.D.U2.EUR.TEST,D,2026-06-01,3.25
"""


def test_ecb_client_parses_csv(monkeypatch):
    captured: dict = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse(text=ECB_CSV)

    monkeypatch.setattr("broker.macro.europe.requests.get", fake_get)
    series = EcbClient().fetch(ECB_SPEC)

    assert series is not None
    assert series.value == pytest.approx(3.25)
    assert series.as_of == date(2026, 6, 1)
    assert series.change_3m == pytest.approx(-0.75)  # Prozentpunkte
    assert "FM/D.U2.EUR.TEST" in captured["url"]
    assert captured["params"]["format"] == "csvdata"


def test_ecb_client_skips_unparsable_rows(monkeypatch):
    csv_text = (
        "KEY,TIME_PERIOD,OBS_VALUE\n"
        "X,2026-04-15,3.75\n"
        "X,2026-05-15,\n"          # leerer Wert
        "X,Sommer,3.60\n"          # unlesbare Periode
        "X,2026-06-15,NA\n"        # kein Zahlwert
        "X,2026-06-30,3.25\n"
    )
    monkeypatch.setattr(
        "broker.macro.europe.requests.get",
        lambda *a, **k: FakeResponse(text=csv_text),
    )
    series = EcbClient().fetch(ECB_SPEC)
    assert series is not None and series.value == pytest.approx(3.25)


def test_ecb_client_survives_error(monkeypatch):
    monkeypatch.setattr(
        "broker.macro.europe.requests.get",
        lambda *a, **k: FakeResponse(text="", status=500),
    )
    assert EcbClient().fetch(ECB_SPEC) is None


def test_specs_have_unique_keys():
    keys = [s.key for s in EUROSTAT_SERIES] + [s.key for s in ECB_SERIES]
    assert len(keys) == len(set(keys))
    assert all(k.startswith("ez_") for k in keys)


def test_fetch_european_series_derives_yield_curve(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if "eurostat" in url:
            raise OSError("Eurostat gerade nicht erreichbar")
        if "SR_10Y" in url:
            return FakeResponse(text="TIME_PERIOD,OBS_VALUE\n2026-06-30,3.20\n")
        if "SR_2Y" in url:
            return FakeResponse(text="TIME_PERIOD,OBS_VALUE\n2026-06-30,2.10\n")
        return FakeResponse(text="TIME_PERIOD,OBS_VALUE\n2026-06-30,3.75\n")

    monkeypatch.setattr("broker.macro.europe.requests.get", fake_get)
    series = fetch_european_series()

    assert "ez_yield_curve" in series
    assert series["ez_yield_curve"].value == pytest.approx(1.10)
    assert series["ez_policy_rate"].value == pytest.approx(3.75)
    # Eurostat war weg — der Rest läuft trotzdem.
    assert "ez_industrial_production" not in series


def test_fetch_european_series_survives_total_outage(monkeypatch):
    def boom(*a, **k):
        raise OSError("kein Netz")

    monkeypatch.setattr("broker.macro.europe.requests.get", boom)
    assert fetch_european_series() == {}


# --- Zusammenspiel mit dem Regime ----------------------------------------


def series(key: str, value: float, change_3m: float | None = None) -> MacroSeries:
    return MacroSeries(key=key, label=key, value=value, change_3m=change_3m, unit="%")


def test_regime_averages_us_and_european_rates():
    """Steigende US-Zinsen und fallende EZB-Zinsen heben sich auf."""
    regime = build_regime(
        {
            "fed_funds": series("fed_funds", 4.0, change_3m=0.60),
            "ez_policy_rate": series("ez_policy_rate", 3.0, change_3m=-0.60),
        }
    )
    assert regime.rate_direction == "neutral"


def test_regime_uses_european_rate_alone_when_fred_missing():
    regime = build_regime({"ez_policy_rate": series("ez_policy_rate", 3.0, change_3m=-0.80)})
    assert regime.rate_direction == "fallend"


def test_regime_averages_yield_curves():
    regime = build_regime(
        {
            "yield_curve": series("yield_curve", -0.40),
            "ez_yield_curve": series("ez_yield_curve", 1.60),
        }
    )
    # Mittel 0,6 -> steil, obwohl die US-Kurve für sich genommen invers ist.
    assert regime.curve_shape == "steil"


def test_regime_european_growth_data_moves_the_signal():
    weak = build_regime(
        {
            "ez_industrial_production": series("ez_industrial_production", 98.0, change_3m=-0.05),
            "ez_unemployment": series("ez_unemployment", 7.5, change_3m=0.8),
            "ez_gdp": series("ez_gdp", 105.0, change_3m=-0.02),
        }
    )
    strong = build_regime(
        {
            "ez_industrial_production": series("ez_industrial_production", 102.0, change_3m=0.05),
            "ez_unemployment": series("ez_unemployment", 6.0, change_3m=-0.8),
            "ez_gdp": series("ez_gdp", 108.0, change_3m=0.02),
        }
    )
    assert weak.growth_signal == "restriktiv"
    assert strong.growth_signal == "expansiv"


def test_bond_yield_prefers_european_series_outside_the_us():
    regime = MacroRegime(
        series={
            "us_10y": series("us_10y", 4.30),
            "ez_yield_10y": series("ez_yield_10y", 2.60),
        }
    )
    assert bond_yield_for(regime, "Germany") == pytest.approx(0.026)
    assert bond_yield_for(regime, "US") == pytest.approx(0.043)


def test_bond_yield_falls_back_to_us_when_europe_missing():
    regime = MacroRegime(series={"us_10y": series("us_10y", 4.30)})
    assert bond_yield_for(regime, "Germany") == pytest.approx(0.043)
