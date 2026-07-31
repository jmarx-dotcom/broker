"""Tests der Plausibilitätsprüfung.

Der Leitgedanke: Ein falsches KGV erzeugt keinen Fehler, sondern einen besonders
attraktiven Treffer. Diese Prüfung soll genau solche Fälle sichtbar machen —
und darf gleichzeitig bei sauberen Daten nicht ständig Fehlalarm schlagen.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from broker.analysis.dataquality import (
    JUMP_THRESHOLD,
    check_data_quality,
    _check_market_cap,
    _check_pe,
    _check_price_jump,
    _check_staleness,
)
from broker.analysis.scoring import combine_scores
from broker.analysis.valuation import _winsorize, sector_median_pe
from broker.config import Weights
from broker.models import (
    DataFlag,
    DataQualityResult,
    Fundamentals,
    PriceHistory,
    QualityResult,
    TechnicalResult,
    ValuationResult,
)
from tests.conftest import make_history


def history_ending(end: date, days: int = 300, start_price: float = 100.0) -> PriceHistory:
    index = pd.bdate_range(end=pd.Timestamp(end), periods=days)
    frame = pd.DataFrame(
        {
            "Open": start_price, "High": start_price, "Low": start_price,
            "Close": start_price, "Volume": 1_000_000.0,
        },
        index=index,
    )
    return PriceHistory(ticker="TEST", frame=frame)


class TestPeConsistency:
    def test_consistent_values_produce_no_flag(self):
        # 100 / 5 = 20, gemeldet 20
        assert _check_pe("KGV", 100.0, 5.0, 20.0) is None

    def test_small_deviation_is_tolerated(self):
        # 100 / 5 = 20 gegen gemeldete 21 -> knapp 5%
        assert _check_pe("KGV", 100.0, 5.0, 21.0) is None

    def test_moderate_deviation_is_a_hint(self):
        # 100 / 5 = 20 gegen gemeldete 17 -> 18%
        flag = _check_pe("KGV", 100.0, 5.0, 17.0)
        assert flag is not None
        assert flag.severe is False
        assert "17" in flag.message and "20" in flag.message

    def test_unprocessed_split_is_severe(self):
        # Klassischer Fall: Kurs nach 2:1-Split halbiert, EPS noch alt.
        flag = _check_pe("KGV", 50.0, 5.0, 20.0)
        assert flag is not None
        assert flag.severe is True
        assert "Aktiensplit" in flag.message

    def test_loss_case_is_not_flagged(self):
        # Negatives EPS wird an anderer Stelle behandelt.
        assert _check_pe("KGV", 100.0, -5.0, 20.0) is None
        assert _check_pe("KGV", 100.0, 5.0, -20.0) is None

    def test_missing_inputs_produce_no_flag(self):
        assert _check_pe("KGV", None, 5.0, 20.0) is None
        assert _check_pe("KGV", 100.0, None, 20.0) is None
        assert _check_pe("KGV", 100.0, 5.0, None) is None

    def test_label_appears_in_the_flag(self):
        flag = _check_pe("Erwartetes KGV", 100.0, 5.0, 10.0)
        assert flag.check == "Erwartetes KGV-Konsistenz"


class TestMarketCapConsistency:
    def test_consistent_values_produce_no_flag(self):
        assert _check_market_cap(50.0, 2.0e8, 1.0e10) is None

    def test_currency_mismatch_is_severe(self):
        # Kurs in einer Währung, Marktkapitalisierung in einer anderen.
        flag = _check_market_cap(50.0, 2.0e8, 1.2e9)
        assert flag is not None
        assert flag.severe is True
        assert "Währungen" in flag.message

    def test_rounding_noise_is_tolerated(self):
        assert _check_market_cap(50.0, 2.0e8, 1.05e10) is None

    def test_missing_shares_produce_no_flag(self):
        assert _check_market_cap(50.0, None, 1.0e10) is None


class TestStaleness:
    def test_fresh_data_produces_no_flag(self):
        today = date(2026, 7, 1)
        assert _check_staleness(history_ending(today), today) is None

    def test_two_week_old_data_is_a_hint(self):
        today = date(2026, 7, 20)
        flag = _check_staleness(history_ending(date(2026, 7, 5)), today)
        assert flag is not None
        assert flag.severe is False

    def test_two_month_old_data_is_severe(self):
        today = date(2026, 7, 20)
        flag = _check_staleness(history_ending(date(2026, 5, 1)), today)
        assert flag is not None
        assert flag.severe is True
        assert "delistet" in flag.message

    def test_empty_history_is_severe(self):
        empty = PriceHistory(
            ticker="X",
            frame=pd.DataFrame({"Close": [], "Volume": []}, index=pd.DatetimeIndex([])),
        )
        flag = _check_staleness(empty, date(2026, 7, 1))
        assert flag is not None and flag.severe is True


class TestPriceJump:
    def test_smooth_series_produces_no_flag(self):
        assert _check_price_jump(make_history(days=300, trend=0.001)) is None

    def test_split_sized_jump_is_flagged(self):
        history = make_history(days=300, trend=0.0)
        # Kurs halbiert sich über Nacht — typisch für einen 2:1-Split.
        history.frame.iloc[-50:, history.frame.columns.get_loc("Close")] *= 0.5
        flag = _check_price_jump(history)

        assert flag is not None
        assert flag.severe is True
        assert "Aktiensplit" in flag.message

    def test_moderate_jump_is_only_a_hint(self):
        history = make_history(days=300, trend=0.0)
        history.frame.iloc[-50:, history.frame.columns.get_loc("Close")] *= 0.58
        flag = _check_price_jump(history)

        assert flag is not None
        assert flag.severe is False

    def test_threshold_is_not_triggered_just_below(self):
        history = make_history(days=300, trend=0.0)
        factor = 1.0 - (JUMP_THRESHOLD - 0.05)
        history.frame.iloc[-50:, history.frame.columns.get_loc("Close")] *= factor
        assert _check_price_jump(history) is None

    def test_short_history_is_skipped(self):
        assert _check_price_jump(make_history(days=10)) is None


class TestCompleteness:
    def test_missing_currency_and_sector_are_flagged(self):
        f = Fundamentals(ticker="X", currency=None, sector=None)
        result = check_data_quality(f, make_history(days=300), today=date(2026, 6, 30))
        checks = [flag.message for flag in result.flags]

        assert any("Währung" in m for m in checks)
        assert any("Sektor" in m for m in checks)

    def test_complete_data_produces_no_completeness_flag(self):
        f = Fundamentals(ticker="X", currency="EUR", sector="Industrials")
        result = check_data_quality(f, make_history(days=300), today=date(2026, 6, 30))
        assert not any(f.check == "Vollständigkeit" for f in result.flags)


class TestEndToEnd:
    @staticmethod
    def clean_fundamentals() -> Fundamentals:
        return Fundamentals(
            ticker="CLEAN.DE", currency="EUR", sector="Industrials",
            trailing_eps=5.0, trailing_pe=20.0,
            forward_eps=5.5, forward_pe=18.2,
            shares_outstanding=1.0e8, market_cap=1.0e10,
        )

    def test_clean_data_is_trustworthy(self):
        history = make_history(days=300, start=100.0, trend=0.0)
        result = check_data_quality(
            self.clean_fundamentals(), history, today=history.close.index[-1].date()
        )

        assert result.flags == []
        assert result.trustworthy is True
        assert result.penalty == 1.0

    def test_broken_pe_is_caught_and_penalised(self):
        f = self.clean_fundamentals()
        f.trailing_pe = 8.0  # Kurs 100 / EPS 5 wäre 20 — hier viel zu billig
        history = make_history(days=300, start=100.0, trend=0.0)

        result = check_data_quality(f, history, today=history.close.index[-1].date())

        assert result.trustworthy is False
        assert result.penalty == 0.6
        assert any(flag.check == "KGV-Konsistenz" for flag in result.flags)

    def test_hint_only_gives_the_milder_penalty(self):
        f = self.clean_fundamentals()
        f.trailing_pe = 17.0  # 15% Abweichung: auffällig, aber nicht schwer
        history = make_history(days=300, start=100.0, trend=0.0)

        result = check_data_quality(f, history, today=history.close.index[-1].date())

        assert result.trustworthy is True
        assert result.penalty == 0.9


class TestScoringPenalty:
    weights = Weights()

    def _score(self, data_quality: DataQualityResult | None) -> float:
        return combine_scores(
            ValuationResult(score=80.0),
            TechnicalResult(score=70.0),
            QualityResult(score=70.0),
            60.0,
            self.weights,
            data_quality,
        )

    def test_clean_data_is_not_penalised(self):
        assert self._score(DataQualityResult()) == self._score(None)

    def test_severe_flag_lowers_the_score(self):
        clean = self._score(DataQualityResult())
        broken = self._score(
            DataQualityResult(flags=[DataFlag("KGV", "kaputt", severe=True)])
        )
        assert broken == pytest.approx(clean * 0.6)

    def test_hint_lowers_the_score_less_than_a_severe_flag(self):
        hint = self._score(DataQualityResult(flags=[DataFlag("KGV", "auffällig")]))
        severe = self._score(
            DataQualityResult(flags=[DataFlag("KGV", "kaputt", severe=True)])
        )
        assert severe < hint < self._score(DataQualityResult())

    def test_a_broken_stock_cannot_outrank_a_clean_one_on_data_alone(self):
        """Der eigentliche Zweck: kein Aufstieg durch ein falsches KGV."""
        broken = combine_scores(
            ValuationResult(score=95.0),  # sieht dank Datenfehler spitze aus
            TechnicalResult(score=70.0),
            QualityResult(score=70.0),
            60.0,
            self.weights,
            DataQualityResult(flags=[DataFlag("KGV", "kaputt", severe=True)]),
        )
        clean = combine_scores(
            ValuationResult(score=75.0),
            TechnicalResult(score=70.0),
            QualityResult(score=70.0),
            60.0,
            self.weights,
            DataQualityResult(),
        )
        assert clean > broken


class TestWinsorize:
    def test_extreme_value_is_pulled_to_the_edge(self):
        values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 90.0]
        result = _winsorize(values)
        assert max(result) < 90.0
        assert min(result) >= 10.0

    def test_small_samples_are_left_alone(self):
        values = [10.0, 90.0, 12.0]
        assert _winsorize(values) == values

    def test_median_is_more_stable_with_an_outlier(self):
        peers = [
            Fundamentals(ticker=f"T{i}", sector="Energy", trailing_pe=pe)
            for i, pe in enumerate([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        ]
        outlier = [Fundamentals(ticker="X", sector="Energy", trailing_pe=95.0)]

        without = sector_median_pe(peers)["Energy"]
        with_outlier = sector_median_pe(peers + outlier)["Energy"]

        # Der Ausreißer verschiebt den Median, aber nur um einen halben Punkt.
        assert abs(with_outlier - without) <= 1.0
