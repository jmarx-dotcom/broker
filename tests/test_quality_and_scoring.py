from __future__ import annotations

from broker.analysis.quality import analyze_quality
from broker.analysis.scoring import combine_scores
from broker.config import Weights
from broker.models import Fundamentals, QualityResult, TechnicalResult, ValuationResult


class TestQuality:
    def test_solid_scores_above_trap(self, solid_fundamentals, trap_fundamentals):
        solid = analyze_quality(solid_fundamentals)
        trap = analyze_quality(trap_fundamentals)
        assert solid.score > trap.score

    def test_trap_collects_multiple_red_flags(self, trap_fundamentals):
        result = analyze_quality(trap_fundamentals)
        joined = " ".join(result.red_flags)

        assert len(result.red_flags) >= 4
        assert "Umsatz schrumpft" in joined
        assert "Gewinn bricht" in joined
        assert "Ausschüttungsquote" in joined
        assert "Cashflow" in joined

    def test_net_cash_is_rewarded(self):
        f = Fundamentals(
            ticker="CASH", total_debt=1.0e8, total_cash=5.0e8, ebitda=2.0e8
        )
        result = analyze_quality(f)
        assert result.net_debt_to_ebitda is not None
        assert result.net_debt_to_ebitda < 0
        assert any("Nettoliquidität" in n for n in result.notes)

    def test_missing_data_yields_neutral_score(self):
        result = analyze_quality(Fundamentals(ticker="EMPTY"))
        assert result.score == 50.0
        assert result.red_flags == []

    def test_high_leverage_is_flagged(self):
        f = Fundamentals(ticker="DEBT", total_debt=5.0e9, total_cash=0.0, ebitda=1.0e9)
        result = analyze_quality(f)
        assert any("Nettoverschuldung" in flag for flag in result.red_flags)

    def test_score_stays_in_bounds(self, solid_fundamentals, trap_fundamentals):
        for f in (solid_fundamentals, trap_fundamentals, Fundamentals(ticker="X")):
            assert 0.0 <= analyze_quality(f).score <= 100.0


class TestScoring:
    weights = Weights()

    def _combine(self, value: float, quality: float, tech: float, macro: float) -> float:
        return combine_scores(
            ValuationResult(score=value),
            TechnicalResult(score=tech),
            QualityResult(score=quality),
            macro,
            self.weights,
        )

    def test_all_equal_returns_that_value(self):
        assert self._combine(70.0, 70.0, 70.0, 70.0) == 70.0

    def test_weak_quality_drags_a_cheap_stock_down(self):
        # Beide sind gleich billig; nur die Substanz unterscheidet sie.
        good = self._combine(90.0, 80.0, 60.0, 50.0)
        bad = self._combine(90.0, 10.0, 60.0, 50.0)
        assert good > bad

    def test_quality_dampener_applies_below_forty(self):
        # Ohne Dämpfung wäre der gewichtete Wert höher als das Ergebnis.
        raw = (90 * 0.40 + 20 * 0.25 + 60 * 0.25 + 50 * 0.10) / 1.0
        assert self._combine(90.0, 20.0, 60.0, 50.0) < raw

    def test_stays_in_bounds(self):
        assert self._combine(100.0, 100.0, 100.0, 100.0) <= 100.0
        assert self._combine(0.0, 0.0, 0.0, 0.0) >= 0.0

    def test_rejects_zero_weights(self):
        import pytest

        with pytest.raises(ValueError):
            combine_scores(
                ValuationResult(score=50.0),
                TechnicalResult(score=50.0),
                QualityResult(score=50.0),
                50.0,
                Weights(value=0, quality=0, technical=0, macro=0),
            )
