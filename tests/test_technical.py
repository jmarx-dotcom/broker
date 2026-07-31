from __future__ import annotations

import numpy as np
import pandas as pd

from broker.analysis.technical import (
    _classify_setup,
    analyze_technical,
    annualized_volatility,
    macd_histogram,
    recent_return,
    relative_strength,
    rsi,
    sma,
    volume_trend,
)
from tests.conftest import make_history


class TestIndicators:
    def test_sma_averages_the_window(self):
        series = pd.Series([float(i) for i in range(1, 101)])
        assert sma(series, 10) == 95.5  # Mittel aus 91..100

    def test_sma_returns_none_when_too_short(self):
        assert sma(pd.Series([1.0, 2.0]), 50) is None

    def test_rsi_is_100_for_pure_uptrend(self):
        series = pd.Series(np.arange(1.0, 60.0))
        assert rsi(series) == 100.0

    def test_rsi_is_low_for_pure_downtrend(self):
        series = pd.Series(np.arange(60.0, 1.0, -1.0))
        value = rsi(series)
        assert value is not None and value < 5.0

    def test_rsi_stays_in_bounds_for_noisy_series(self):
        rng = np.random.default_rng(7)
        series = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
        value = rsi(series)
        assert value is not None and 0.0 <= value <= 100.0

    def test_macd_histogram_positive_when_momentum_accelerates(self):
        # Beschleunigender Anstieg -> kurzer EMA zieht davon.
        series = pd.Series(np.exp(np.linspace(0, 1.2, 120)) * 100)
        value = macd_histogram(series)
        assert value is not None and value > 0

    def test_volatility_is_zero_for_flat_series(self):
        assert annualized_volatility(pd.Series([100.0] * 200)) == 0.0

    def test_volume_trend_detects_pickup(self):
        volume = pd.Series([1000.0] * 80 + [3000.0] * 20)
        value = volume_trend(volume)
        assert value is not None and value > 1.5

    def test_relative_strength_compares_against_benchmark(self):
        own = pd.Series(np.linspace(100, 130, 200))
        bench = pd.Series(np.linspace(100, 110, 200))
        value = relative_strength(own, bench)
        assert value is not None and value > 0

    def test_relative_strength_needs_benchmark(self):
        assert relative_strength(pd.Series(np.arange(300.0)), None) is None


class TestSetupClassification:
    """Direkte Abdeckung der Entscheidungstabelle in _classify_setup."""

    def classify(self, **kwargs):
        defaults = dict(
            above_sma200=False,
            rsi14=50.0,
            macd_hist=0.0,
            drawdown=-0.05,
            recent=0.0,
            recent_short=0.0,
        )
        defaults.update(kwargs)
        return _classify_setup(**defaults)[0]

    def test_deep_drawdown_with_real_momentum_is_a_bottom(self):
        assert (
            self.classify(drawdown=-0.35, macd_hist=0.001, recent=0.04)
            == "Bodenbildung nach Rücksetzer"
        )

    def test_bottom_beats_uptrend_when_both_apply(self):
        # Tief unterm Hoch und über dem 200er-Schnitt: der Rücksetzer-Fall
        # ist für einen Value-Screener der interessantere.
        assert (
            self.classify(
                drawdown=-0.35, macd_hist=0.001, recent=0.04, above_sma200=True
            )
            == "Bodenbildung nach Rücksetzer"
        )

    def test_positive_macd_without_price_gain_is_not_momentum(self):
        # Der Kern des Bugs: MACD-Histogramm positiv, Kurs fällt trotzdem.
        assert (
            self.classify(drawdown=-0.35, macd_hist=0.002, recent=-0.08,
                          recent_short=-0.03)
            == "fallendes Messer"
        )

    def test_oversold_needs_short_term_stabilisation(self):
        still_falling = self.classify(
            drawdown=-0.35, rsi14=20.0, recent=-0.08, recent_short=-0.03
        )
        stabilised = self.classify(
            drawdown=-0.35, rsi14=20.0, recent=-0.08, recent_short=0.01
        )

        assert still_falling == "fallendes Messer"
        assert stabilised == "überverkauft, Trendwende offen"

    def test_shallow_drawdown_falls_back_to_sideways(self):
        assert self.classify(above_sma200=True) == "seitwärts über dem 200er-Schnitt"
        assert self.classify(above_sma200=False) == "seitwärts unter dem 200er-Schnitt"

    def test_missing_indicators_do_not_crash(self):
        assert self.classify(
            above_sma200=None, rsi14=None, macd_hist=None,
            drawdown=None, recent=None, recent_short=None,
        )


class TestAnalyzeTechnical:
    def test_rejects_too_short_history(self):
        result = analyze_technical(make_history(days=30))
        assert result.score == 0.0
        assert result.setup == "zu wenig Historie"

    def test_uptrend_is_above_sma200(self):
        result = analyze_technical(make_history(days=400, trend=0.0012))
        assert result.above_sma200 is True
        assert result.golden_cross is True

    def test_steady_decline_is_a_falling_knife_not_a_bottom(self):
        """Regression: das MACD-Histogramm allein war hier irreführend.

        Bei stetigem exponentiellem Verfall wird es positiv, weil die
        EMA-Differenz mit dem Kursniveau schrumpft — der Titel wurde dadurch
        als 'Bodenbildung' eingestuft, obwohl er ununterbrochen fiel.
        """
        knife = analyze_technical(make_history(days=400, trend=-0.0025, seed=1))

        assert knife.setup == "fallendes Messer"
        assert knife.macd_histogram is not None and knife.macd_histogram > 0

    def test_recovery_after_selloff_scores_above_falling_knife(self):
        knife = analyze_technical(make_history(days=400, trend=-0.0025, seed=1))
        recovery = analyze_technical(self._selloff_then_recovery())

        assert recovery.setup == "Bodenbildung nach Rücksetzer"
        assert recovery.score > knife.score

    @staticmethod
    def _selloff_then_recovery():
        """Langer Absturz, dann eine kurze Erholung — noch tief unterm Hoch."""
        falling = make_history(days=370, trend=-0.003, seed=2).frame
        rising = make_history(
            days=30, start=float(falling["Close"].iloc[-1]), trend=0.004, seed=3
        ).frame
        rising.index = pd.bdate_range(
            start=falling.index[-1] + pd.Timedelta(days=1), periods=len(rising)
        )
        combined = make_history(days=1)
        combined.frame = pd.concat([falling, rising])
        return combined

    def test_drawdown_and_upside_are_consistent(self):
        result = analyze_technical(make_history(days=400, trend=-0.001))
        assert result.drawdown_from_52w_high is not None
        assert result.drawdown_from_52w_high < 0
        assert result.upside_to_52w_high is not None
        assert result.upside_to_52w_high > 0

    def test_high_volatility_dampens_score(self):
        # Gleicher Kursverlauf, nur mit aufgesetztem Zittern: Trend, Drawdown
        # und Schlusskurs bleiben gleich, allein die Schwankung steigt.
        base = make_history(days=400, trend=0.0008, noise=0.004, seed=5)
        jittered = make_history(days=400, trend=0.0008, noise=0.004, seed=5)
        jitter = np.tile([1.06, 0.94], len(jittered.frame) // 2)
        jitter[-1] = 1.0  # Schlusskurs unverändert lassen
        jittered.frame["Close"] = jittered.frame["Close"].to_numpy() * jitter

        calm = analyze_technical(base)
        wild = analyze_technical(jittered)

        assert calm.setup == wild.setup, "Setup muss identisch bleiben"
        assert wild.annualized_volatility > 0.60
        assert any("Volatilität" in n for n in wild.notes)
        assert wild.score < calm.score

    def test_score_stays_in_bounds(self):
        for trend in (-0.004, -0.001, 0.0, 0.001, 0.004):
            result = analyze_technical(make_history(days=400, trend=trend, noise=0.01))
            assert 0.0 <= result.score <= 100.0
