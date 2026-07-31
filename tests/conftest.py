"""Synthetische Fixtures — alle Tests laufen ohne Netzwerkzugriff."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from broker.models import Fundamentals, PriceHistory


def make_history(
    ticker: str = "TEST",
    days: int = 750,
    start: float = 100.0,
    trend: float = 0.0,
    noise: float = 0.0,
    seed: int = 42,
) -> PriceHistory:
    """Kurshistorie mit definiertem Trend. `trend` ist die tägliche Drift."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=days)
    steps = trend + rng.normal(0.0, noise, days) if noise else np.full(days, trend)
    close = start * np.exp(np.cumsum(steps))
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(days, 1_000_000.0),
        },
        index=dates,
    )
    return PriceHistory(ticker=ticker, frame=frame)


def make_quarterly_eps(
    quarters: int = 16, base: float = 1.0, growth: float = 0.0
) -> pd.Series:
    """Quartals-EPS mit konstantem Wachstum pro Quartal."""
    dates = pd.date_range(end=pd.Timestamp("2026-06-30"), periods=quarters, freq="QE")
    values = [base * (1 + growth) ** i for i in range(quarters)]
    return pd.Series(values, index=dates)


@pytest.fixture
def flat_history() -> PriceHistory:
    return make_history(trend=0.0)


@pytest.fixture
def solid_fundamentals() -> Fundamentals:
    return Fundamentals(
        ticker="SOLID",
        name="Solide AG",
        sector="Industrials",
        country="Germany",
        currency="EUR",
        market_cap=5.0e9,
        trailing_eps=5.0,
        forward_eps=5.5,
        trailing_pe=12.0,
        forward_pe=10.5,
        dividend_yield=0.03,
        payout_ratio=0.4,
        total_debt=1.0e9,
        total_cash=6.0e8,
        ebitda=8.0e8,
        free_cashflow=4.0e8,
        revenue_growth=0.06,
        earnings_growth=0.10,
        return_on_equity=0.18,
        profit_margin=0.11,
        quarterly_eps=make_quarterly_eps(growth=0.02),
    )


@pytest.fixture
def trap_fundamentals() -> Fundamentals:
    """Der klassische Value-Fallen-Kandidat: optisch billig, real kaputt."""
    return Fundamentals(
        ticker="TRAP",
        name="Falle SE",
        sector="Industrials",
        country="Germany",
        currency="EUR",
        market_cap=8.0e8,
        trailing_eps=2.0,
        forward_eps=0.9,
        trailing_pe=6.0,
        forward_pe=13.0,  # Forward über Trailing: Gewinne brechen ein
        dividend_yield=0.09,
        payout_ratio=1.4,  # Dividende aus der Substanz
        total_debt=3.0e9,
        total_cash=1.0e8,
        ebitda=5.0e8,  # Nettoschulden/EBITDA ~5.8
        free_cashflow=-1.0e8,
        revenue_growth=-0.18,
        earnings_growth=-0.45,
        return_on_equity=-0.05,
        profit_margin=-0.02,
        quarterly_eps=make_quarterly_eps(base=2.0, growth=-0.05),
    )
