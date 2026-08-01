"""Chart-Analyse.

Bewusst auf das beschränkt, was sich sauber berechnen und begründen lässt:
Trendstruktur, Momentum, Volatilität, Volumen und relative Stärke. Keine
Mustererkennung à la Schulter-Kopf-Schulter — dafür gibt es keine belastbare
Evidenz, und ein Screener, der so etwas behauptet, erzeugt Scheingenauigkeit.

Die Chart-Seite dient hier einem einzigen Zweck: zu unterscheiden, ob ein
günstiger Titel gerade *ausverkauft* ist (Boden bildet sich, Momentum dreht)
oder ob er noch im freien Fall ist. Beim zweiten Fall ist das niedrige KGV
eine Falle, kein Angebot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from broker.models import PriceHistory, TechnicalResult

TRADING_DAYS = 252


def sma(series: pd.Series, window: int) -> float | None:
    if len(series) < window:
        return None
    return float(series.iloc[-window:].mean())


def rsi(series: pd.Series, window: int = 14) -> float | None:
    """Wilder-RSI. Werte < 30 gelten als überverkauft, > 70 als überkauft."""
    if len(series) < window + 1:
        return None
    delta = series.diff().dropna()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1 / window, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def macd_histogram(series: pd.Series) -> float | None:
    """MACD-Histogramm (12/26/9), normiert auf den Kurs.

    Die Normierung macht den Wert über Titel hinweg vergleichbar — ein
    Histogramm von 0,5 bedeutet bei einem 10-Euro-Wert etwas völlig anderes
    als bei einem 500-Euro-Wert.

    Wichtig: Das Vorzeichen allein ist kein Momentum-Signal. Bei einem stetigen
    exponentiellen Kursverfall schrumpft die Differenz der beiden EMAs mit dem
    Kursniveau, wodurch das Histogramm positiv wird, obwohl der Kurs
    ununterbrochen fällt. Deshalb wird es unten immer zusammen mit der
    tatsächlichen jüngsten Kursentwicklung ausgewertet.
    """
    if len(series) < 35:
        return None
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    price = float(series.iloc[-1])
    if price <= 0:
        return None
    return float((macd - signal).iloc[-1] / price)


def recent_return(series: pd.Series, days: int = 20) -> float | None:
    """Kursentwicklung der letzten `days` Handelstage."""
    if len(series) < days + 1:
        return None
    past = float(series.iloc[-days - 1])
    if past <= 0:
        return None
    return float(series.iloc[-1]) / past - 1.0


def annualized_volatility(series: pd.Series, window: int = 60) -> float | None:
    if len(series) < window + 1:
        return None
    returns = series.pct_change().dropna().iloc[-window:]
    if returns.empty:
        return None
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def atr(frame: pd.DataFrame, window: int = 14) -> float | None:
    """Average True Range — mittlere Tagesspanne inklusive Kurslücken.

    Anders als die Standardabweichung der Schlusskurse erfasst die True Range
    auch Overnight-Gaps. Für die Wahl einer Stopp-Distanz ist das die
    ehrlichere Größe, und genau dafür wird sie im Hebelmodul gebraucht.
    """
    if len(frame) < window + 1 or not {"High", "Low", "Close"} <= set(frame.columns):
        return None
    high, low, close = frame["High"], frame["Low"], frame["Close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    value = true_range.ewm(alpha=1 / window, adjust=False).mean().iloc[-1]
    return None if pd.isna(value) else float(value)


def bollinger(series: pd.Series, window: int = 20, sigma: float = 2.0):
    """Gibt (%B, Bandbreite) zurück.

    %B verortet den Kurs im Band: 0 = unteres Band, 1 = oberes Band. Werte
    unter 0 bedeuten, dass der Kurs unter das Band gefallen ist. Die
    Bandbreite misst, wie eng der Markt gerade ist — ein enges Band geht
    Ausbrüchen oft voraus.
    """
    if len(series) < window:
        return None, None
    window_values = series.iloc[-window:]
    middle = float(window_values.mean())
    deviation = float(window_values.std(ddof=0))
    if deviation == 0 or middle == 0:
        return None, None
    upper, lower = middle + sigma * deviation, middle - sigma * deviation
    price = float(series.iloc[-1])
    percent_b = (price - lower) / (upper - lower)
    return float(percent_b), float((upper - lower) / middle)


def stochastic(frame: pd.DataFrame, window: int = 14, smooth: int = 3):
    """Gibt (%K, %D) zurück — Position im Hoch-Tief-Bereich der letzten Tage.

    Ergänzt den RSI: Der RSI misst die Stärke der Bewegungen, die Stochastik
    die Lage im Kursband. Beide zugleich im überverkauften Bereich ist ein
    deutlich verlässlicheres Zeichen als eines von beiden.
    """
    if len(frame) < window + smooth or not {"High", "Low", "Close"} <= set(frame.columns):
        return None, None
    high = frame["High"].rolling(window).max()
    low = frame["Low"].rolling(window).min()
    span = high - low
    raw = ((frame["Close"] - low) / span.where(span != 0)) * 100.0
    raw = raw.dropna()
    if len(raw) < smooth:
        return None, None
    percent_k = float(raw.iloc[-1])
    percent_d = float(raw.iloc[-smooth:].mean())
    return percent_k, percent_d


def volume_trend(volume: pd.Series) -> float | None:
    """Verhältnis 20-Tage- zu 90-Tage-Durchschnittsvolumen."""
    if len(volume) < 90:
        return None
    short = float(volume.iloc[-20:].mean())
    long = float(volume.iloc[-90:].mean())
    if long <= 0:
        return None
    return short / long


def relative_strength(
    series: pd.Series, benchmark: pd.Series | None, window: int = 126
) -> float | None:
    """Performance-Differenz zum Index über ~6 Monate, in Prozentpunkten."""
    if benchmark is None or len(series) < window + 1 or len(benchmark) < window + 1:
        return None
    own = float(series.iloc[-1] / series.iloc[-window - 1] - 1.0)
    bench = float(benchmark.iloc[-1] / benchmark.iloc[-window - 1] - 1.0)
    return own - bench


def _classify_setup(
    above_sma200: bool | None,
    rsi14: float | None,
    macd_hist: float | None,
    drawdown: float | None,
    recent: float | None,
    recent_short: float | None,
) -> tuple[str, float]:
    """Ordnet das Chartbild einer von vier Lagen zu und gibt Punkte dafür.

    Für einen Value-Screener ist der interessante Fall nicht der intakte
    Aufwärtstrend (dann ist der Titel meist nicht mehr günstig), sondern die
    Bodenbildung nach einem Rückgang.

    Momentum gilt nur dann als aufwärts gerichtet, wenn das MACD-Histogramm
    positiv ist *und* der Kurs zuletzt tatsächlich gestiegen ist. Ohne die
    zweite Bedingung würde ein stetiger Kursverfall als Bodenbildung
    durchgehen — siehe die Anmerkung bei macd_histogram().
    """
    deep = drawdown is not None and drawdown <= -0.20
    momentum_up = (
        macd_hist is not None
        and macd_hist > 0
        and recent is not None
        and recent > 0
    )
    # "Überverkauft" ist nur dann eine Chance, wenn der Abwärtsdruck zumindest
    # kurzfristig nachlässt. Ein Titel mit RSI 5, der weiter täglich fällt, ist
    # ein fallendes Messer — der niedrige RSI ist dann Symptom, nicht Signal.
    stabilizing = recent_short is not None and recent_short >= 0.0
    oversold = rsi14 is not None and rsi14 < 35 and stabilizing

    # Der Rücksetzer-Fall wird vor dem Aufwärtstrend geprüft: ein Titel, der
    # tief unter seinem Hoch steht und gerade dreht, ist für einen
    # Value-Screener der interessantere Fall.
    if deep and momentum_up:
        return "Bodenbildung nach Rücksetzer", 85.0
    if above_sma200 and momentum_up:
        return "intakter Aufwärtstrend", 75.0
    if deep and oversold:
        return "überverkauft, Trendwende offen", 55.0
    if deep:
        return "fallendes Messer", 20.0
    if above_sma200:
        return "seitwärts über dem 200er-Schnitt", 60.0
    return "seitwärts unter dem 200er-Schnitt", 40.0


def analyze_technical(
    history: PriceHistory, benchmark: pd.Series | None = None
) -> TechnicalResult:
    """Score 0-100, höher = besseres Einstiegs-Setup."""
    close = history.close
    notes: list[str] = []

    if len(close) < 60:
        return TechnicalResult(
            score=0.0, setup="zu wenig Historie", notes=["Weniger als 60 Handelstage."]
        )

    price = float(close.iloc[-1])
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    rsi14 = rsi(close)
    macd_hist = macd_histogram(close)
    recent = recent_return(close, 20)
    recent_short = recent_return(close, 5)
    vol = annualized_volatility(close)
    atr14 = atr(history.frame)
    percent_b, bandwidth = bollinger(close)
    stoch_k, stoch_d = stochastic(history.frame)
    vol_trend = volume_trend(history.volume)
    rel_strength = relative_strength(close, benchmark)

    window = close.iloc[-min(len(close), TRADING_DAYS) :]
    high_52w = float(window.max())
    low_52w = float(window.min())
    drawdown = price / high_52w - 1.0 if high_52w > 0 else None
    upside = high_52w / price - 1.0 if price > 0 else None
    off_low = price / low_52w - 1.0 if low_52w > 0 else None

    above_sma200 = None if sma200 is None else price > sma200
    golden_cross = None if (sma50 is None or sma200 is None) else sma50 > sma200

    setup, setup_score = _classify_setup(
        above_sma200, rsi14, macd_hist, drawdown, recent, recent_short
    )

    components: list[tuple[float, float]] = [(setup_score, 0.35)]

    # RSI: Für einen Value-Einstieg ist der Bereich 30-50 am attraktivsten —
    # gedrückt, aber nicht im Absturz. Über 70 ist der Zug abgefahren.
    if rsi14 is not None:
        if rsi14 < 25:
            rsi_score = 55.0  # extrem überverkauft: oft noch kein Boden
        elif rsi14 < 50:
            rsi_score = 90.0 - (rsi14 - 25) * 0.4
        elif rsi14 < 70:
            rsi_score = 70.0 - (rsi14 - 50) * 1.5
        else:
            rsi_score = max(0.0, 40.0 - (rsi14 - 70) * 2.0)
        components.append((rsi_score, 0.20))
        if rsi14 < 30:
            notes.append(f"RSI {rsi14:.0f} — überverkauft.")
        elif rsi14 > 70:
            notes.append(f"RSI {rsi14:.0f} — überkauft, Rücksetzer wahrscheinlich.")

    # Abstand zum 52-Wochen-Hoch: Rückschlagpotenzial nach oben.
    if drawdown is not None:
        # -40% -> 100 Punkte, 0% -> 30 Punkte
        components.append((float(np.clip(30.0 - drawdown * 175.0, 0, 100)), 0.15))
        notes.append(f"{abs(drawdown) * 100:.0f}% unter dem 52-Wochen-Hoch.")

    # Relative Stärke: leichte Underperformance ist für Value gut, starke ist
    # ein Warnsignal — der Markt weiß meist etwas.
    if rel_strength is not None:
        if rel_strength < -0.35:
            rs_score = 25.0
            notes.append(
                f"{abs(rel_strength) * 100:.0f} Prozentpunkte schlechter als der "
                "Index — deutliche Schwäche, Ursache prüfen."
            )
        elif rel_strength < 0:
            rs_score = 75.0
        elif rel_strength < 0.20:
            rs_score = 65.0
        else:
            rs_score = 45.0
        components.append((rs_score, 0.15))

    # Bollinger %B: Ein Kurs am oder unter dem unteren Band ist gemessen an
    # der eigenen jüngsten Schwankungsbreite gedrückt — das ist der Zustand,
    # in dem ein günstig bewerteter Titel interessant wird.
    if percent_b is not None:
        if percent_b <= 0.0:
            bb_score = 85.0
        elif percent_b < 0.5:
            bb_score = 85.0 - percent_b * 60.0
        else:
            bb_score = max(0.0, 55.0 - (percent_b - 0.5) * 90.0)
        components.append((bb_score, 0.10))
        if percent_b <= 0.0:
            notes.append("Kurs unter dem unteren Bollinger-Band.")

    # Stochastik als zweite Meinung zum RSI. Nur wenn beide überverkauft sind,
    # gibt es Zusatzpunkte — ein einzelner Indikator taugt nicht als Beleg.
    if stoch_k is not None and rsi14 is not None:
        both_oversold = stoch_k < 20 and rsi14 < 40
        both_overbought = stoch_k > 80 and rsi14 > 60
        if both_oversold:
            components.append((85.0, 0.05))
            notes.append("RSI und Stochastik zugleich überverkauft.")
        elif both_overbought:
            components.append((20.0, 0.05))

    # Volumen: anziehendes Volumen bestätigt eine Trendwende.
    if vol_trend is not None:
        components.append((float(np.clip(40.0 + (vol_trend - 1.0) * 100.0, 0, 100)), 0.10))
        if vol_trend > 1.4:
            notes.append("Deutlich erhöhtes Handelsvolumen.")

    # Volatilität dämpft den Score, statt eigene Punkte zu vergeben.
    total_weight = sum(w for _, w in components)
    score = sum(s * w for s, w in components) / total_weight
    if vol is not None and vol > 0.60:
        score *= 0.85
        notes.append(f"Sehr hohe Volatilität ({vol * 100:.0f}% p.a.).")

    return TechnicalResult(
        score=float(np.clip(score, 0, 100)),
        price=round(price, 2),
        sma50=None if sma50 is None else round(sma50, 2),
        sma200=None if sma200 is None else round(sma200, 2),
        above_sma200=above_sma200,
        golden_cross=golden_cross,
        rsi14=None if rsi14 is None else round(rsi14, 1),
        macd_histogram=None if macd_hist is None else round(macd_hist, 5),
        drawdown_from_52w_high=None if drawdown is None else round(drawdown, 3),
        upside_to_52w_high=None if upside is None else round(upside, 3),
        distance_to_52w_low=None if off_low is None else round(off_low, 3),
        annualized_volatility=None if vol is None else round(vol, 3),
        atr14=None if atr14 is None else round(atr14, 3),
        atr_percent=None if (atr14 is None or not price) else round(atr14 / price, 4),
        bollinger_percent_b=None if percent_b is None else round(percent_b, 3),
        bollinger_bandwidth=None if bandwidth is None else round(bandwidth, 4),
        stochastic_k=None if stoch_k is None else round(stoch_k, 1),
        stochastic_d=None if stoch_d is None else round(stoch_d, 1),
        volume_trend=None if vol_trend is None else round(vol_trend, 2),
        relative_strength_6m=None if rel_strength is None else round(rel_strength, 3),
        setup=setup,
        notes=notes,
    )
