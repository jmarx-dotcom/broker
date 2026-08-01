"""Plausibilitätsprüfung der Rohdaten.

Der Screener glaubt dem Datenanbieter bislang jede Zahl. Das ist die riskanteste
Stelle des ganzen Werkzeugs: Ein falsches KGV erzeugt keinen Fehler, sondern
einen besonders attraktiven Treffer — genau den, der oben auf der Liste landet
und den man kauft. Die häufigsten Ursachen sind ein nicht verarbeiteter
Aktiensplit, ein veralteter Gewinn und Kennzahlen, die in unterschiedlichen
Währungen gemeldet werden.

Solche Fehler lassen sich billig erkennen, weil die Kennzahlen redundant sind:
Kurs geteilt durch Gewinn je Aktie *muss* das gemeldete KGV ergeben, und
Kurs mal Aktienzahl *muss* die Marktkapitalisierung ergeben. Wo das nicht
aufgeht, stimmt etwas nicht — und man weiß es, ohne die Wahrheit zu kennen.

Betroffene Titel fliegen nicht raus. Sie werden im Report benannt und im Score
gedämpft: Ein stiller Ausschluss verdeckt genau die Fälle, die man sehen will.
"""

from __future__ import annotations

from datetime import date, timedelta

from broker.models import DataFlag, DataQualityResult, Fundamentals, PriceHistory

#: Ab dieser relativen Abweichung gilt eine Kennzahl als auffällig.
TOLERANCE_HINT = 0.10
#: Ab hier ist der Widerspruch so groß, dass die Bewertung unbrauchbar ist.
TOLERANCE_SEVERE = 0.35
#: Marktkapitalisierung reagiert empfindlicher auf Rundung und Zeitversatz.
MCAP_TOLERANCE_HINT = 0.15
MCAP_TOLERANCE_SEVERE = 0.50
#: Bis zu diesem Vielfachen erklären mehrere Aktiengattungen den Unterschied.
#: Kein Unternehmen im Universum hat mehr als eine Handvoll Gattungen; darüber
#: ist die Aktienzahl schlicht falsch.
MCAP_MULTIPLE_SHARE_CLASSES = 5.0
#: Ein Tagessprung darüber ist auffällig.
JUMP_THRESHOLD = 0.40
#: Ab hier sind die Chart-Kennzahlen in jedem Fall unbrauchbar. Der Wert liegt
#: bewusst bei 50%: Genau so zeigt sich ein nicht verarbeiteter 2:1-Split, der
#: häufigste Datenfehler dieser Art.
JUMP_SEVERE = 0.50
#: Ältere Kursdaten sprechen für ausgesetzten Handel oder Delisting.
STALE_HINT_DAYS = 10
STALE_SEVERE_DAYS = 30


def _relative_deviation(actual: float, expected: float) -> float | None:
    if expected == 0:
        return None
    return abs(actual - expected) / abs(expected)


def _check_pe(
    label: str, price: float | None, eps: float | None, reported_pe: float | None
) -> DataFlag | None:
    """Kurs / Gewinn je Aktie muss dem gemeldeten KGV entsprechen."""
    if price is None or eps is None or reported_pe is None:
        return None
    if eps <= 0 or reported_pe <= 0:
        return None  # Verlustfall — anderswo behandelt

    implied = price / eps
    deviation = _relative_deviation(implied, reported_pe)
    if deviation is None or deviation < TOLERANCE_HINT:
        return None

    return DataFlag(
        check=f"{label}-Konsistenz",
        message=(
            f"{label} laut Anbieter {reported_pe:.1f}, aus Kurs und Gewinn je "
            f"Aktie errechnet sich aber {implied:.1f} "
            f"({deviation * 100:.0f}% Abweichung). Mögliche Ursachen: nicht "
            f"verarbeiteter Aktiensplit, veralteter Gewinn oder abweichende "
            f"Währungen."
        ),
        severe=bool(deviation >= TOLERANCE_SEVERE),
    )


def _check_market_cap(
    price: float | None, shares: float | None, market_cap: float | None
) -> DataFlag | None:
    """Kurs mal Aktienzahl muss die Marktkapitalisierung ergeben.

    Die Richtung der Abweichung entscheidet, ob es sich um einen Fehler
    handelt — siehe die beiden Zweige unten.
    """
    if price is None or shares is None or market_cap is None:
        return None
    if shares <= 0 or market_cap <= 0 or price <= 0:
        return None

    implied = price * shares
    deviation = _relative_deviation(implied, market_cap)
    if deviation is None or deviation < MCAP_TOLERANCE_HINT:
        return None

    numbers = (
        f"Marktkapitalisierung laut Anbieter {market_cap / 1e9:.2f} Mrd., "
        f"aus Kurs und Aktienzahl errechnet sich {implied / 1e9:.2f} Mrd. "
        f"({deviation * 100:.0f}% Abweichung)."
    )

    if market_cap > implied:
        # Der Anbieter meldet den Wert des ganzen Unternehmens, die Aktienzahl
        # aber nur für die abgefragte Gattung. Bei deutschen Vorzugsaktien —
        # Kürzel auf 3, etwa VOW3 oder HEN3 — ist das der Regelfall, und der
        # Faktor entspricht dann genau dem Verhältnis aller Aktien zu denen
        # dieser Gattung.
        #
        # Folgenlos für die Bewertung: EV/EBITDA und FCF-Rendite setzen den
        # Unternehmenswert ins Verhältnis zu Unternehmenszahlen, beide Seiten
        # beziehen sich also aufs ganze Unternehmen. KGV und KBV rechnen
        # ohnehin je Aktie. Die errechnete Marktkapitalisierung selbst
        # verwendet der Screener nirgends.
        multiple = market_cap / implied
        if multiple <= MCAP_MULTIPLE_SHARE_CLASSES:
            return DataFlag(
                check="Marktkapitalisierung",
                message=(
                    f"{numbers} Der Anbieter meldet den Wert des ganzen "
                    f"Unternehmens, die Aktienzahl nur für diese Gattung "
                    f"(Faktor {multiple:.1f}) — typisch für Titel mit Stamm- "
                    f"und Vorzugsaktien. Für die Bewertung folgenlos."
                ),
                informational=True,
            )
        return DataFlag(
            check="Marktkapitalisierung",
            message=(
                f"{numbers} Die gemeldete Marktkapitalisierung ist das "
                f"{multiple:.0f}-fache des Werts aus Kurs und Aktienzahl. Mit "
                f"mehreren Aktiengattungen ist das nicht mehr zu erklären — "
                f"wahrscheinlich ist die Aktienzahl veraltet oder falsch."
            ),
            severe=True,
        )

    # Die gemeldete Marktkapitalisierung ist zu klein. Das trifft anders als
    # der Fall oben genau die Kennzahlen, die auf ihr aufbauen.
    return DataFlag(
        check="Marktkapitalisierung",
        message=(
            f"{numbers} Die gemeldete Marktkapitalisierung ist zu klein für "
            f"Kurs und Aktienzahl. Häufigste Ursache: beide in verschiedenen "
            f"Währungen, etwa Pence gegen Pfund. EV/EBITDA und FCF-Rendite "
            f"sind dann verzerrt."
        ),
        severe=bool(deviation >= MCAP_TOLERANCE_SEVERE),
    )


def _check_staleness(history: PriceHistory, today: date) -> DataFlag | None:
    close = history.close
    if close.empty:
        return DataFlag("Kurshistorie", "Keine Kursdaten vorhanden.", severe=True)

    last = close.index[-1]
    last_date = last.date() if hasattr(last, "date") else last
    age = (today - last_date).days
    if age < STALE_HINT_DAYS:
        return None

    return DataFlag(
        check="Aktualität",
        message=(
            f"Letzter Kurs ist {age} Tage alt ({last_date}). Der Handel könnte "
            f"ausgesetzt oder der Titel delistet sein."
        ),
        severe=bool(age >= STALE_SEVERE_DAYS),
    )


def _check_price_jump(history: PriceHistory) -> DataFlag | None:
    """Ein einzelner Tagessprung von 40%+ ist fast immer ein Datenartefakt."""
    close = history.close
    if len(close) < 30:
        return None

    window = close.iloc[-252:]
    changes = window.pct_change().dropna()
    if changes.empty:
        return None

    largest = changes.abs().max()
    if largest < JUMP_THRESHOLD:
        return None

    when = changes.abs().idxmax()
    when_date = when.date() if hasattr(when, "date") else when
    return DataFlag(
        check="Kurssprung",
        message=(
            f"Tagesveränderung von {largest * 100:.0f}% am {when_date}. Das ist "
            f"entweder ein nicht verarbeiteter Aktiensplit oder ein echter "
            f"Kurssturz — in beiden Fällen sind die daraus abgeleiteten "
            f"Chart-Kennzahlen für diesen Zeitraum verzerrt."
        ),
        severe=bool(largest >= JUMP_SEVERE),
    )


def _check_completeness(f: Fundamentals) -> list[DataFlag]:
    flags: list[DataFlag] = []
    if not f.currency:
        flags.append(
            DataFlag(
                "Vollständigkeit",
                "Keine Währung gemeldet — Beträge sind nicht einzuordnen.",
            )
        )
    if not f.sector:
        flags.append(
            DataFlag(
                "Vollständigkeit",
                "Kein Sektor gemeldet — der Branchenvergleich entfällt.",
            )
        )
    return flags


def check_data_quality(
    fundamentals: Fundamentals,
    history: PriceHistory,
    today: date | None = None,
) -> DataQualityResult:
    """Prüft die Rohdaten eines Titels auf innere Widersprüche."""
    today = today or date.today()
    price = history.last_close

    candidates = [
        _check_pe("KGV", price, fundamentals.trailing_eps, fundamentals.trailing_pe),
        _check_pe(
            "Erwartetes KGV",
            price,
            fundamentals.forward_eps,
            fundamentals.forward_pe,
        ),
        _check_market_cap(
            price, fundamentals.shares_outstanding, fundamentals.market_cap
        ),
        _check_staleness(history, today),
        _check_price_jump(history),
    ]

    flags = [flag for flag in candidates if flag is not None]
    flags.extend(_check_completeness(fundamentals))
    return DataQualityResult(flags=flags)
