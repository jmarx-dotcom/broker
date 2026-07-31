"""Benachrichtigung über neue Treffer: Telegram oder E-Mail.

Beides ist optional. Ohne konfigurierte Zugangsdaten passiert schlicht nichts —
der Report liegt dann nur als Datei vor.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path

import requests

from broker.config import env
from broker.screener import ScreeningResult

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


@dataclass
class NotificationOutcome:
    """Trennt sauber zwischen 'nicht eingerichtet' und 'ging schief'."""

    sent: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def configured(self) -> bool:
        return bool(self.sent or self.failed)


def build_summary(result: ScreeningResult, limit: int = 10) -> str:
    """Kurzfassung als Klartext, für Telegram und den Mail-Betreff."""
    if not result.candidates:
        return "Aktien-Screening: keine Treffer über der Score-Schwelle."

    lines = [f"Aktien-Screening: {len(result.candidates)} Treffer", ""]
    for c in result.candidates[:limit]:
        pe = "–" if c.valuation.trailing_pe is None else f"{c.valuation.trailing_pe:.1f}"
        line = f"{c.total_score:.0f} | {c.name} ({c.ticker}) | KGV {pe}"
        if c.llm and c.llm.summary:
            line += f"\n     {c.llm.summary}"
        lines.append(line)

    if result.regime.live:
        lines += ["", f"Makro: {result.regime.summary}"]
    return "\n".join(lines)


# -- Telegram --------------------------------------------------------------


def telegram_configured() -> bool:
    return bool(env("TELEGRAM_BOT_TOKEN") and env("TELEGRAM_CHAT_ID"))


def send_telegram(text: str) -> bool:
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False

    try:
        response = requests.post(
            f"{TELEGRAM_API}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=20,
        )
    except Exception as exc:
        log.error("Telegram nicht erreichbar: %s", exc)
        return False

    if response.ok:
        return True

    # Telegram antwortet mit einem sprechenden Fehlertext — den wollen wir
    # sehen, statt nur des HTTP-Codes.
    detail = ""
    try:
        detail = response.json().get("description", "")
    except Exception:
        detail = response.text[:200]

    log.error("Telegram-Versand fehlgeschlagen (%s): %s", response.status_code, detail)
    if response.status_code == 401:
        log.error(
            "Der Bot-Token wird abgelehnt. Prüfen mit: "
            "curl %s/bot<TOKEN>/getMe — das muss die Bot-Daten zurückgeben.",
            TELEGRAM_API,
        )
    elif response.status_code in (400, 403):
        log.error(
            "Häufigste Ursache: Du hast dem Bot noch nie geschrieben. Ein Bot "
            "darf niemanden von sich aus anschreiben — öffne den Chat mit "
            "deinem Bot und drücke einmal Start."
        )
    return False


def check_telegram() -> tuple[bool, str]:
    """Prüft den Token über getMe, ohne eine Nachricht zu verschicken."""
    token = env("TELEGRAM_BOT_TOKEN")
    if not token:
        return False, "TELEGRAM_BOT_TOKEN ist nicht gesetzt."
    if not env("TELEGRAM_CHAT_ID"):
        return False, "TELEGRAM_CHAT_ID ist nicht gesetzt."

    try:
        response = requests.get(f"{TELEGRAM_API}/bot{token}/getMe", timeout=20)
    except Exception as exc:
        return False, f"Telegram nicht erreichbar: {exc}"

    if not response.ok:
        return False, (
            f"Token abgelehnt ({response.status_code}). Der Token sieht so aus: "
            "'8012345678:AAF…' — Zahlenblock, Doppelpunkt, Buchstabenblock."
        )

    name = response.json().get("result", {}).get("username", "?")
    return True, f"Token gültig, Bot: @{name}"


# -- E-Mail ----------------------------------------------------------------


def email_configured() -> bool:
    return all(
        env(name) for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_TO")
    )


def send_email(subject: str, body: str, attachment: Path | None = None) -> bool:
    host = env("SMTP_HOST")
    user = env("SMTP_USER")
    password = env("SMTP_PASSWORD")
    recipient = env("SMTP_TO")
    if not all((host, user, password, recipient)):
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = user
    message["To"] = recipient
    message.set_content(body)

    if attachment and attachment.is_file():
        message.add_attachment(
            attachment.read_bytes(),
            maintype="text",
            subtype="html",
            filename=attachment.name,
        )

    try:
        port = int(env("SMTP_PORT") or "587")
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(message)
        return True
    except Exception as exc:
        log.error("E-Mail-Versand fehlgeschlagen: %s", exc)
        return False


# -- Fassade ---------------------------------------------------------------


def send_all(subject: str, body: str, attachment: Path | None = None):
    """Verschickt über alle *eingerichteten* Kanäle und meldet, was klappte."""
    outcome = NotificationOutcome()

    if telegram_configured():
        (outcome.sent if send_telegram(body) else outcome.failed).append("Telegram")
    if email_configured():
        target = outcome.sent if send_email(subject, body, attachment) else outcome.failed
        target.append("E-Mail")

    if not outcome.configured:
        log.info("Kein Benachrichtigungskanal eingerichtet — nur Datei-Report.")
    elif outcome.failed:
        # Das war vorher als 'nicht konfiguriert' gemeldet worden und hat den
        # eigentlichen Fehler verdeckt.
        log.error(
            "Versand fehlgeschlagen über: %s. Die Zugangsdaten sind gesetzt, "
            "werden aber abgelehnt.",
            ", ".join(outcome.failed),
        )
    return outcome


def notify(result: ScreeningResult, report_path: Path | None = None):
    return send_all(
        subject=f"Aktien-Screening: {len(result.candidates)} Treffer",
        body=build_summary(result),
        attachment=report_path,
    )
