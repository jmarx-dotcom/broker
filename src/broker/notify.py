"""Benachrichtigung über neue Treffer: Telegram oder E-Mail.

Beides ist optional. Ohne konfigurierte Zugangsdaten passiert schlicht nichts —
der Report liegt dann nur als Datei vor.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import requests

from broker.screener import ScreeningResult

log = logging.getLogger(__name__)


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


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=20,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        log.error("Telegram-Versand fehlgeschlagen: %s", exc)
        return False


def send_email(subject: str, body: str, attachment: Path | None = None) -> bool:
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    recipient = os.environ.get("SMTP_TO")
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
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(message)
        return True
    except Exception as exc:
        log.error("E-Mail-Versand fehlgeschlagen: %s", exc)
        return False


def notify(result: ScreeningResult, report_path: Path | None = None) -> list[str]:
    """Verschickt die Zusammenfassung über alle konfigurierten Kanäle."""
    summary = build_summary(result)
    used: list[str] = []

    if send_telegram(summary):
        used.append("telegram")
    if send_email(
        f"Aktien-Screening: {len(result.candidates)} Treffer", summary, report_path
    ):
        used.append("email")

    if not used:
        log.info("Kein Benachrichtigungskanal konfiguriert — nur Datei-Report.")
    return used
