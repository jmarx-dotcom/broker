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
    """Kurzfassung als Klartext, für Telegram und den Mail-Betreff.

    Ein gestörter Lauf darf sich nicht als ruhiger Markt ausgeben. Am 4. und
    5. August meldete diese Funktion zwei Tage lang "keine Treffer über der
    Score-Schwelle", während von 217 gefilterten Titeln genau einer bewertet
    worden war — die übrigen 216 hatte Yahoo abgewiesen. Die Nachricht war
    beruhigend und falsch, und ohne einen Blick ins Log war ihr das nicht
    anzusehen.
    """
    if result.trouble:
        return (
            "Aktien-Screening: Lauf unvollständig — keine Aussage möglich.\n\n"
            f"{result.trouble}\n\n"
            "Es wurde nichts ins Journal übernommen."
        )

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


def discover_chats(token: str) -> list[tuple[str, str]]:
    """Listet die Chats, aus denen der Bot zuletzt Nachrichten bekommen hat.

    Telegram hält eingegangene Nachrichten rund 24 Stunden vor. Wer dem Bot
    einmal geschrieben hat, findet hier seine Chat-ID — das erspart die Suche
    danach, welche Nummer in TELEGRAM_CHAT_ID gehört.

    Leere Liste heißt: Der Bot hat noch nie eine Nachricht erhalten. Genau
    dann darf er auch keine verschicken.
    """
    try:
        response = requests.get(f"{TELEGRAM_API}/bot{token}/getUpdates", timeout=20)
        updates = response.json().get("result", []) if response.ok else []
    except Exception as exc:
        log.debug("getUpdates nicht abrufbar: %s", exc)
        return []

    chats: dict[str, str] = {}
    for update in updates if isinstance(updates, list) else []:
        if not isinstance(update, dict):
            continue
        message = (
            update.get("message")
            or update.get("edited_message")
            or update.get("channel_post")
            or {}
        )
        chat = message.get("chat") or {}
        if not isinstance(chat, dict):
            continue
        chat_id = chat.get("id")
        if chat_id is None:
            continue
        label = (
            chat.get("title")
            or " ".join(
                part for part in (chat.get("first_name"), chat.get("last_name")) if part
            )
            or chat.get("username")
            or chat.get("type")
            or "?"
        )
        chats[str(chat_id)] = label
    return sorted(chats.items())


def _log_chat_hint(token: str, chat_id: str) -> None:
    """Sagt bei 'chat not found', welche IDs tatsächlich in Frage kommen."""
    chats = discover_chats(token)
    if not chats:
        log.error(
            "Der Bot hat noch nie eine Nachricht erhalten. Ein Bot darf "
            "niemanden von sich aus anschreiben — öffne in Telegram den Chat "
            "mit deinem Bot und drücke einmal Start. Wichtig: Bei einem neu "
            "angelegten Bot zählt der alte Chat nicht, Start muss für den "
            "neuen Bot noch einmal gedrückt werden."
        )
        return

    log.error(
        "Eingestellt ist TELEGRAM_CHAT_ID=%s. Der Bot hat zuletzt aus diesen "
        "Chats gehört: %s. Trage die passende Nummer als Secret ein.",
        chat_id,
        "; ".join(f"{cid} ({label})" for cid, label in chats),
    )


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
        _log_chat_hint(token, chat_id)
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
    message = f"Token gültig, Bot: @{name}"

    # Der Token allein sagt nichts über die Chat-ID. Beides zusammen zu prüfen
    # spart den Umweg über einen fehlgeschlagenen Versand.
    chat_id = env("TELEGRAM_CHAT_ID")
    chats = dict(discover_chats(token))
    if not chats:
        return True, (
            f"{message}\nChat-ID {chat_id} nicht überprüfbar: Der Bot hat noch "
            "nie eine Nachricht erhalten. Öffne den Chat mit @"
            f"{name} und drücke einmal Start."
        )
    if chat_id in chats:
        return True, f"{message}\nChat-ID {chat_id} bestätigt ({chats[chat_id]})."
    return False, (
        f"{message}\nChat-ID {chat_id} taucht in den letzten Nachrichten nicht "
        "auf. Der Bot hat zuletzt aus diesen Chats gehört: "
        + "; ".join(f"{cid} ({label})" for cid, label in sorted(chats.items()))
    )


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
    subject = (
        "Aktien-Screening: Lauf unvollständig"
        if result.degraded
        else f"Aktien-Screening: {len(result.candidates)} Treffer"
    )
    return send_all(
        subject=subject,
        body=build_summary(result),
        attachment=report_path,
    )
