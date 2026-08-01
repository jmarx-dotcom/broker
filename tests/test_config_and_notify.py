"""Regressionstests für zwei Fehler aus dem ersten Produktivlauf.

1. Ein API-Key mit angehängtem Leerzeichen führte zu 400/401 bei FRED und
   Telegram — im Log maskiert und dadurch praktisch nicht diagnostizierbar.
2. Ein fehlgeschlagener Versand wurde als "kein Kanal konfiguriert" gemeldet
   und verdeckte damit genau den Fehler, den man sehen wollte.
"""

from __future__ import annotations

import pytest

from broker import notify as notify_module
from broker.config import Config, env
from broker.notify import (
    NotificationOutcome,
    check_telegram,
    email_configured,
    send_all,
    telegram_configured,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "FRED_API_KEY", "ANTHROPIC_API_KEY", "FMP_API_KEY", "BROKER_PROVIDER",
        "BROKER_LLM_MODEL", "BROKER_LLM_EFFORT", "BROKER_MAX_CANDIDATES",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_TO",
    ):
        monkeypatch.delenv(name, raising=False)


class TestEnvTrimming:
    def test_trailing_space_is_removed(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "abc123 ")
        assert env("FRED_API_KEY") == "abc123"

    def test_newline_and_tab_are_removed(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "\tabc123\n")
        assert env("FRED_API_KEY") == "abc123"

    def test_whitespace_only_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "   ")
        assert env("FRED_API_KEY") is None

    def test_missing_variable_is_none(self):
        assert env("FRED_API_KEY") is None

    def test_config_trims_every_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FRED_API_KEY", " fred ")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xyz\n")
        monkeypatch.setenv("BROKER_MAX_CANDIDATES", " 7 ")

        config = Config.from_env(dotenv=tmp_path / "nichtvorhanden.env")

        assert config.fred_api_key == "fred"
        assert config.anthropic_api_key == "sk-ant-xyz"
        assert config.max_candidates == 7
        assert config.macro_live is True
        assert config.llm_enabled is True

    def test_blank_key_does_not_enable_the_feature(self, monkeypatch, tmp_path):
        # Ein leeres GitHub Secret darf nicht als "eingerichtet" gelten.
        monkeypatch.setenv("FRED_API_KEY", "  ")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        config = Config.from_env(dotenv=tmp_path / "nichtvorhanden.env")

        assert config.fred_api_key is None
        assert config.macro_live is False
        assert config.llm_enabled is False


class TestChannelDetection:
    def test_telegram_needs_both_values(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        assert telegram_configured() is False  # Chat-ID fehlt noch

        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        assert telegram_configured() is True

    def test_telegram_ignores_whitespace_only_values(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", " ")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        assert telegram_configured() is False

    def test_email_needs_all_four_values(self, monkeypatch):
        for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
            monkeypatch.setenv(name, "x")
        assert email_configured() is False

        monkeypatch.setenv("SMTP_TO", "x@example.com")
        assert email_configured() is True


class TestNotificationOutcome:
    def test_failed_send_is_not_reported_as_unconfigured(self, monkeypatch, caplog):
        """Der eigentliche Bug: 401 wurde als 'nicht konfiguriert' gemeldet."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        monkeypatch.setattr(notify_module, "send_telegram", lambda text: False)

        with caplog.at_level("INFO"):
            outcome = send_all("Betreff", "Text")

        assert outcome.failed == ["Telegram"]
        assert outcome.sent == []
        assert outcome.configured is True
        assert "fehlgeschlagen" in caplog.text.lower()
        assert "kein benachrichtigungskanal" not in caplog.text.lower()

    def test_successful_send_is_reported(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        monkeypatch.setattr(notify_module, "send_telegram", lambda text: True)

        outcome = send_all("Betreff", "Text")

        assert outcome.sent == ["Telegram"]
        assert outcome.failed == []

    def test_nothing_configured_says_so(self, caplog):
        with caplog.at_level("INFO"):
            outcome = send_all("Betreff", "Text")

        assert outcome.configured is False
        assert "kein benachrichtigungskanal" in caplog.text.lower()

    def test_one_channel_can_fail_while_the_other_succeeds(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        for name, value in (
            ("SMTP_HOST", "smtp.example.com"), ("SMTP_USER", "u"),
            ("SMTP_PASSWORD", "p"), ("SMTP_TO", "to@example.com"),
        ):
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(notify_module, "send_telegram", lambda text: False)
        monkeypatch.setattr(notify_module, "send_email", lambda s, b, a=None: True)

        outcome = send_all("Betreff", "Text")

        assert outcome.sent == ["E-Mail"]
        assert outcome.failed == ["Telegram"]


class TestTelegramCheck:
    def test_reports_missing_token(self):
        ok, message = check_telegram()
        assert ok is False
        assert "TELEGRAM_BOT_TOKEN" in message

    def test_reports_missing_chat_id(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        ok, message = check_telegram()
        assert ok is False
        assert "TELEGRAM_CHAT_ID" in message

    def test_reports_rejected_token(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

        class Response:
            ok = False
            status_code = 401

        monkeypatch.setattr(notify_module.requests, "get", lambda *a, **k: Response())

        ok, message = check_telegram()
        assert ok is False
        assert "401" in message

    def test_reports_valid_token_with_bot_name(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

        class Response:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"result": {"username": "mein_screener_bot"}}

        monkeypatch.setattr(notify_module.requests, "get", lambda *a, **k: Response())

        ok, message = check_telegram()
        assert ok is True
        assert "mein_screener_bot" in message


class TestChatDiscovery:
    """Ein gültiger Token sagt nichts über die Chat-ID — das war der zweite
    Fehler in Folge, den erst ein fehlgeschlagener Versand offenbart hat."""

    @staticmethod
    def updates_response(updates):
        class Response:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"ok": True, "result": updates}

        return Response()

    def test_reads_chat_ids_and_names(self, monkeypatch):
        updates = [
            {
                "message": {
                    "chat": {
                        "id": 123456789,
                        "type": "private",
                        "first_name": "Jan",
                        "last_name": "Marx",
                    }
                }
            },
            {"message": {"chat": {"id": -100999, "type": "group", "title": "Redaktion"}}},
        ]
        monkeypatch.setattr(
            notify_module.requests, "get", lambda *a, **k: self.updates_response(updates)
        )
        assert notify_module.discover_chats("123:ABC") == [
            ("-100999", "Redaktion"),
            ("123456789", "Jan Marx"),
        ]

    def test_deduplicates_repeated_messages(self, monkeypatch):
        updates = [
            {"message": {"chat": {"id": 42, "type": "private", "first_name": "Jan"}}},
            {"message": {"chat": {"id": 42, "type": "private", "first_name": "Jan"}}},
        ]
        monkeypatch.setattr(
            notify_module.requests, "get", lambda *a, **k: self.updates_response(updates)
        )
        assert notify_module.discover_chats("123:ABC") == [("42", "Jan")]

    def test_reads_edited_messages_and_channel_posts(self, monkeypatch):
        updates = [
            {"edited_message": {"chat": {"id": 1, "type": "private", "username": "a"}}},
            {"channel_post": {"chat": {"id": 2, "type": "channel", "title": "Kanal"}}},
        ]
        monkeypatch.setattr(
            notify_module.requests, "get", lambda *a, **k: self.updates_response(updates)
        )
        assert notify_module.discover_chats("123:ABC") == [("1", "a"), ("2", "Kanal")]

    def test_survives_unexpected_payloads(self, monkeypatch):
        for payload in ([], [{}], ["quatsch"], [{"message": {}}], {"nicht": "liste"}):
            monkeypatch.setattr(
                notify_module.requests,
                "get",
                lambda *a, **k: self.updates_response(payload),
            )
            assert notify_module.discover_chats("123:ABC") == []

    def test_survives_network_error(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("kein Netz")

        monkeypatch.setattr(notify_module.requests, "get", boom)
        assert notify_module.discover_chats("123:ABC") == []

    def test_check_confirms_matching_chat_id(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

        def fake_get(url, *a, **k):
            if "getUpdates" in url:
                return self.updates_response(
                    [{"message": {"chat": {"id": 42, "type": "private", "first_name": "Jan"}}}]
                )
            return self.updates_response({"username": "screener_bot"})

        monkeypatch.setattr(notify_module.requests, "get", fake_get)
        ok, message = check_telegram()
        assert ok is True
        assert "bestätigt" in message and "Jan" in message

    def test_check_names_the_actual_chat_ids_on_mismatch(self, monkeypatch):
        """Der Kern: nicht 'ID falsch', sondern welche ID richtig wäre."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")

        def fake_get(url, *a, **k):
            if "getUpdates" in url:
                return self.updates_response(
                    [{"message": {"chat": {"id": 42, "type": "private", "first_name": "Jan"}}}]
                )
            return self.updates_response({"username": "screener_bot"})

        monkeypatch.setattr(notify_module.requests, "get", fake_get)
        ok, message = check_telegram()
        assert ok is False
        assert "42 (Jan)" in message

    def test_check_explains_silence(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

        def fake_get(url, *a, **k):
            if "getUpdates" in url:
                return self.updates_response([])
            return self.updates_response({"username": "screener_bot"})

        monkeypatch.setattr(notify_module.requests, "get", fake_get)
        ok, message = check_telegram()
        # Kein Fehler — nur nicht überprüfbar. Der Token ist ja in Ordnung.
        assert ok is True
        assert "Start" in message and "screener_bot" in message

    def test_failed_send_reports_available_chats(self, monkeypatch, caplog):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")

        class Rejected:
            ok = False
            status_code = 400

            @staticmethod
            def json():
                return {"description": "Bad Request: chat not found"}

        monkeypatch.setattr(notify_module.requests, "post", lambda *a, **k: Rejected())
        monkeypatch.setattr(
            notify_module.requests,
            "get",
            lambda *a, **k: self.updates_response(
                [{"message": {"chat": {"id": 42, "type": "private", "first_name": "Jan"}}}]
            ),
        )

        with caplog.at_level("ERROR"):
            assert notify_module.send_telegram("Test") is False
        assert "chat not found" in caplog.text
        assert "42 (Jan)" in caplog.text


class TestOutcomeDataclass:
    def test_defaults_are_independent_between_instances(self):
        first, second = NotificationOutcome(), NotificationOutcome()
        first.sent.append("Telegram")
        assert second.sent == []
