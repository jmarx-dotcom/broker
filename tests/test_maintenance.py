"""Tests der Drift-Prüfung.

Der Maßstab sind die drei Fälle, die im Betrieb tatsächlich aufgetreten sind:
delistete Ticker (B4B.DE, COP.DE, SANT.DE), eine ausgefallene Eurostat-Reihe
(EA20 statt EA21) und umbenannte Yahoo-Bilanzzeilen. Die Prüfung muss genau
diese finden — und darf bei einer Störung der Datenquelle *nichts* vorschlagen.
"""

from __future__ import annotations

import pandas as pd
import pytest

from broker.maintenance import (
    OUTAGE_RATIO,
    STATEMENT_FIELDS,
    DoctorReport,
    Finding,
    check_macro_series,
    check_statement_fields,
    check_tickers,
    expected_macro_keys,
    remove_tickers,
    run_doctor,
)
from broker.models import Fundamentals, MacroSeries, PriceHistory


class FakeProvider:
    """Provider, bei dem sich Ausfälle gezielt einstellen lassen."""

    name = "fake"

    def __init__(self, dead: set[str] | None = None, fields: dict | None = None,
                 broken: set[str] | None = None) -> None:
        self.dead = dead or set()
        self.fields = fields if fields is not None else {n: 1.0 for n in STATEMENT_FIELDS}
        self.broken = broken or set()

    def history(self, ticker: str, period: str = "3y") -> PriceHistory:
        if ticker in self.broken:
            raise RuntimeError("Verbindung abgebrochen")
        frame = pd.DataFrame(
            {"Close": [] if ticker in self.dead else [100.0, 101.0],
             "Volume": [] if ticker in self.dead else [1e6, 1e6]},
            index=pd.to_datetime([] if ticker in self.dead else ["2026-07-30", "2026-07-31"]),
        )
        return PriceHistory(ticker=ticker, frame=frame)

    def fundamentals(self, ticker: str) -> Fundamentals:
        if ticker in self.broken:
            raise RuntimeError("Verbindung abgebrochen")
        return Fundamentals(ticker=ticker, **self.fields)

    def news(self, ticker: str, limit: int = 5):
        return []


class TestDeadTickers:
    def test_finds_the_delisted_ones(self, monkeypatch):
        """Der reale Fall vom 31. Juli: drei Titel ohne Kursdaten."""
        from broker import maintenance

        universe = ["SAP.DE", "BMW.DE", "B4B.DE", "COP.DE", "SANT.DE"] + [
            f"OK{i}.DE" for i in range(20)
        ]
        monkeypatch.setattr(
            maintenance, "load_universe",
            lambda group="all": [_entry(t) for t in universe],
        )

        provider = FakeProvider(dead={"B4B.DE", "COP.DE", "SANT.DE"})
        findings, checked, aborted = check_tickers(provider, workers=2)

        assert aborted is None
        assert checked == 25
        assert [f.subject for f in findings] == ["B4B.DE", "COP.DE", "SANT.DE"]
        assert all(f.fixable for f in findings)

    def test_exceptions_count_as_dead(self, monkeypatch):
        from broker import maintenance

        universe = ["A.DE"] + [f"OK{i}.DE" for i in range(20)]
        monkeypatch.setattr(
            maintenance, "load_universe",
            lambda group="all": [_entry(t) for t in universe],
        )
        findings, _, aborted = check_tickers(FakeProvider(broken={"A.DE"}), workers=2)
        assert aborted is None
        assert [f.subject for f in findings] == ["A.DE"]

    def test_outage_proposes_nothing(self, monkeypatch):
        """Die wichtigste Prüfung: Bei einer Störung nichts zur Löschung anbieten.

        Ein Vorschlag, halb Europa aus dem Universum zu entfernen, weil Yahoo
        zehn Minuten nicht antwortet, wäre das Gefährlichste am ganzen Werkzeug.
        """
        from broker import maintenance

        universe = [f"T{i}.DE" for i in range(50)]
        monkeypatch.setattr(
            maintenance, "load_universe",
            lambda group="all": [_entry(t) for t in universe],
        )
        provider = FakeProvider(dead={f"T{i}.DE" for i in range(30)})
        findings, checked, aborted = check_tickers(provider, workers=4)

        assert aborted == "Ausfallquote zu hoch"
        assert checked == 50
        assert len(findings) == 1
        assert findings[0].fixable is False
        assert "Störung" in findings[0].message
        assert "60%" in findings[0].message

    def test_threshold_is_where_it_says(self, monkeypatch):
        """Knapp unter der Schwelle wird noch repariert."""
        from broker import maintenance

        universe = [f"T{i}.DE" for i in range(100)]
        monkeypatch.setattr(
            maintenance, "load_universe",
            lambda group="all": [_entry(t) for t in universe],
        )
        below = int(100 * OUTAGE_RATIO)  # exakt 20 -> nicht "mehr als"
        provider = FakeProvider(dead={f"T{i}.DE" for i in range(below)})
        findings, _, aborted = check_tickers(provider, workers=4)
        assert aborted is None
        assert len(findings) == below

    def test_empty_universe(self, monkeypatch):
        from broker import maintenance

        monkeypatch.setattr(maintenance, "load_universe", lambda group="all": [])
        findings, checked, aborted = check_tickers(FakeProvider())
        assert findings == [] and checked == 0 and aborted == "Universum leer"


class TestMacroSeries:
    def test_finds_the_missing_series(self):
        """Der reale Fall: ez_unemployment fehlte, weil EA20 nicht mehr galt."""
        expected = {
            "ez_gdp": "Euroraum-BIP",
            "ez_unemployment": "Euroraum-Arbeitslosenquote",
            "ez_policy_rate": "EZB-Satz",
        }
        arrived = {
            "ez_gdp": _series("ez_gdp"),
            "ez_policy_rate": _series("ez_policy_rate"),
        }
        findings = check_macro_series(arrived, expected)
        assert [f.subject for f in findings] == ["ez_unemployment"]
        assert findings[0].fixable is False
        assert "Codeliste" in findings[0].message

    def test_all_present(self):
        expected = {"ez_gdp": "BIP"}
        assert check_macro_series({"ez_gdp": _series("ez_gdp")}, expected) == []

    def test_expected_keys_cover_the_european_sources(self):
        keys = expected_macro_keys(with_fred=False)
        for key in ("ez_gdp", "ez_unemployment", "ez_policy_rate",
                    "ez_yield_10y", "ez_yield_curve"):
            assert key in keys
        assert not any(k.startswith("us_") for k in keys)

    def test_fred_keys_only_with_a_key(self):
        with_fred = expected_macro_keys(with_fred=True)
        without = expected_macro_keys(with_fred=False)
        assert len(with_fred) > len(without)
        # Ohne FRED-Key darf das Fehlen der US-Reihen kein Befund sein.
        assert check_macro_series({}, without) != []
        assert all(f.subject.startswith("ez_") for f in check_macro_series({}, without))


class TestStatementFields:
    def test_finds_a_renamed_row(self, monkeypatch):
        """Fehlt ein Feld bei allen Titeln, hat Yahoo die Zeile umbenannt."""
        from broker import maintenance

        monkeypatch.setattr(
            maintenance, "load_universe",
            lambda group="all": [_entry(f"T{i}.DE") for i in range(40)],
        )
        fields = {n: 1.0 for n in STATEMENT_FIELDS}
        fields["ebit"] = None
        findings, usable = check_statement_fields(FakeProvider(fields=fields), workers=4)

        assert usable == 40
        assert [f.subject for f in findings] == ["ebit"]
        assert "providers/yahoo.py" in findings[0].message

    def test_partial_gaps_are_normal(self, monkeypatch):
        """Nebenwerte liefern oft keine Abschlüsse — das ist kein Drift."""
        from broker import maintenance

        entries = [_entry(f"T{i}.DE") for i in range(40)]
        monkeypatch.setattr(maintenance, "load_universe", lambda group="all": entries)

        class Patchy(FakeProvider):
            def fundamentals(self, ticker):
                # Nur jeder vierte Titel liefert das EBIT.
                value = 1.0 if ticker.endswith(("0.DE", "4.DE", "8.DE")) else None
                return Fundamentals(
                    ticker=ticker,
                    **{n: (value if n == "ebit" else 1.0) for n in STATEMENT_FIELDS},
                )

        findings, usable = check_statement_fields(Patchy(), workers=4)
        assert findings == []
        assert usable == 40

    def test_too_few_answers_proves_nothing(self, monkeypatch):
        from broker import maintenance

        entries = [_entry(f"T{i}.DE") for i in range(40)]
        monkeypatch.setattr(maintenance, "load_universe", lambda group="all": entries)
        # Fast alle Abrufe scheitern: daraus lässt sich nichts schließen.
        provider = FakeProvider(broken={f"T{i}.DE" for i in range(35)})
        findings, usable = check_statement_fields(provider, workers=4)
        assert findings == []
        assert usable == 5


class TestMechanicalFix:
    def test_removes_only_the_named_rows(self, tmp_path):
        path = tmp_path / "europe.csv"
        path.write_text(
            "ticker,index,region\n"
            "SAP.DE,DAX,DE\n"
            "B4B.DE,MDAX,DE\n"
            "BMW.DE,DAX,DE\n",
            encoding="utf-8",
        )
        removed = remove_tickers(["B4B.DE"], data_dir=tmp_path)

        assert removed == {"europe.csv": 1}
        assert path.read_text(encoding="utf-8") == (
            "ticker,index,region\nSAP.DE,DAX,DE\nBMW.DE,DAX,DE\n"
        )

    def test_leaves_the_header_and_untouched_files_alone(self, tmp_path):
        europe = tmp_path / "europe.csv"
        us = tmp_path / "us.csv"
        europe.write_text("ticker,index,region\nSAP.DE,DAX,DE\n", encoding="utf-8")
        us.write_text("ticker,index,region\nAAPL,SP500,US\n", encoding="utf-8")

        removed = remove_tickers(["NOPE.DE"], data_dir=tmp_path)
        assert removed == {}
        assert europe.read_text(encoding="utf-8").startswith("ticker,index,region\n")
        assert "AAPL" in us.read_text(encoding="utf-8")

    def test_does_not_match_on_substrings(self, tmp_path):
        """'SAP.DE' darf nicht 'SAP.DEX' mitentfernen."""
        path = tmp_path / "europe.csv"
        path.write_text(
            "ticker,index,region\nSAP.DE,DAX,DE\nSAP.DEX,DAX,DE\n", encoding="utf-8"
        )
        remove_tickers(["SAP.DE"], data_dir=tmp_path)
        assert "SAP.DEX" in path.read_text(encoding="utf-8")
        assert "SAP.DE,DAX" not in path.read_text(encoding="utf-8")

    def test_removes_across_both_files(self, tmp_path):
        (tmp_path / "europe.csv").write_text(
            "ticker,index,region\nA.DE,DAX,DE\n", encoding="utf-8")
        (tmp_path / "us.csv").write_text(
            "ticker,index,region\nB,SP500,US\n", encoding="utf-8")
        removed = remove_tickers(["A.DE", "B"], data_dir=tmp_path)
        assert removed == {"europe.csv": 1, "us.csv": 1}


class TestReport:
    def test_summary_counts_by_check(self):
        report = DoctorReport(findings=[
            Finding("Toter Ticker", "A.DE", "", fixable=True),
            Finding("Toter Ticker", "B.DE", "", fixable=True),
            Finding("Makroreihe", "ez_gdp", ""),
        ])
        assert report.summary() == "1× Makroreihe, 2× Toter Ticker"
        assert len(report.fixable) == 2
        assert report.clean is False

    def test_clean_report(self):
        report = DoctorReport(checked={"Ticker": 674})
        assert report.clean is True
        assert report.summary() == "Keine Abweichungen gefunden."

    def test_full_run_combines_all_three(self, monkeypatch):
        """Alle drei realen Driftfälle in einem Lauf."""
        from broker import maintenance

        universe = ["B4B.DE"] + [f"OK{i}.DE" for i in range(39)]
        monkeypatch.setattr(
            maintenance, "load_universe",
            lambda group="all": [_entry(t) for t in universe],
        )
        fields = {n: 1.0 for n in STATEMENT_FIELDS}
        fields["ebit"] = None
        provider = FakeProvider(dead={"B4B.DE"}, fields=fields)

        report = run_doctor(
            provider,
            series={"ez_gdp": _series("ez_gdp")},
            with_fred=False,
        )

        checks = {f.check for f in report.findings}
        assert checks == {"Toter Ticker", "Makroreihe", "Abschlussfeld"}
        assert [f.subject for f in report.fixable] == ["B4B.DE"]
        assert report.checked["Ticker"] == 40

    def test_skips_are_recorded(self, monkeypatch):
        from broker import maintenance

        monkeypatch.setattr(maintenance, "load_universe", lambda group="all": [])
        report = run_doctor(
            FakeProvider(),
            series=expected_macro_keys(with_fred=False),
            with_fred=False,
            skip_tickers=True,
            skip_statements=True,
        )
        assert "Ticker" in report.skipped
        assert "Abschlussfelder" in report.skipped
        assert "Ticker" not in report.checked


def _entry(ticker: str):
    from broker.universe import UniverseEntry

    return UniverseEntry(ticker=ticker, index="DAX", region="DE")


def _series(key: str) -> MacroSeries:
    return MacroSeries(key=key, label=key, value=1.0, unit="%")


class TestTriageScript:
    """Das Skript, das den Bericht in Pull-Request- und Issue-Texte übersetzt."""

    @staticmethod
    def load():
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / ".github/scripts/triage_doctor.py"
        spec = importlib.util.spec_from_file_location("triage_doctor", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def write_report(self, tmp_path, findings):
        import json

        path = tmp_path / "doctor.json"
        path.write_text(
            json.dumps({
                "clean": not findings,
                "summary": "Testbericht",
                "findings": findings,
            }),
            encoding="utf-8",
        )
        return path

    def test_splits_fixable_from_manual(self, tmp_path, monkeypatch, capsys):
        """Der reale Fall vom 31. Juli, alle drei Driftarten in einem Bericht."""
        module = self.load()
        report = self.write_report(tmp_path, [
            {"check": "Toter Ticker", "subject": "B4B.DE",
             "message": "B4B.DE liefert keine Kursdaten mehr.", "fixable": True},
            {"check": "Makroreihe", "subject": "ez_unemployment",
             "message": "Reihe kam nicht an.", "fixable": False},
            {"check": "Abschlussfeld", "subject": "ebit",
             "message": "Feld fehlt überall.", "fixable": False},
        ])
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))

        assert module.main(str(report)) == 0

        outputs = (tmp_path / "out.txt").read_text(encoding="utf-8")
        assert "has_fixable=true" in outputs
        assert "has_manual=true" in outputs
        assert "clean=false" in outputs

        pr = (tmp_path / "pr_body.md").read_text(encoding="utf-8")
        issue = (tmp_path / "issue_body.md").read_text(encoding="utf-8")
        # Jeder Befund landet in genau einem der beiden Texte.
        assert "B4B.DE" in pr and "B4B.DE" not in issue
        assert "ez_unemployment" not in pr
        assert "Reihe kam nicht an" in issue and "Feld fehlt überall" in issue
        assert "Vor dem Merge kurz prüfen" in pr

    def test_clean_report_writes_no_files(self, tmp_path, monkeypatch):
        module = self.load()
        report = self.write_report(tmp_path, [])
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))

        module.main(str(report))

        assert not (tmp_path / "pr_body.md").exists()
        assert not (tmp_path / "issue_body.md").exists()
        outputs = (tmp_path / "out.txt").read_text(encoding="utf-8")
        assert "clean=true" in outputs
        assert "has_fixable=false" in outputs and "has_manual=false" in outputs

    def test_outage_finding_opens_no_pull_request(self, tmp_path, monkeypatch):
        """Eine Störung ist nicht reparierbar — sie darf keinen PR auslösen."""
        module = self.load()
        report = self.write_report(tmp_path, [
            {"check": "Datenquelle", "subject": "Kurshistorie",
             "message": "412 von 674 Titeln ohne Kursdaten (61%). Störung.",
             "fixable": False},
        ])
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))

        module.main(str(report))

        assert not (tmp_path / "pr_body.md").exists()
        assert "has_fixable=false" in (tmp_path / "out.txt").read_text(encoding="utf-8")
        assert "Störung" in (tmp_path / "issue_body.md").read_text(encoding="utf-8")

    def test_runs_without_github_output(self, tmp_path, monkeypatch):
        """Lokal aufgerufen darf es nicht am fehlenden Actions-Umfeld scheitern."""
        module = self.load()
        report = self.write_report(tmp_path, [])
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        assert module.main(str(report)) == 0
