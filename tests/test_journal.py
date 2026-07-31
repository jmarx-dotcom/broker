"""Tests für das Journal und die Vorwärtsmessung."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from broker.journal import (
    HORIZONS,
    MIN_SAMPLE,
    Journal,
    JournalEntry,
    Observation,
    _deduplicate,
    _price_on_or_before,
    build_report,
    collect_observations,
    evaluate,
)
from broker.models import (
    Candidate,
    Fundamentals,
    LLMContext,
    QualityResult,
    TechnicalResult,
    ValuationResult,
)
from tests.conftest import make_history

BENCHMARKS = {"US": "^GSPC", "DE": "^GDAXI", "EU": "^STOXX50E"}


def make_candidate(
    ticker: str = "SAP.DE",
    score: float = 70.0,
    price: float = 100.0,
    verdict: str = "zyklisch-guenstig",
    setup: str = "Bodenbildung nach Rücksetzer",
) -> Candidate:
    return Candidate(
        ticker=ticker,
        fundamentals=Fundamentals(
            ticker=ticker, name=f"{ticker} AG", sector="Technology", currency="EUR"
        ),
        valuation=ValuationResult(score=65.0, trailing_pe=14.0, forward_pe=12.0),
        technical=TechnicalResult(score=60.0, price=price, rsi14=45.0, setup=setup),
        quality=QualityResult(score=72.0),
        macro_score=50.0,
        total_score=score,
        llm=LLMContext(verdict=verdict),
    )


def make_entry(
    ticker: str = "SAP.DE",
    day: str = "2026-01-15",
    score: float = 70.0,
    price: float = 100.0,
    verdict: str = "zyklisch-guenstig",
    setup: str = "Bodenbildung nach Rücksetzer",
    benchmark: str = "^GDAXI",
) -> JournalEntry:
    return JournalEntry(
        date=day, ticker=ticker, name=f"{ticker} AG", region="DE",
        benchmark=benchmark, currency="EUR", price=price, total_score=score,
        valuation_score=65.0, quality_score=72.0, technical_score=60.0,
        macro_score=50.0, llm_verdict=verdict, setup=setup, sector="Technology",
    )


class TestWriting:
    def test_appends_one_line_per_candidate(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        written = journal.append(
            [make_candidate("SAP.DE"), make_candidate("BMW.DE")], BENCHMARKS
        )

        assert written == 2
        assert len(journal.entries()) == 2

    def test_creates_parent_directory(self, tmp_path):
        journal = Journal(tmp_path / "tief" / "verschachtelt" / "history.jsonl")
        journal.append([make_candidate()], BENCHMARKS)
        assert journal.path.is_file()

    def test_second_run_appends_instead_of_overwriting(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        journal.append([make_candidate("SAP.DE")], BENCHMARKS, run_date=date(2026, 1, 1))
        journal.append([make_candidate("BMW.DE")], BENCHMARKS, run_date=date(2026, 1, 2))

        assert len(journal.entries()) == 2
        assert journal.run_count == 2

    def test_rerun_on_same_day_does_not_duplicate(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        day = date(2026, 1, 15)
        journal.append([make_candidate("SAP.DE")], BENCHMARKS, run_date=day)
        second = journal.append([make_candidate("SAP.DE")], BENCHMARKS, run_date=day)

        assert second == 0
        assert len(journal.entries()) == 1

    def test_candidate_without_price_is_skipped(self, tmp_path):
        candidate = make_candidate()
        candidate.technical.price = None
        journal = Journal(tmp_path / "history.jsonl")

        assert journal.append([candidate], BENCHMARKS) == 0

    def test_empty_result_writes_nothing(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        assert journal.append([], BENCHMARKS) == 0
        assert not journal.path.exists()

    def test_region_is_derived_from_ticker_suffix(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        journal.append(
            [make_candidate("SAP.DE"), make_candidate("AAPL"), make_candidate("MC.PA")],
            BENCHMARKS,
        )
        regions = {e.ticker: e.region for e in journal.entries()}

        assert regions == {"SAP.DE": "DE", "AAPL": "US", "MC.PA": "EU"}

    def test_benchmark_is_stored_per_region(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        journal.append([make_candidate("AAPL")], BENCHMARKS)
        assert journal.entries()[0].benchmark == "^GSPC"


class TestReading:
    def test_missing_file_yields_empty_list(self, tmp_path):
        assert Journal(tmp_path / "gibtsnicht.jsonl").entries() == []

    def test_corrupt_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "history.jsonl"
        good = json.dumps(make_entry().__dict__)
        path.write_text(f"{good}\nkaputt keine json\n{good}\n", encoding="utf-8")

        assert len(Journal(path).entries()) == 2

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "history.jsonl"
        good = json.dumps(make_entry().__dict__)
        path.write_text(f"\n{good}\n\n", encoding="utf-8")

        assert len(Journal(path).entries()) == 1


class TestAppearances:
    def test_counts_runs_per_ticker(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        for day in range(1, 6):
            journal.append(
                [make_candidate("SAP.DE")], BENCHMARKS, run_date=date(2026, 1, day)
            )
        journal.append([make_candidate("BMW.DE")], BENCHMARKS, run_date=date(2026, 1, 6))

        counts = journal.appearances()
        assert counts["SAP.DE"] == 5
        assert counts["BMW.DE"] == 1

    def test_lookback_limits_the_window(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        for day in range(1, 11):
            journal.append(
                [make_candidate("SAP.DE")], BENCHMARKS, run_date=date(2026, 1, day)
            )

        assert journal.appearances(lookback_runs=3)["SAP.DE"] == 3

    def test_empty_journal_returns_empty_mapping(self, tmp_path):
        assert Journal(tmp_path / "history.jsonl").appearances() == {}


class TestDeduplication:
    def test_keeps_only_first_mention_per_month(self):
        entries = [
            make_entry(day="2026-01-05"),
            make_entry(day="2026-01-20"),
            make_entry(day="2026-02-03"),
        ]
        result = _deduplicate(entries)

        assert len(result) == 2
        assert sorted(e.date for e in result) == ["2026-01-05", "2026-02-03"]

    def test_different_tickers_are_kept_separately(self):
        entries = [
            make_entry("SAP.DE", day="2026-01-05"),
            make_entry("BMW.DE", day="2026-01-06"),
        ]
        assert len(_deduplicate(entries)) == 2

    def test_thirty_mentions_of_one_move_count_once(self):
        # Der eigentliche Zweck: eine einzige Kursbewegung darf nicht als
        # dreißig unabhängige Beobachtungen in die Statistik gehen.
        entries = [make_entry(day=f"2026-01-{d:02d}") for d in range(1, 31)]
        assert len(_deduplicate(entries)) == 1


class TestPriceLookup:
    def test_finds_last_close_on_or_before_target(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-09"])
        close = pd.Series([10.0, 11.0, 12.0], index=index)

        assert _price_on_or_before(close, date(2026, 1, 6)) == 11.0
        assert _price_on_or_before(close, date(2026, 1, 8)) == 11.0  # Wochenende
        assert _price_on_or_before(close, date(2026, 1, 9)) == 12.0

    def test_returns_none_before_the_first_observation(self):
        close = pd.Series([10.0], index=pd.to_datetime(["2026-01-05"]))
        assert _price_on_or_before(close, date(2026, 1, 1)) is None

    def test_handles_timezone_aware_index(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06"]).tz_localize("UTC")
        close = pd.Series([10.0, 11.0], index=index)

        assert _price_on_or_before(close, date(2026, 1, 6)) == 11.0

    def test_empty_series_returns_none(self):
        assert _price_on_or_before(pd.Series(dtype=float), date(2026, 1, 1)) is None


class FakeProvider:
    """Liefert konstruierte Kursverläufe für die Auswertung."""

    name = "fake"

    def __init__(self, series: dict[str, pd.Series]):
        self._series = series

    def history(self, ticker: str, period: str = "3y"):
        if ticker not in self._series:
            raise RuntimeError(f"keine Daten für {ticker}")

        class Wrapper:
            close = self._series[ticker]

        return Wrapper()

    def fundamentals(self, ticker: str):
        raise NotImplementedError

    def news(self, ticker: str, limit: int = 5):
        return []


def constant_growth(start: float, daily: float, days: int = 500,
                    end: date = date(2026, 7, 1)) -> pd.Series:
    index = pd.date_range(end=pd.Timestamp(end), periods=days, freq="D")
    values = [start * (1 + daily) ** i for i in range(days)]
    return pd.Series(values, index=index)


class TestObservations:
    def test_entry_too_young_is_not_evaluated(self):
        today = date(2026, 7, 1)
        entries = [make_entry(day=(today - timedelta(days=10)).isoformat())]
        provider = FakeProvider({"SAP.DE": constant_growth(100, 0.001)})

        assert collect_observations(entries, provider, 30, today=today) == []

    def test_mature_entry_is_evaluated(self):
        today = date(2026, 7, 1)
        entries = [make_entry(day=(today - timedelta(days=60)).isoformat())]
        provider = FakeProvider(
            {
                "SAP.DE": constant_growth(100, 0.002),
                "^GDAXI": constant_growth(1000, 0.001),
            }
        )

        observations = collect_observations(entries, provider, 30, today=today)

        assert len(observations) == 1
        assert observations[0].own_return > 0
        assert observations[0].excess > 0  # steigt schneller als der Index

    def test_underperformer_has_negative_excess(self):
        today = date(2026, 7, 1)
        entries = [make_entry(day=(today - timedelta(days=60)).isoformat())]
        provider = FakeProvider(
            {
                "SAP.DE": constant_growth(100, -0.001),
                "^GDAXI": constant_growth(1000, 0.001),
            }
        )

        observations = collect_observations(entries, provider, 30, today=today)
        assert observations[0].excess < 0

    def test_window_is_fixed_not_open_ended(self):
        """Die Rendite muss über genau das Fenster laufen, nicht bis heute."""
        today = date(2026, 7, 1)
        entries = [make_entry(day=(today - timedelta(days=300)).isoformat())]
        provider = FakeProvider({"SAP.DE": constant_growth(100, 0.001, days=600)})

        thirty = collect_observations(entries, provider, 30, today=today)
        ninety = collect_observations(entries, provider, 91, today=today)

        # Gleicher Eintrag, längeres Fenster -> größere Rendite.
        assert ninety[0].own_return > thirty[0].own_return

    def test_missing_benchmark_leaves_excess_undefined(self):
        today = date(2026, 7, 1)
        entries = [make_entry(day=(today - timedelta(days=60)).isoformat())]
        provider = FakeProvider({"SAP.DE": constant_growth(100, 0.001)})

        observations = collect_observations(entries, provider, 30, today=today)
        assert observations[0].benchmark_return is None
        assert observations[0].excess is None

    def test_unknown_ticker_is_skipped(self):
        today = date(2026, 7, 1)
        entries = [make_entry(day=(today - timedelta(days=60)).isoformat())]
        assert collect_observations(entries, FakeProvider({}), 30, today=today) == []


class TestReport:
    @staticmethod
    def observations(count: int, excess: float, score: float = 72.0,
                     verdict: str = "zyklisch-guenstig") -> list[Observation]:
        return [
            Observation(
                entry=make_entry(ticker=f"T{i}.DE", score=score, verdict=verdict),
                own_return=excess + 0.01,
                benchmark_return=0.01,
            )
            for i in range(count)
        ]

    def test_small_group_is_not_reportable(self):
        report = build_report(self.observations(3, 0.05), "1M")
        top = next(b for b in report.by_score if b.label == "70+")

        assert top.sample == 3
        assert top.reportable is False

    def test_large_group_is_reportable(self):
        report = build_report(self.observations(MIN_SAMPLE, 0.05), "1M")
        top = next(b for b in report.by_score if b.label == "70+")

        assert top.reportable is True
        assert top.median_excess == pytest.approx(0.05, abs=1e-9)
        assert top.hit_rate == 1.0

    def test_hit_rate_counts_only_positive_excess(self):
        good = self.observations(6, 0.05)
        bad = self.observations(4, -0.05)
        for i, obs in enumerate(bad):
            obs.entry.ticker = f"B{i}.DE"

        report = build_report(good + bad, "1M")
        top = next(b for b in report.by_score if b.label == "70+")

        assert top.sample == 10
        assert top.hit_rate == 0.6

    def test_median_is_used_not_mean(self):
        # Ein einzelner Verdoppler darf die Gruppe nicht dominieren.
        observations = self.observations(10, 0.01)
        observations[0].own_return = 5.0

        report = build_report(observations, "1M")
        top = next(b for b in report.by_score if b.label == "70+")

        assert top.median_excess == pytest.approx(0.01, abs=1e-9)

    def test_scores_land_in_the_right_bucket(self):
        observations = (
            self.observations(2, 0.01, score=57.0)
            + self.observations(2, 0.01, score=62.0)
            + self.observations(2, 0.01, score=67.0)
            + self.observations(2, 0.01, score=75.0)
        )
        report = build_report(observations, "1M")

        assert {b.label: b.sample for b in report.by_score} == {
            "55–60": 2, "60–65": 2, "65–70": 2, "70+": 2
        }

    def test_verdict_groups_are_predeclared(self):
        report = build_report(self.observations(2, 0.01), "1M")
        assert [b.label for b in report.by_verdict] == [
            "zyklisch-guenstig", "strukturell-billig", "unklar"
        ]

    def test_setup_groups_come_from_the_data(self):
        observations = self.observations(2, 0.01)
        observations[0].entry.setup = "fallendes Messer"
        report = build_report(observations, "1M")

        assert {b.label for b in report.by_setup} == {
            "fallendes Messer", "Bodenbildung nach Rücksetzer"
        }

    def test_empty_report_is_not_reportable(self):
        report = build_report([], "1M")
        assert report.total_observations == 0
        assert report.reportable is False


class TestEvaluate:
    def test_produces_one_report_per_horizon(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        journal.append([make_candidate()], BENCHMARKS, run_date=date(2026, 1, 1))
        reports = evaluate(journal, FakeProvider({}), today=date(2026, 7, 1))

        assert [r.horizon for r in reports] == list(HORIZONS)

    def test_empty_journal_yields_empty_reports(self, tmp_path):
        reports = evaluate(Journal(tmp_path / "leer.jsonl"), FakeProvider({}))
        assert all(r.total_observations == 0 for r in reports)


class TestRoundTrip:
    def test_written_entry_survives_reading(self, tmp_path):
        journal = Journal(tmp_path / "history.jsonl")
        journal.append(
            [make_candidate("SAP.DE", score=71.5, price=123.45)],
            BENCHMARKS,
            run_date=date(2026, 3, 17),
        )
        entry = journal.entries()[0]

        assert entry.ticker == "SAP.DE"
        assert entry.total_score == 71.5
        assert entry.price == 123.45
        assert entry.recorded_on == date(2026, 3, 17)
        assert entry.llm_verdict == "zyklisch-guenstig"
        assert entry.setup == "Bodenbildung nach Rücksetzer"
