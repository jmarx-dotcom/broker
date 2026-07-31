"""Kommandozeile.

  broker screen --universe europe --report
  broker macro
  broker universe --list
"""

from __future__ import annotations

import argparse
import logging
import sys

from broker.config import Config
from broker.macro.fred import FredClient
from broker.macro.regime import build_regime, neutral_regime
from broker.macro.sensitivity import sector_label
from broker.models import MacroRegime
from broker.providers.cache import DayCache
from broker.providers.factory import get_provider
from broker.screener import Screener
from broker.universe import INDEX_GROUPS, load_universe

log = logging.getLogger("broker")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # yfinance ist im Normalbetrieb sehr gesprächig.
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _load_regime(config: Config, use_cache: bool) -> MacroRegime:
    if not config.macro_live:
        return neutral_regime(
            "Kein FRED_API_KEY gesetzt — der Makro-Teil ist neutral bewertet. "
            "Kostenlosen Key unter fredaccount.stlouisfed.org/apikeys anlegen."
        )
    client = FredClient(config.fred_api_key, cache=DayCache(config.cache_dir, use_cache))
    series = client.fetch_all()
    if not series:
        return neutral_regime("FRED lieferte keine Daten — Makro-Teil neutral bewertet.")
    return build_regime(series)


# -- Befehle ---------------------------------------------------------------


def cmd_screen(args: argparse.Namespace) -> int:
    config = Config.from_env()
    use_cache = not args.no_cache

    entries = load_universe(args.universe, extra_tickers=args.ticker)
    if not entries:
        log.error("Universum %r ist leer.", args.universe)
        return 1
    if args.limit:
        entries = entries[: args.limit]

    log.info("Universum: %s (%d Titel)", args.universe, len(entries))

    regime = _load_regime(config, use_cache)
    log.info("Makro: %s", regime.summary)

    provider = get_provider(config, use_cache=use_cache)
    screener = Screener(config, provider, workers=args.workers)
    result = screener.run(entries, regime)

    # LLM-Einordnung nur für die tatsächlichen Treffer — pro Titel ein Aufruf.
    if result.candidates and not args.no_llm:
        if config.llm_enabled:
            from broker.context.llm import LLMEnricher
            from broker.context.news import collect_news

            cache = DayCache(config.cache_dir, enabled=use_cache)
            log.info("Sammle Nachrichten für %d Treffer …", len(result.candidates))
            for candidate in result.candidates:
                candidate.news = collect_news(
                    provider,
                    candidate.ticker,
                    company_name=candidate.fundamentals.name,
                    cache=cache,
                )

            log.info("Hole LLM-Einordnung (%s) …", config.llm_model)
            enricher = LLMEnricher(
                api_key=config.anthropic_api_key or "",
                model=config.llm_model,
                effort=config.llm_effort,
            )
            enricher.enrich_all(result.candidates, regime)
        else:
            log.warning(
                "Kein ANTHROPIC_API_KEY gesetzt — Report ohne LLM-Einordnung."
            )

    # Journal ---------------------------------------------------------------
    # Vor der Ausgabe, damit die Beständigkeit den aktuellen Lauf einschließt.
    from broker.journal import Journal
    from broker.screener import BENCHMARKS

    journal = Journal(config.journal_path)
    appearances: dict[str, int] = {}
    if not args.no_journal:
        written = journal.append(result.candidates, BENCHMARKS)
        if written:
            log.info("%d Treffer im Journal festgehalten (%s).", written, journal.path)
        appearances = journal.appearances()

    # Ausgabe --------------------------------------------------------------
    print()
    if not result.candidates:
        print("Keine Treffer über der Score-Schwelle.")
    else:
        print(f"{'Score':>5}  {'Ticker':<12} {'KGV':>6} {'RSI':>5} {'Läufe':>6}  Titel")
        print("-" * 82)
        for c in result.candidates:
            pe = "–" if c.valuation.trailing_pe is None else f"{c.valuation.trailing_pe:.1f}"
            rsi = "–" if c.technical.rsi14 is None else f"{c.technical.rsi14:.0f}"
            seen = appearances.get(c.ticker, 0)
            seen_text = f"{seen}" if seen else "–"
            print(
                f"{c.total_score:5.0f}  {c.ticker:<12} {pe:>6} {rsi:>5} "
                f"{seen_text:>6}  {c.name}"
            )
    print()
    log.info(result.stats.summary())

    if args.report or args.json:
        from broker.report.html import write_json, write_report

        if args.report:
            path = write_report(
                result,
                config.out_dir,
                universe_label=args.universe,
                appearances=appearances,
                journal_runs=journal.run_count,
            )
            print(f"HTML-Report: {path}")
        if args.json:
            path = write_json(result, config.out_dir)
            print(f"JSON-Export: {path}")

    if args.notify:
        from broker.notify import notify

        outcome = notify(result)
        if outcome.sent:
            print(f"Benachrichtigung verschickt über: {', '.join(outcome.sent)}")
        if outcome.failed:
            print(f"Benachrichtigung fehlgeschlagen: {', '.join(outcome.failed)}")

    return 0


def cmd_macro(args: argparse.Namespace) -> int:
    config = Config.from_env()
    regime = _load_regime(config, use_cache=not args.no_cache)

    print(f"\n{regime.summary}\n")
    if regime.series:
        print(f"{'Reihe':<32} {'Wert':>10} {'3 Monate':>12}")
        print("-" * 58)
        for series in regime.series.values():
            value = "n/a" if series.value is None else f"{series.value:.2f}"
            change = "–" if series.change_3m is None else f"{series.change_3m:+.2f}"
            print(f"{series.label:<32} {value:>10} {change:>12}")

    if regime.sector_scores:
        print(f"\n{'Sektor':<24} {'Makro-Score':>12}")
        print("-" * 38)
        for sector, score in sorted(
            regime.sector_scores.items(), key=lambda kv: kv[1], reverse=True
        ):
            print(f"{sector_label(sector):<24} {score:>12.0f}")
    print()
    return 0


def cmd_universe(args: argparse.Namespace) -> int:
    if args.list:
        print("\nVerfügbare Gruppen:")
        for name, indices in INDEX_GROUPS.items():
            entries = load_universe(name)
            print(f"  {name:<10} {len(entries):>4} Titel   ({', '.join(indices)})")
        print()
        return 0

    entries = load_universe(args.group)
    for entry in entries:
        print(f"{entry.ticker:<12} {entry.index:<10} {entry.region}")
    print(f"\n{len(entries)} Titel", file=sys.stderr)
    return 0


def _print_buckets(title: str, buckets) -> None:
    from broker.journal import MIN_SAMPLE

    printable = [b for b in buckets if b.sample > 0]
    if not printable:
        return

    print(f"\n  {title}")
    print(f"  {'Gruppe':<28} {'n':>4} {'Median vs. Index':>18} {'Trefferquote':>13}")
    print("  " + "-" * 66)
    for bucket in printable:
        if not bucket.reportable:
            print(
                f"  {bucket.label:<28} {bucket.sample:>4} "
                f"{'zu wenig Daten':>18} {'':>13}"
            )
            continue
        excess = (
            "–" if bucket.median_excess is None else f"{bucket.median_excess * 100:+.1f} %"
        )
        hit = "–" if bucket.hit_rate is None else f"{bucket.hit_rate * 100:.0f} %"
        print(f"  {bucket.label:<28} {bucket.sample:>4} {excess:>18} {hit:>13}")
    print(f"  (Gruppen unter {MIN_SAMPLE} Beobachtungen werden nicht ausgewiesen.)")


def cmd_track(args: argparse.Namespace) -> int:
    """Wertet das Journal aus: Wie liefen die bisherigen Treffer?"""
    from broker.journal import HORIZONS, Journal, evaluate

    config = Config.from_env()
    journal = Journal(config.journal_path)
    entries = journal.entries()

    if not entries:
        print(
            f"\nNoch kein Journal unter {journal.path}.\n"
            "Es füllt sich mit jedem Screening-Lauf.\n"
        )
        return 0

    first = min(e.date for e in entries)
    print(
        f"\nJournal: {len(entries)} Nennungen aus {journal.run_count} Läufen "
        f"seit {first}"
    )
    print(f"Titel insgesamt: {len({e.ticker for e in entries})}")

    if args.list:
        counts = journal.appearances(lookback_runs=args.lookback)
        print(f"\nBeständigkeit (letzte {args.lookback} Läufe):")
        for ticker, count in sorted(counts.items(), key=lambda kv: -kv[1])[:25]:
            print(f"  {ticker:<12} {count:>3}×")
        print()
        return 0

    provider = get_provider(config, use_cache=not args.no_cache)
    reports = evaluate(journal, provider)

    any_output = False
    for report in reports:
        days = HORIZONS[report.horizon]
        if report.total_observations == 0:
            print(
                f"\n── {report.horizon} ({days} Tage) ── "
                "noch keine Einträge alt genug."
            )
            continue
        any_output = True
        print(f"\n── {report.horizon} ({days} Tage) ── "
              f"{report.total_observations} Beobachtungen")
        _print_buckets("Nach Score", report.by_score)
        _print_buckets("Nach LLM-Urteil", report.by_verdict)
        _print_buckets("Nach Chart-Setup", report.by_setup)

    if not any_output:
        print(
            "\nNoch nichts auswertbar. Das erste Fenster schließt 30 Tage nach\n"
            "der ersten Aufzeichnung."
        )
    else:
        print(
            "\nHinweis: Das ist kein Backtest, sondern eine Vorwärtsmessung ohne\n"
            "Rückschaufehler. Sie wird erst nach vielen Monaten aussagekräftig —\n"
            "bis dahin ist jede Zahl hier vor allem eins: eine kleine Stichprobe.\n"
        )
    return 0


def cmd_notify(args: argparse.Namespace) -> int:
    """Prüft die Benachrichtigungskanäle, ohne ein Screening zu rechnen."""
    from broker.notify import (
        check_telegram,
        email_configured,
        send_all,
        telegram_configured,
    )

    Config.from_env()  # lädt die .env-Datei

    print("\nEingerichtete Kanäle:")
    print(f"  Telegram: {'ja' if telegram_configured() else 'nein'}")
    print(f"  E-Mail:   {'ja' if email_configured() else 'nein'}")

    if telegram_configured():
        ok, message = check_telegram()
        print(f"\nTelegram-Token: {message}")
        if not ok:
            return 1

    if not args.send:
        print("\n(Mit --send wird zusätzlich eine Testnachricht verschickt.)\n")
        return 0

    outcome = send_all(
        subject="Aktien-Screening: Testnachricht",
        body=(
            "Testnachricht vom Aktien-Screener.\n\n"
            "Wenn du das liest, ist die Benachrichtigung korrekt eingerichtet."
        ),
    )
    if outcome.sent:
        print(f"\nVerschickt über: {', '.join(outcome.sent)}")
    if outcome.failed:
        print(f"Fehlgeschlagen: {', '.join(outcome.failed)}")
        return 1
    if not outcome.configured:
        print("\nKein Kanal eingerichtet — es wurde nichts verschickt.")
        return 1
    print()
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    config = Config.from_env()
    cache = DayCache(config.cache_dir)
    removed = cache.purge_old(keep_days=args.keep_days)
    print(f"{removed} alte Cache-Verzeichnisse gelöscht.")
    return 0


# -- Parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="broker",
        description="Screener für günstig bewertete Aktien: KGV, Chart und Makrokontext.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-Ausgaben")
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser("screen", help="Screening-Lauf starten")
    screen.add_argument(
        "--universe",
        default="europe",
        help=f"Gruppe oder Indexliste. Bekannt: {', '.join(INDEX_GROUPS)}",
    )
    screen.add_argument(
        "--ticker", action="append", help="Zusätzlicher Titel (mehrfach möglich)"
    )
    screen.add_argument("--limit", type=int, help="Nur die ersten N Titel prüfen")
    screen.add_argument("--workers", type=int, default=8, help="Parallele Abrufe")
    screen.add_argument("--report", action="store_true", help="HTML-Report schreiben")
    screen.add_argument("--json", action="store_true", help="JSON-Export schreiben")
    screen.add_argument("--notify", action="store_true", help="Benachrichtigung senden")
    screen.add_argument("--no-llm", action="store_true", help="Ohne LLM-Einordnung")
    screen.add_argument("--no-cache", action="store_true", help="Tagescache ignorieren")
    screen.add_argument(
        "--no-journal", action="store_true", help="Treffer nicht im Journal festhalten"
    )
    screen.set_defaults(func=cmd_screen)

    track = sub.add_parser("track", help="Journal auswerten: wie liefen die Treffer?")
    track.add_argument(
        "--list", action="store_true", help="Nur Beständigkeit je Titel auflisten"
    )
    track.add_argument(
        "--lookback", type=int, default=20, help="Läufe für die Beständigkeit"
    )
    track.add_argument("--no-cache", action="store_true")
    track.set_defaults(func=cmd_track)

    macro = sub.add_parser("macro", help="Aktuelles Makrobild anzeigen")
    macro.add_argument("--no-cache", action="store_true")
    macro.set_defaults(func=cmd_macro)

    universe = sub.add_parser("universe", help="Universum anzeigen")
    universe.add_argument("group", nargs="?", default="all")
    universe.add_argument("--list", action="store_true", help="Gruppen auflisten")
    universe.set_defaults(func=cmd_universe)

    notify_cmd = sub.add_parser(
        "notify", help="Benachrichtigungskanäle prüfen (ohne Screening)"
    )
    notify_cmd.add_argument(
        "--send", action="store_true", help="Testnachricht tatsächlich verschicken"
    )
    notify_cmd.set_defaults(func=cmd_notify)

    cache = sub.add_parser("cache", help="Cache aufräumen")
    cache.add_argument("--keep-days", type=int, default=3)
    cache.set_defaults(func=cmd_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nAbgebrochen.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
