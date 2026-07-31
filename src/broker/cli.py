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

    # Ausgabe --------------------------------------------------------------
    print()
    if not result.candidates:
        print("Keine Treffer über der Score-Schwelle.")
    else:
        print(f"{'Score':>5}  {'Ticker':<12} {'KGV':>6} {'RSI':>5}  Titel")
        print("-" * 78)
        for c in result.candidates:
            pe = "–" if c.valuation.trailing_pe is None else f"{c.valuation.trailing_pe:.1f}"
            rsi = "–" if c.technical.rsi14 is None else f"{c.technical.rsi14:.0f}"
            print(f"{c.total_score:5.0f}  {c.ticker:<12} {pe:>6} {rsi:>5}  {c.name}")
    print()
    log.info(result.stats.summary())

    if args.report or args.json:
        from broker.report.html import write_json, write_report

        if args.report:
            path = write_report(result, config.out_dir, universe_label=args.universe)
            print(f"HTML-Report: {path}")
        if args.json:
            path = write_json(result, config.out_dir)
            print(f"JSON-Export: {path}")

    if args.notify:
        from broker.notify import notify

        channels = notify(result)
        if channels:
            print(f"Benachrichtigung verschickt über: {', '.join(channels)}")

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
    screen.set_defaults(func=cmd_screen)

    macro = sub.add_parser("macro", help="Aktuelles Makrobild anzeigen")
    macro.add_argument("--no-cache", action="store_true")
    macro.set_defaults(func=cmd_macro)

    universe = sub.add_parser("universe", help="Universum anzeigen")
    universe.add_argument("group", nargs="?", default="all")
    universe.add_argument("--list", action="store_true", help="Gruppen auflisten")
    universe.set_defaults(func=cmd_universe)

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
