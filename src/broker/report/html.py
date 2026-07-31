"""Erzeugt den HTML-Report und den JSON-Export."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from broker.macro.sensitivity import sector_label
from broker.screener import ScreeningResult

TEMPLATE_DIR = Path(__file__).parent

VERDICT_LABELS = {
    "zyklisch-guenstig": "zyklisch günstig",
    "strukturell-billig": "strukturell billig",
    "unklar": "unklar",
}


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "–"
    return f"{value:,.{digits}f}".replace(",", " ")


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "–"
    return f"{value * 100:+.{digits}f} %"


def _verdict_label(verdict: str) -> str:
    return VERDICT_LABELS.get(verdict, verdict)


def render_html(
    result: ScreeningResult,
    universe_label: str = "",
    appearances: dict[str, int] | None = None,
    journal_runs: int = 0,
) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("template.html")
    return template.render(
        candidates=result.candidates,
        regime=result.regime,
        stats=result.stats,
        universe_label=universe_label or "—",
        generated=datetime.now().strftime("%d.%m.%Y %H:%M"),
        appearances=appearances or {},
        journal_runs=journal_runs,
        fmt=_fmt,
        pct=_pct,
        sector_label=sector_label,
        verdict_label=_verdict_label,
    )


def write_report(
    result: ScreeningResult,
    out_dir: Path | str,
    universe_label: str = "",
    appearances: dict[str, int] | None = None,
    journal_runs: int = 0,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"screening-{datetime.now():%Y-%m-%d}.html"
    path.write_text(
        render_html(result, universe_label, appearances, journal_runs), encoding="utf-8"
    )
    return path


def write_json(result: ScreeningResult, out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"screening-{datetime.now():%Y-%m-%d}.json"
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "macro": {
            "live": result.regime.live,
            "summary": result.regime.summary,
            "rate_direction": result.regime.rate_direction,
            "curve_shape": result.regime.curve_shape,
            "inflation_trend": result.regime.inflation_trend,
            "growth_signal": result.regime.growth_signal,
            "risk_appetite": result.regime.risk_appetite,
            "sector_scores": {
                k: round(v, 1) for k, v in result.regime.sector_scores.items()
            },
        },
        "stats": {
            "universe_size": result.stats.universe_size,
            "fundamentals_ok": result.stats.fundamentals_ok,
            "passed_filters": result.stats.passed_filters,
            "scored": result.stats.scored,
            "error_count": len(result.stats.errors),
        },
        "candidates": [c.to_dict() for c in result.candidates],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path
