"""Konfiguration aus Umgebungsvariablen und optionaler .env-Datei."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def env(name: str) -> str | None:
    """Liest eine Umgebungsvariable und entfernt umschließenden Leerraum.

    Beim Kopieren von API-Keys — besonders in Web-Formulare wie die GitHub
    Secrets — rutscht regelmäßig ein Leerzeichen oder Zeilenumbruch mit ans
    Ende. In einer URL wird daraus ein '+', und die Gegenstelle antwortet mit
    einem nichtssagenden 400 oder 401 auf einen Key, der maskiert im Log
    völlig korrekt aussieht. Deshalb wird hier grundsätzlich getrimmt.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimaler .env-Loader. Setzt nur Variablen, die noch nicht gesetzt sind."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Weights:
    """Gewichte des Gesamtscores. Summe wird beim Scoring normalisiert."""

    value: float = 0.40
    quality: float = 0.25
    technical: float = 0.25
    macro: float = 0.10


@dataclass(frozen=True)
class Thresholds:
    """Harte Filter. Ein Titel muss alle bestehen, um überhaupt bewertet zu werden."""

    min_market_cap: float = 3.0e8  # 300 Mio., filtert illiquide Nebenwerte
    min_avg_volume: float = 20_000  # Stück/Tag im 3-Monats-Schnitt
    max_trailing_pe: float = 40.0
    min_trailing_pe: float = 0.0  # negatives KGV = Verlust, fliegt raus
    min_history_days: int = 260  # ~1 Handelsjahr für Chart-Analyse
    min_score: float = 55.0  # ab hier landet ein Titel im Report


@dataclass(frozen=True)
class Config:
    provider: str = "yfinance"
    fmp_api_key: str | None = None
    fred_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5"
    llm_effort: str = "medium"
    cache_dir: Path = Path("cache")
    out_dir: Path = Path("out")
    max_candidates: int = 15
    weights: Weights = field(default_factory=Weights)
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def macro_live(self) -> bool:
        return bool(self.fred_api_key)

    @classmethod
    def from_env(cls, dotenv: str | Path = ".env") -> Config:
        load_dotenv(dotenv)
        return cls(
            provider=env("BROKER_PROVIDER") or "yfinance",
            fmp_api_key=env("FMP_API_KEY"),
            fred_api_key=env("FRED_API_KEY"),
            anthropic_api_key=env("ANTHROPIC_API_KEY"),
            llm_model=env("BROKER_LLM_MODEL") or "claude-opus-5",
            llm_effort=env("BROKER_LLM_EFFORT") or "medium",
            cache_dir=Path(env("BROKER_CACHE_DIR") or "cache"),
            out_dir=Path(env("BROKER_OUT_DIR") or "out"),
            max_candidates=int(env("BROKER_MAX_CANDIDATES") or "15"),
        )
