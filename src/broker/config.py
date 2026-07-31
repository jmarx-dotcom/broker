"""Konfiguration aus Umgebungsvariablen und optionaler .env-Datei."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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
            provider=os.environ.get("BROKER_PROVIDER", "yfinance"),
            fmp_api_key=os.environ.get("FMP_API_KEY") or None,
            fred_api_key=os.environ.get("FRED_API_KEY") or None,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            llm_model=os.environ.get("BROKER_LLM_MODEL", "claude-opus-5"),
            llm_effort=os.environ.get("BROKER_LLM_EFFORT", "medium"),
            cache_dir=Path(os.environ.get("BROKER_CACHE_DIR", "cache")),
            out_dir=Path(os.environ.get("BROKER_OUT_DIR", "out")),
            max_candidates=int(os.environ.get("BROKER_MAX_CANDIDATES", "15")),
        )
