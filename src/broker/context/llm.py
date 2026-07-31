"""LLM-Einordnung der Kandidaten über die Claude API.

Aufgabe des Modells ist bewusst eng gefasst: Es bekommt die fertig gerechneten
Kennzahlen, das Makrobild und die aktuellen Schlagzeilen und beantwortet eine
Frage, die sich aus Zahlen allein nicht beantworten lässt — *warum* ist dieser
Titel billig, und passt der Grund zum makroökonomischen Umfeld. Es rechnet
nichts nach und trifft keine Kaufentscheidung.

Zwei Dinge halten die Kosten niedrig:
  * Strukturierte Ausgabe per JSON-Schema, damit keine Nachbearbeitung nötig ist.
  * Prompt-Caching auf dem Makro-Block, der bei allen Kandidaten identisch ist.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from broker.macro.sensitivity import sector_label
from broker.models import Candidate, LLMContext, MacroRegime

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist ein nüchterner Aktienanalyst. Du bekommst für einen \
Einzeltitel fertig berechnete Kennzahlen, das aktuelle makroökonomische Umfeld \
und aktuelle Schlagzeilen.

Deine Aufgabe ist ausschließlich die Einordnung: Warum ist dieser Titel \
niedrig bewertet, und passt dieser Grund zum Umfeld?

Regeln:
- Rechne keine Kennzahlen nach und erfinde keine Zahlen. Nutze nur, was dir \
gegeben wurde.
- Unterscheide klar zwischen einem zyklischen Tief (das Geschäftsmodell ist \
intakt, der Zyklus oder die Stimmung drückt) und struktureller Billigkeit \
(das Geschäftsmodell erodiert). Das ist der Kern deiner Antwort.
- Wenn die Schlagzeilen dünn oder nichtssagend sind, sage das und setze die \
Konfidenz niedrig. Spekuliere nicht.
- Gib keine Kauf- oder Verkaufsempfehlung. Du lieferst Kontext für eine \
eigene Prüfung.
- Antworte auf Deutsch, sachlich und knapp. Keine Werbesprache."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cheap_because": {
            "type": "string",
            "description": (
                "Ein bis drei Sätze: Warum ist der Titel niedrig bewertet? "
                "Auf die konkreten Kennzahlen und Meldungen beziehen."
            ),
        },
        "macro_alignment": {
            "type": "string",
            "description": (
                "Ein bis zwei Sätze: Wie passt der Sektor zum aktuellen "
                "Zins-, Inflations- und Wachstumsumfeld?"
            ),
        },
        "key_risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Zwei bis vier konkrete Risiken, je ein kurzer Satz.",
        },
        "verdict": {
            "type": "string",
            "enum": ["zyklisch-guenstig", "strukturell-billig", "unklar"],
            "description": (
                "zyklisch-guenstig: Geschäftsmodell intakt, Bewertung gedrückt. "
                "strukturell-billig: Geschäft erodiert, Bewertung zu Recht "
                "niedrig. unklar: Datenlage reicht nicht."
            ),
        },
        "confidence": {
            "type": "string",
            "enum": ["hoch", "mittel", "niedrig"],
            "description": "Wie belastbar ist die Einordnung angesichts der Datenlage?",
        },
        "summary": {
            "type": "string",
            "description": "Ein einziger Satz als Kurzfassung für die Übersicht.",
        },
    },
    "required": [
        "cheap_because",
        "macro_alignment",
        "key_risks",
        "verdict",
        "confidence",
        "summary",
    ],
    "additionalProperties": False,
}


class LLMEnricher:
    """Reichert Kandidaten um eine LLM-Einordnung an."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-5",
        effort: str = "medium",
        max_tokens: int = 8000,
    ) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    # -- Prompt-Aufbau ----------------------------------------------------

    def _macro_block(self, regime: MacroRegime) -> str:
        """Der bei allen Kandidaten identische Teil — wird gecacht."""
        lines = [
            "AKTUELLES MAKROBILD",
            f"Zinsrichtung: {regime.rate_direction}",
            f"Renditekurve: {regime.curve_shape}",
            f"Inflationstrend: {regime.inflation_trend}",
            f"Wachstumssignal: {regime.growth_signal}",
            f"Risikoneigung: {regime.risk_appetite}",
        ]
        if not regime.live:
            lines.append(
                "Hinweis: Es liegen keine Live-Makrodaten vor. Behandle das "
                "Makroumfeld als unbekannt statt als neutral."
            )
        if regime.series:
            lines.append("")
            lines.append("Zeitreihen:")
            for series in regime.series.values():
                value = "n/a" if series.value is None else f"{series.value:.2f}"
                change = (
                    "" if series.change_3m is None else f" (3M: {series.change_3m:+.2f})"
                )
                lines.append(f"- {series.label}: {value}{series.unit}{change}")
        return "\n".join(lines)

    def _candidate_block(self, candidate: Candidate) -> str:
        data = candidate.to_dict()
        data.pop("llm", None)
        lines = [
            "TITEL",
            f"{candidate.name} ({candidate.ticker}), Sektor: "
            f"{sector_label(candidate.fundamentals.sector)}",
            "",
            "KENNZAHLEN (bereits berechnet, nicht nachrechnen):",
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
        ]

        if candidate.news:
            lines += ["", "AKTUELLE SCHLAGZEILEN:"]
            for item in candidate.news:
                stamp = f"{item.published}: " if item.published else ""
                source = f" [{item.publisher}]" if item.publisher else ""
                lines.append(f"- {stamp}{item.title}{source}")
        else:
            lines += ["", "AKTUELLE SCHLAGZEILEN: keine gefunden."]

        lines += [
            "",
            "Ordne diesen Titel gemäß deinen Regeln ein.",
        ]
        return "\n".join(lines)

    # -- API-Aufruf -------------------------------------------------------

    def enrich(self, candidate: Candidate, regime: MacroRegime) -> LLMContext:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        # Der Makro-Block ist über alle Kandidaten identisch.
                        # Der Cache-Marker hier spart bei jedem weiteren Titel
                        # rund 90 % der Kosten für diesen Präfix.
                        "type": "text",
                        "text": self._macro_block(regime),
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
                },
                messages=[
                    {"role": "user", "content": self._candidate_block(candidate)}
                ],
            )
        except Exception as exc:
            log.warning("LLM-Einordnung für %s fehlgeschlagen: %s", candidate.ticker, exc)
            return LLMContext(error=str(exc))

        # Sicherheits-Klassifikatoren können ablehnen; dann ist content leer
        # oder unvollständig. Erst prüfen, dann lesen.
        if response.stop_reason == "refusal":
            return LLMContext(error="Anfrage wurde vom Modell abgelehnt.")
        if response.stop_reason == "max_tokens":
            return LLMContext(error="Antwort war unvollständig (max_tokens erreicht).")

        text = next(
            (block.text for block in response.content if block.type == "text"), None
        )
        if not text:
            return LLMContext(error="Leere Antwort vom Modell.")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return LLMContext(error=f"Antwort war kein gültiges JSON: {exc}")

        return LLMContext(
            cheap_because=payload.get("cheap_because", ""),
            macro_alignment=payload.get("macro_alignment", ""),
            key_risks=list(payload.get("key_risks", [])),
            verdict=payload.get("verdict", "unklar"),
            confidence=payload.get("confidence", "mittel"),
            summary=payload.get("summary", ""),
        )

    def enrich_all(
        self, candidates: list[Candidate], regime: MacroRegime
    ) -> list[Candidate]:
        """Reichert alle Kandidaten an. Der erste Aufruf füllt den Prompt-Cache."""
        for candidate in candidates:
            candidate.llm = self.enrich(candidate, regime)
        return candidates
