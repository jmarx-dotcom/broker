"""Übersetzt einen `broker doctor --json`-Bericht in Workflow-Ausgaben.

Warum ein eigenes Skript und kein Shell-Einzeiler: Die Texte für Pull Request
und Issue sind mehrzeilig und enthalten Backticks, Anführungszeichen und
Sternchen. In einem YAML-Block, der durch die Shell läuft, ist das eine
Fehlerquelle ohne Gegenwert. Hier lässt es sich außerdem testen.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PR_HINT = """
**Vor dem Merge kurz prüfen:** Ein Titel kann auch wegen einer Umbenennung
oder eines Börsenwechsels ausfallen — dann gehört statt der Entfernung das
neue Kürzel eingetragen. Die Prüfung unterscheidet das nicht.

Bei einer breiten Störung der Datenquelle entsteht dieser Pull Request gar
nicht erst: Fallen mehr als 20 % der Titel gleichzeitig aus, meldet die
Prüfung eine Störung statt Löschvorschläge.
""".strip()

ISSUE_HINT = """
Eine ausgefallene Makroreihe protokolliert im Fehlerfall die gültige
Codeliste des Datensatzes — der richtige Code steht also im Log des letzten
Laufs. Fehlt ein Abschlussfeld über das ganze Universum, braucht die
Bezeichnungsliste in `providers/yahoo.py` (`_statements`) einen weiteren
Namen.
""".strip()


def render(findings: list[dict], intro: str, hint: str) -> str:
    lines = [intro, ""]
    lines += [f"- **{f['check']}** — {f['message']}" for f in findings]
    lines += ["", hint, ""]
    return "\n".join(lines)


def main(path: str) -> int:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    findings = report.get("findings", [])
    fixable = [f for f in findings if f.get("fixable")]
    manual = [f for f in findings if not f.get("fixable")]

    if fixable:
        Path("pr_body.md").write_text(
            render(
                fixable,
                "Automatisch erstellt von der wöchentlichen Drift-Prüfung "
                "(`broker doctor`). Diese Titel liefern keine Kursdaten mehr "
                "und wurden aus den Universum-Dateien entfernt:",
                PR_HINT,
            ),
            encoding="utf-8",
        )
    if manual:
        Path("issue_body.md").write_text(
            render(
                manual,
                "Die wöchentliche Drift-Prüfung hat Befunde gefunden, die sich "
                "nicht mechanisch beheben lassen:",
                ISSUE_HINT,
            ),
            encoding="utf-8",
        )

    # Klartext ins Log, damit der Lauf ohne Artifact lesbar ist.
    print(report.get("summary", ""))
    for f in findings:
        mark = "reparierbar" if f.get("fixable") else "prüfen"
        print(f"  [{mark:>11}] {f['message']}")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"clean={str(report.get('clean', False)).lower()}\n")
            fh.write(f"has_fixable={str(bool(fixable)).lower()}\n")
            fh.write(f"has_manual={str(bool(manual)).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "doctor.json"))
