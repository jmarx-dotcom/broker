"""Dateibasierter Tagescache.

Ein Screening-Lauf über ~900 Titel macht sonst bei jedem Neustart dieselben
tausend Requests. Der Cache ist bewusst simpel: ein Verzeichnis pro Tag, und
was von gestern ist, wird ignoriert.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class DayCache:
    def __init__(self, root: Path | str, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        return self.root / date.today().isoformat() / namespace / f"{digest}.pkl"

    def get_or_compute(self, namespace: str, key: str, compute: Callable[[], T]) -> T:
        if not self.enabled:
            return compute()

        path = self._path(namespace, key)
        if path.is_file():
            try:
                with path.open("rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:  # beschädigter Eintrag: neu holen
                log.debug("Cache-Eintrag %s unlesbar (%s), wird neu geholt", path, exc)

        value = compute()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as fh:
                pickle.dump(value, fh)
        except Exception as exc:
            log.debug("Cache-Eintrag %s nicht schreibbar: %s", path, exc)
        return value

    def purge_old(self, keep_days: int = 3) -> int:
        """Löscht Cache-Verzeichnisse älterer Tage. Gibt die Anzahl zurück."""
        if not self.root.is_dir():
            return 0
        dirs = sorted((d for d in self.root.iterdir() if d.is_dir()), reverse=True)
        removed = 0
        for stale in dirs[keep_days:]:
            for item in sorted(stale.rglob("*"), reverse=True):
                item.unlink() if item.is_file() else item.rmdir()
            stale.rmdir()
            removed += 1
        return removed


def cache_key(*parts: Any) -> str:
    return "|".join(str(p) for p in parts)
