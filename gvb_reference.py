"""Lädt die eingefrorenen Messungen (`gvb_tables.json`, erzeugt von `tools/gvb_make_tables.py`, Quelle und Stand je Teil im Feld „meta“).

Die schweren Zufallsmessungen (1 000 Läufe je Zelle, 1 000 Läufe je Blockgröße, 2 000 Läufe der Handverfahren) laufen nicht live; die App zeigt sie neben den live gerechneten Läufen."""

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def tables():
    return json.loads((HERE / "gvb_tables.json").read_text(encoding="utf-8"))


def fit_cell(xi, m):
    """Eingefrorene Zelle (ξ, m) der Tabelle A oder None, wenn (ξ, m) nicht im Raster liegt."""
    for c in tables()["fit"]["daten"]:
        if abs(c["xi"] - xi) < 1e-9 and c["m"] == m:
            return c
    return None


def block_cell(source, b):
    """Eingefrorene Zelle (Quelle, b) der Tabelle B oder None."""
    d = tables()["block"]["daten"].get(source)
    return None if d is None else d["zellen"].get(str(b))


def true_level_frozen(source):
    d = tables()["block"]["daten"].get(source)
    return None if d is None else d["wahrer_pegel"]


def positions_rows():
    return tables()["positionen"]["daten"]


def meta(part):
    return tables()[part]["meta"]
