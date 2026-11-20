"""Claims: jede Zahl der README hat eine Herkunft. Die Zahlen der Messungen stehen als Bausteine in `tests/claims_fragmente.py` (aus den eingefrorenen Tabellen und den deterministischen Rechnungen berechnet);
hier wird geprüft, dass jeder Baustein wörtlich in der README steht, dass die „Zahl → Test“-Tabelle nur vorhandene Tests nennt und dass App-Text und README dieselben Texte haben."""

import ast
import re
from pathlib import Path

from claims_fragmente import build

import gvb_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_jeder_messbaustein_steht_woertlich_in_der_readme():
    fehlt = {k: v for k, v in build().items() if v not in README}
    assert not fehlt, f"in der README fehlen: {fehlt}"


def test_baustein_zaehlung_stimmt_mit_den_tabellen_ueberein():
    """Die Zählungen, die die README nennt, sind aus den Zellen gerechnet (nicht von Hand): hier die Kontrolle der Werte, die die README als Aussage mit Bedingung nennt."""
    f = build()
    assert f["lm_kleiner_m_le_20"] == "10 von 12" and f["lm_kleiner_m100"] == "0 von 6" and f["n_gumbel_better"] == "1 von 7" and f["e_gumbel_better"] == "7 von 7"
    assert f["n_best"] == "b = 30" and f["b_best"].startswith("b = 6 ") and f["pop_b12_pct"] == "67 %" and f["pop_14"].startswith("14 von 14")


def _test_funktionen():
    namen = {}
    for p in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        namen[p.name] = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    return namen


def test_zahl_test_tabelle_nennt_nur_vorhandene_tests():
    namen = _test_funktionen()
    refs = re.findall(r"`tests/(test_[a-z_]+\.py)::(test_[a-z0-9_]+)`", README)
    assert len(refs) >= 25, "die Tabelle „Zahl → Test“ ist zu kurz"
    for datei, name in refs:
        assert datei in namen and name in namen[datei], f"{datei}::{name} gibt es nicht"


def test_annahmen_und_vehikel_text_stehen_in_app_und_readme_gleich():
    for text in (T.ASSUMPTIONS.strip(), T.VEHIKEL_TEXT):
        assert text in README, text[:60]


def test_readme_ohne_verbotene_woerter_und_gedankenstriche_im_fliesstext():
    body = README.split("\n---\n")[0]
    for wort in ("Saatwert", "Schwanz", "trägt", "tragen", "TODO", "geplant", "Phase 0", "läuft lokal"):
        assert wort not in re.sub(r"<!--.*?-->", "", body, flags=re.S), wort
    ohne_code = re.sub(r"```.*?```", "", re.sub(r"<!--.*?-->", "", body, flags=re.S), flags=re.S)
    assert " — " not in ohne_code and "—" not in ohne_code
    assert re.search(r"\bPareto\b(?!-Verteilung)", ohne_code) is None
    assert re.search(r"\bWeibull\b(?!-Typ| \(|\))", re.sub(r"Positionen nach Weibull|nach Weibull", "", ohne_code)) is None


def test_pflichtabschnitte_der_readme():
    for kopf in ("## Warum dieses Problem", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Ehrliche Grenzen", "## Befunde und Korrekturen", "## Quellen", "## Tests",
                 "## Dateistruktur", "## Bewusst nicht umgesetzt", "## Lokal ausführen"):
        assert kopf in README, kopf
    assert "🚧" in README and "nicht gemessen" in README
    for muss in ("Zufalls-Seed", "Weibull-Typ", "Pareto-Verteilung", "Pegel", "Läufe", "Extremwertindex"):
        assert muss in README, muss
