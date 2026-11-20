"""Eingefrorene Tabellen: Aufbau, Quelle und Stand je Teil, innere Konsistenz, und Neu-Rechnen als Band (nie auf das letzte Bit: Linux-CI, `log`/`pow` aus der C-Bibliothek)."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import gvb_constants as C
import gvb_evaluation as EV
import gvb_fit as F
import gvb_reference as REF
import gvb_scenario as SC
import gvb_sources as S

ROOT = Path(__file__).resolve().parent.parent


def test_aufbau_der_tabellen_quelle_und_stand_je_teil():
    t = REF.tables()
    assert set(t) == {"fit", "block", "positionen"}
    for key, laeufe in (("fit", 1000), ("block", 1000), ("positionen", 2000)):
        meta = t[key]["meta"]
        assert meta["stand"] == "2026-11-20" and "gvb_make_tables.py" in meta["quelle"] and meta["laeufe_je_zelle"] == laeufe and meta["zufalls_seed_basis"]
    cells = t["fit"]["daten"]
    assert {(c["xi"], c["m"]) for c in cells} == {(xi, m) for xi in C.XI_GRID for m in C.M_GRID} and len(cells) == 28
    assert set(t["block"]["daten"]) == set(C.SOURCES4)
    for source, d in t["block"]["daten"].items():
        assert d["wahrer_pegel"] == pytest.approx(S.true_level(source, C.Q_TARGET), rel=1e-12)
        assert [d["zellen"][str(b)]["m"] for b in C.BLOCKS] == [C.N_OBS // b for b in C.BLOCKS]


def test_zellen_haben_die_erwarteten_zaehlungen_und_grenzen():
    for c in REF.tables()["fit"]["daten"]:
        assert c["laeufe"] == 1000
        assert c["ml"]["endpunkt_unter_max"]["anteil"] == 0.0                                  # ML: nie, in allen 28 Zellen
        assert c["ml"]["xi"]["min"] >= -1.0 and 0.0 <= c["ml"]["rand"]["anteil"] <= 1.0
        assert c["lm_klammer"]["xi"]["min"] >= -0.999 - 1e-12 and c["lm_klammer"]["xi"]["max"] <= 0.99 + 1e-12
        assert c["lm_ohne"]["endpunkt_unter_max"]["anteil"] >= c["lm_klammer"]["endpunkt_unter_max"]["anteil"] - 1e-12      # ohne Klammer nie weniger unmögliche Anpassungen
    for source in C.SOURCES4:
        for b in C.BLOCKS:
            assert REF.block_cell(source, b)["ml"]["n_gueltig"] == 1000 and REF.block_cell(source, b)["gumbel"]["n_gueltig"] == 1000


def test_umkehrtabelle_der_mischung_ist_aktuell():
    spec = importlib.util.spec_from_file_location("gvb_make_tables", ROOT / "tools" / "gvb_make_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gvb_make_tables"] = mod
    spec.loader.exec_module(mod)
    neu = mod.make_inv()["lnsf"]
    alt = json.loads((ROOT / "gvb_mixture_inv.json").read_text(encoding="utf-8"))["lnsf"]
    assert len(alt) == len(neu)
    assert max(abs(a - b) for a, b in zip(alt, neu)) < 1e-9


def test_neu_rechnen_fit_zelle_als_band_xi_0_m_50():
    """200 neue Läufe (Zufalls-Seed 31) gegen die eingefrorene Zelle (1 000 Läufe): SD von ξ̂ für ML und L-Momente, Mittel, im Band von ± 4 Standardfehlern."""
    A = EV.fit_cell_runs(0.0, 50, range(200), 31)
    s = EV.cell_summary(A)
    ref = REF.fit_cell(0.0, 50)
    for key in ("ml", "lm_klammer"):
        n, r = s[key]["xi"], ref[key]["xi"]
        se_sd = math.hypot(n["sd"] / math.sqrt(2 * 199), r["sd"] / math.sqrt(2 * 999))
        assert abs(n["sd"] - r["sd"]) <= 4 * se_sd
        se_m = math.hypot(n["sd"] / math.sqrt(200), r["sd"] / math.sqrt(1000))
        assert abs(n["mittel"] - r["mittel"]) <= 4 * se_m


def test_neu_rechnen_rand_anteile_xi_minus_0_9_m_20_als_band():
    A = EV.fit_cell_runs(-0.9, 20, range(200), 32)
    s = EV.cell_summary(A)
    ref = REF.fit_cell(-0.9, 20)
    for key, val in (("rand", s["ml"]["rand"]), ):
        r = ref["ml"][key]
        assert abs(val["anteil"] - r["anteil"]) <= 4 * math.hypot(val["se"], r["se"])
    for key in ("lm_klammer", "lm_ohne"):
        n, r = s[key]["endpunkt_unter_max"], ref[key]["endpunkt_unter_max"]
        assert abs(n["anteil"] - r["anteil"]) <= 4 * math.hypot(n["se"], r["se"])
    assert s["ml"]["endpunkt_unter_max"]["anteil"] == 0.0


def test_neu_rechnen_blockgroesse_normal_b60_als_band():
    """60 neue Reihen (je N = 18 000) für Normal, b = 60: Bias und SD von ln V der ML gegen die eingefrorene Zelle (1 000 Läufe)."""
    ref = REF.block_cell("Normal", 60)["ml"]
    b, zt, pb = 60, S.true_level("Normal", C.Q_TARGET), S.block_exceedance(60, C.Q_TARGET)
    lv = []
    for r in range(60):
        v = SC.series_v(SC.rng_for(555, 7, r))
        lv.append(math.log(F.fit_ml(S.isf("Normal", SC.block_minima_of_v(v, b))).level(pb) / zt))
    lv = np.array(lv)
    assert abs(lv.mean() - ref["bias_ln"]) <= 4 * math.hypot(lv.std() / math.sqrt(60), ref["sd_ln"] / math.sqrt(1000))
    assert abs(lv.std() - ref["sd_ln"]) <= 4 * math.hypot(lv.std() / math.sqrt(120), ref["sd_ln"] / math.sqrt(2000))


def test_zeit_je_lauf_fuer_die_obergrenzen_der_regler():
    """Die Obergrenzen (Läufe ≤ 100, m ≤ 500) halten die Live-Rechnung klein: 100 Läufe bei m = 500 unter 20 s (grobe Obergrenze, kein Befund; Streamlit Cloud ist langsamer)."""
    import time
    t0 = time.perf_counter()
    EV.fit_cell_runs(0.0, 500, range(100), 3)
    assert time.perf_counter() - t0 < 20.0
