"""Verdikte als reine Funktionen: Schwellen und jede Zahl im Text an handgesetzten Werten; je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a, Punkt 7)."""

import math

import pytest

import gvb_constants as C
import gvb_evaluation as EV
from gvb_texts import de, pct


def _analysis(v_ml=1.0, source="Normal", b=60, m=300, xi_true=None):
    ratios = {"ML": v_ml, "L-Momente": v_ml, "Gumbel": 1.146}
    return EV.Analysis(source, b, m, 18000, (1.0,) * m, 0.01, 4.0, xi_true, {}, {}, ratios)


def _ref(median=0.975, q10=0.926, q90=1.040, gum=1.146):
    return {"ml": {"median": median, "q10": q10, "q90": q90, "n_gueltig": 1000}, "gumbel": {"median": gum, "q10": 1.0, "q90": 1.3, "n_gueltig": 1000}}


@pytest.mark.parametrize("v,kind", [(1.0, "success"), (1.05, "success"), (0.93, "success"), (1.2, "info"), (0.8, "info"), (1.5, "warning"), (0.5, "warning"), (2.0, "warning")])
def test_verdict_fit_art_nach_dem_abstand_von_1(v, kind):
    k, _ = EV.verdict_fit(_analysis(v), "ML", None)
    assert k == kind


def test_verdict_fit_nennt_jede_zahl_der_quelle():
    k, t = EV.verdict_fit(_analysis(0.938), "ML", _ref())
    assert "Normal, b = 60 (m = 300)" in t and "0,938-Fache" in t
    assert "Median 0,975" in t and "0,926/1,040" in t and "1000 Läufe" in t and "Gumbel mit ξ = 0 fest: Median 1,146" in t
    assert "kein Streuband" in t


def test_verdict_fit_ohne_referenz_und_ohne_anpassung():
    _, t = EV.verdict_fit(_analysis(1.1), "ML", None)
    assert "Läufe dieser Quelle" not in t
    a = EV.Analysis("Normal", 360, 1, 18000, (1.0,), 0.01, 4.0, None, {"ML": None}, {"ML": "Die ML-Anpassung braucht mindestens 2 Block-Maxima."}, {})
    k, t = EV.verdict_fit(a, "ML", None)
    assert k == "warning" and "mindestens 2" in t and "m = 1" in t


def test_verdict_fit_exakte_gev_nennt_xi():
    _, t = EV.verdict_fit(_analysis(1.0, source="Exakte GEV", b=0, m=20, xi_true=-0.9), "ML", None)
    assert "Exakte GEV mit ξ = −0,9 (m = 20)" in t and "wahrer Pegel der erzeugenden GEV" in t and "aus F^b" not in t
    _, t2 = EV.verdict_fit(_analysis(1.0), "ML", None)
    assert "wahrer Pegel aus F^b" in t2


def _live(sd_ml=0.12, sd_lm=0.11, rand=0.3, emp=0.2, emp_o=0.35, mx=7.0):
    xi = lambda sd: {"n": 50, "mittel": 0.0, "sd": sd, "median": 0.01, "iqr_sd": 0.1, "min": -1.0, "max": mx}
    sh = lambda p: {"anteil": p, "se": 0.05, "n": 50}
    return {"laeufe": 50, "ml": {"xi": xi(sd_ml), "rand": sh(rand), "endpunkt_unter_max": sh(0.0)}, "lm_klammer": {"xi": xi(sd_lm), "endpunkt_unter_max": sh(emp)}, "lm_ohne": {"xi": xi(sd_lm), "endpunkt_unter_max": sh(emp_o)}}


def test_verdict_xi_spread_zahlen_und_vergleich_nur_mit_eingefrorener_zelle():
    ref = {"laeufe": 1000, "ml": {"xi": {"sd": 0.115, "iqr_sd": 0.106}}, "lm_klammer": {"xi": {"sd": 0.106, "iqr_sd": 0.102}}, "lm_ohne": {"xi": {"sd": 0.2, "iqr_sd": 0.1}}}
    k, t = EV.verdict_xi_spread(_live(), ref, 0.0, 50, True)
    assert "ξ = 0,0, m = 50, 50 Läufe live" in t and "ML hat SD von ξ̂ 0,120" in t and "mit Klammer ξ̂ ≥ −0,999) 0,110" in t
    assert "Eingefroren (1000 Läufe): ML SD 0,115 (IQR/1,349 0,106), L-Momente 0,106" in t and "die kleinere SD hat dort die L-Momente" in t
    assert "Ausreißer" not in t
    _, t2 = EV.verdict_xi_spread(_live(), ref, 0.0, 50, False)
    assert "ohne Klammer" in t2 and "L-Momente 0,200" in t2 and "die kleinere SD hat dort ML" in t2
    _, t3 = EV.verdict_xi_spread(_live(), None, 0.0, 500, True)
    assert "keine eingefrorene Zelle" in t3 and "kleinere SD hat" not in t3


def test_verdict_xi_spread_bei_m_10_warnt_vor_der_sd():
    _, t = EV.verdict_xi_spread(_live(mx=7.3), None, 0.0, 10, True)
    assert "Ausreißer" not in t and "wenigen Läufen mit riesigem ξ̂" in t and "7,3" in t and "Median und IQR/1,349 lesen" in t


def test_verdict_edge_zahlen_klammer_und_spanne_aus_der_tabelle():
    k, t = EV.verdict_edge(_live(rand=0.72, emp=0.10, emp_o=0.26), -0.9, 20, True)
    assert k == "warning" and "72 %" in t and "10 %" in t and "mit der Klammer ξ̂ ≥ −0,999" in t and "bei ML nie" in t
    lo1, hi1, lo2, hi2 = EV.lm_endpoint_range()
    assert f"zwischen {pct(lo1)} und {pct(hi1)} mit Klammer" in t and f"({pct(lo2)} bis {pct(hi2)} ohne" in t and "diesem ξ-Raster" in t
    assert "Zufallsextreme" in t and "m ≤ 100) bei bis zu einem Drittel" in t and "m = 500 gibt es keine eingefrorene Zelle" in t
    _, t2 = EV.verdict_edge(_live(rand=0.72, emp=0.10, emp_o=0.26), -0.9, 20, False)
    assert "26 %" in t2 and "ohne Klammer (ξ̂ bis −60 zugelassen)" in t2
    k3, t3 = EV.verdict_edge(_live(rand=0.0, emp=0.0, emp_o=0.0), 0.0, 50, True)
    assert k3 == "info" and "Spanne" not in t3 and "Raster" not in t3
    _, t4 = EV.verdict_edge(_live(), -1.2, 20, True)
    assert "unbeschränkt" in t4


def _blockrow(b, m, ml_rmse, g_rmse, bias=0.0, sd=0.05):
    return {"b": b, "m": m, "ml": {"ln_rmse": ml_rmse, "bias_ln": bias, "sd_ln": sd, "n_gueltig": 1000}, "gumbel": {"ln_rmse": g_rmse, "bias_ln": 0.0, "sd_ln": 0.0, "n_gueltig": 1000}}


def test_verdict_blocksize_minimum_im_inneren_und_am_rand():
    rows = [_blockrow(6, 3000, 0.08, 0.4, -0.07, 0.03), _blockrow(30, 600, 0.05, 0.2), _blockrow(360, 50, 0.07, 0.06, -0.001, 0.07)]
    k, t = EV.verdict_blocksize("Normal", rows, None)
    assert "kleinster lnRMSE der freien GEV (ML) bei b = 30 (m = 600): 0,050" in t and "b = 6 hat Bias −0,070 bei SD 0,030" in t and "b = 360 Bias −0,001 bei SD 0,070" in t
    assert "1 von 3 Blockgrößen (kleinster 0,060 bei b = 360)" in t and "vorab richtig bekannt" in t and "Minimum liegt am Rand" not in t and "1000 Läufe" in t
    rand = [_blockrow(6, 3000, 0.23, 1.4), _blockrow(12, 1500, 0.52, 1.2), _blockrow(30, 600, 0.50, 0.9)]
    _, t2 = EV.verdict_blocksize("Vehikel B", rand, None)
    assert "globale Minimum liegt am Rand des gemessenen Bereichs (b = 6)" in t2 and "kleinere Blöcke sind nicht gemessen" in t2 and "0 von 3 Blockgrößen" in t2
    assert "lokales Minimum" not in t2 and "im Inneren gibt es" not in t2
    # Vehikel B: globales Minimum am Rand, daneben ein lokales im Inneren (b = 180), das der Text nennen muss
    vb = [_blockrow(b, 18000 // b, r, 1.0) for b, r in ((6, 0.236), (12, 0.530), (30, 0.523), (60, 0.331), (90, 0.281), (180, 0.274), (360, 0.297))]
    _, t3 = EV.verdict_blocksize("Vehikel B", vb, None)
    assert "Daneben hat die Kurve ein lokales Minimum bei b = 180 (lnRMSE 0,274, die Nachbarn b = 90 und b = 360 haben 0,281 und 0,297)" in t3 and "im Inneren gibt es" not in t3
    assert "lokales Minimum" not in EV.verdict_blocksize("Normal", rows, None)[1]
    assert EV.verdict_blocksize("Normal", [], None)[0] == "info"


def test_verdict_positions_aus_den_eingefrorenen_zeilen():
    import gvb_reference as REF
    k, t = EV.verdict_positions(REF.positions_rows())
    rows = REF.positions_rows()
    g = next(r for r in rows if r["xi"] == 0.2 and r["verfahren"].startswith("Gumbel-ML"))
    assert k == "info" and f"Bias {de(g['bias'], 2)} σ" in t and pct(g["anteil_unter_wahr"]) in t and "Positionen nach Weibull" in t and "Gringorten" in t and "2 000 Läufe" in t
    assert "Standardfehler rund 0,015 σ, von null nicht zu unterscheiden" in t and "ebenfalls m = 50, 100er-Wert" in t


# ---------------------------------------------------------------------------------------------------
# Je Preset die erwartete Verdikt-Zeile (kein Preset ohne Wirkung meldet eine Wirkung)
# ---------------------------------------------------------------------------------------------------
PRESET_ERWARTUNG = {
    "Normal, Blockgröße b = 60 (Voreinstellung)": ("Normal, b = 60 (m = 300)", "kein Streuband"),
    "Exponential: Gumbel passt (b = 30)": ("Exponential, b = 30 (m = 600)", "V im Median"),
    "Normal, kleiner Block (b = 6)": ("Normal, b = 6 (m = 3000)", "kein Streuband"),
    "Normal, Jahresblock (b = 360)": ("Normal, b = 360 (m = 50)", "kein Streuband"),
    "Vehikel B, b = 12: der Pegel kippt nach oben": ("Vehikel B, b = 12 (m = 1500)", "kein Streuband"),
    "Kleine Reihe, m = 10": ("Exakte GEV mit ξ = 0,0 (m = 10)", None),
    "ξ = −0,9: ML klebt am Rand": ("Exakte GEV mit ξ = −0,9 (m = 20)", None),
    "Fréchet exakt (max-stabil)": ("Standard-Fréchet, b = 60 (m = 300)", None),
}


def _settings_of(name):
    return EV.settings_from_state({**{k: v for k, v in C.PRESETS[name].items()}})


def test_jedes_preset_hat_eine_erwartete_verdikt_zeile():
    assert set(PRESET_ERWARTUNG) == set(C.PRESETS)
    for name, (wo, extra) in PRESET_ERWARTUNG.items():
        s = _settings_of(name)
        a = EV.analyse(s)
        ref = None
        if s.quelle in C.SOURCES4:
            import gvb_reference as REF
            ref = REF.block_cell(s.quelle, a.b)
        _, t = EV.verdict_fit(a, s.verfahren, ref)
        assert wo in t, (name, t)
        if extra:
            assert extra in t, (name, t)
        if s.quelle not in C.SOURCES4:
            assert "eingefroren" not in t, (name, t)       # keine Streuung über Läufe für Quellen ohne eingefrorenen Satz


def test_preset_aussagen_der_hilfetexte_gegen_die_tabelle():
    import gvb_population as PO
    import gvb_reference as REF
    # Exponential: Gumbel schlägt die freie GEV bei allen sieben Blockgrößen
    assert all(REF.block_cell("Exponential", b)["gumbel"]["ln_rmse"] < REF.block_cell("Exponential", b)["ml"]["ln_rmse"] for b in C.BLOCKS)
    # Normal b = 6: Block-Bias nach unten bei kleiner Streuung; b = 360: praktisch kein Bias, größere Streuung als bei b = 30
    n6, n30, n360 = (REF.block_cell("Normal", b)["ml"] for b in (6, 30, 360))
    assert n6["bias_ln"] < -0.05 and n6["sd_ln"] < n30["sd_ln"] < n360["sd_ln"] and abs(n360["bias_ln"]) < 0.02
    # Zahlen der Hilfetexte (gvb_constants.PRESET_HELP): −0,07 bei b = 6, −0,01 und SD 0,066 gegen 0,040 bei b = 360 gegen b = 30
    assert round(n6["bias_ln"], 2) == -0.07 and round(n360["bias_ln"], 2) == -0.01 and round(n360["sd_ln"], 3) == 0.066 and round(n30["sd_ln"], 3) == 0.040
    assert "−0,07 in ln bei m = 3 000" in C.PRESET_HELP["Normal, kleiner Block (b = 6)"] and "SD 0,066 gegen 0,040 bei b = 30" in C.PRESET_HELP["Normal, Jahresblock (b = 360)"]
    # Vehikel B, b = 12: +67 % im Grenzwert m → ∞
    assert 100 * (math.exp(PO.population_bias("Vehikel B", 12)["ln_v"]) - 1) == pytest.approx(67, abs=0.5)
