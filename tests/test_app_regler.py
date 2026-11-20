"""Jeder Wert jedes Reglers der Seitenleiste rechnet ohne Ausnahme; jedes Preset zeigt seine Verdikt-Zeile und die Kernfragen stehen vor den Experimenten; Permalink-Werte werden übernommen."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gvb_constants as C
import gvb_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in ("select_slider", "selectbox", "slider", "toggle", "number_input"):
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def test_jede_option_jedes_auswahlreglers_rechnet_ohne_ausnahme():
    at = _app()
    n = 0
    for key, spec in P.SETTING_SPECS.items():
        if spec.options is None:
            continue
        for wert in spec.options:
            _regler(at, key).set_value(wert).run()
            assert not at.exception, (key, wert, [str(e) for e in at.exception])
            n += 1
        _regler(at, key).set_value(spec.default).run()
    assert n == sum(len(s.options) for s in P.SETTING_SPECS.values() if s.options is not None)


def test_schalter_und_laeufe_und_zufalls_seed_rechnen():
    at = _app()
    for key, werte in (("klammer_tg", (False, True)), ("runs_sl", (C.RUNS_MIN, C.RUNS_MAX)), ("seed_input", (0, C.SEED_MAX))):
        for w in werte:
            _regler(at, key).set_value(w).run()
            assert not at.exception, (key, w)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_zeigt_seine_verdikt_zeile_und_die_kernfragen_stehen_davor(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    texte = [e.value for kind in ("success", "info", "warning") for e in getattr(at, kind)]
    assert any("der geschätzte Pegel ist das" in t or "keine Anpassung" in t for t in texte)
    assert any(m.value.startswith("## ❓ Kernfragen") for m in at.markdown)
    subs = [h.value for h in at.subheader]
    ordnung = [next(i for i, v in enumerate(subs) if f"Experiment {n}" in v) for n in (2, 3, 4, 5)]
    assert ordnung == sorted(ordnung) and any("Wo die Annahmen enden" in v for v in subs)
    kopf = " ".join(h.proto.body for h in at.get("html"))
    assert "st-key-kopf" in kopf


def test_permalink_wird_uebernommen_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["quelle"] = "Vehikel B"
    at.query_params["b"] = "13"             # kein gültiges b: rastet auf 12 ein
    at.query_params["xi"] = "-0.9"
    at.query_params["klammer"] = "nein"
    at.run()
    assert not at.exception
    assert at.session_state["quelle_sel"] == "Vehikel B" and at.session_state["b_sl"] == 12
    assert at.session_state["xi_sl"] == -0.9 and at.session_state["klammer_tg"] is False


def test_keine_obere_einstellung_ist_toter_regler_klammer_wirkt_in_experiment_2_und_3():
    """Der Schalter „Klammer“ ändert die L-Moment-Zahlen in Experiment 3 (ξ = −0,9, m = 20): ohne Klammer mehr unmögliche Anpassungen, die Anzeige ändert sich."""
    at = _app()
    for key, w in (("quelle_sel", "Exakte GEV"), ("xi_sl", -0.9), ("m_sl", 20), ("runs_sl", 100)):
        _regler(at, key).set_value(w).run()
    mit = [m.value for m in at.metric if "mit Klammer" in m.label][0]
    ohne = [m.value for m in at.metric if "ohne Klammer" in m.label][0]
    assert mit != ohne
    _regler(at, "klammer_tg").set_value(False).run()
    texte = " ".join(e.value for e in at.info) + " ".join(e.value for e in at.warning)
    assert "ohne Klammer (ξ̂ bis −60 zugelassen)" in texte
