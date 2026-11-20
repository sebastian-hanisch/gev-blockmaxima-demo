"""Permalink (Begrenzen/Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import gvb_constants as C


def _bool(s):
    s = str(s).strip().lower()
    if s in ("1", "true", "ja", "yes"):
        return True
    if s in ("0", "false", "nein", "no"):
        return False
    raise ValueError(s)


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # bei Auswahlreglern: nur diese Werte sind gültig (Permalink rastet ein oder wird verworfen)


SETTING_SPECS = {
    "quelle_sel": SettingSpec("quelle", str, C.DEFAULTS["quelle_sel"], options=C.SOURCES),
    "b_sl": SettingSpec("b", int, C.DEFAULTS["b_sl"], options=C.BLOCKS),
    "verfahren_sel": SettingSpec("verfahren", str, C.DEFAULTS["verfahren_sel"], options=("ML", "L-Momente", "Gumbel")),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "xi_sl": SettingSpec("xi", float, C.DEFAULTS["xi_sl"], options=C.XI_OPTIONS),
    "m_sl": SettingSpec("m", int, C.DEFAULTS["m_sl"], options=C.M_OPTIONS),
    "runs_sl": SettingSpec("laeufe", int, C.DEFAULT_RUNS, C.RUNS_MIN, C.RUNS_MAX),
    "klammer_tg": SettingSpec("klammer", _bool, C.DEFAULTS["klammer_tg"]),
    "quelle4_sel": SettingSpec("quelle4", str, C.DEFAULTS["quelle4_sel"], options=C.SOURCES4),
}
PRESET_KEYS = {k: k for k in SETTING_SPECS}   # ein Preset setzt Regler unter ihrem Schlüssel; nicht genannte Regler gehen auf ihren Standardwert


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(spec, value):
    """Wert auf den gültigen Bereich bringen: Auswahlregler nehmen den nächsten gültigen Wert (Zahlen) oder verwerfen (Texte); Zahlenregler werden begrenzt."""
    if spec.options is not None:
        if value in spec.options:
            return value
        if isinstance(value, (int, float)) and all(isinstance(o, (int, float)) for o in spec.options):
            return min(spec.options, key=lambda o: abs(o - value))
        raise ValueError(value)
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = _snap(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen (Regler ohne Eintrag im Preset gelten als Standardwert), sonst None.
    Auch der Zufalls-Seed zählt: ein neuer Seed hebt die Markierung auf (Test: `tests/test_layout.py`)."""
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
