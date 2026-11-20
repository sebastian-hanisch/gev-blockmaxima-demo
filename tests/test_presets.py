"""Presets: vollständig, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink begrenzt und rastet ein."""

import pytest

import gvb_constants as C
import gvb_evaluation as EV
import gvb_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_option_values():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key, value)
            elif spec.lo is not None:
                assert spec.lo <= value <= spec.hi


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {} and EV.settings_from_state(C.PRESETS[first]) == EV.Settings()


def test_setting_specs_cover_every_preset_key_and_the_defaults_agree():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, spec in P.SETTING_SPECS.items():
        if key in C.DEFAULTS:
            assert spec.default == C.DEFAULTS[key]
        if spec.options is not None:
            assert spec.default in spec.options


def test_presets_sind_verschieden_und_zeigen_unterschiedliche_szenarien():
    zustaende = []
    for pr in C.PRESETS.values():
        zustaende.append(tuple(pr.get(k, sp.default) for k, sp in P.SETTING_SPECS.items()))
    assert len(set(zustaende)) == len(zustaende)


@pytest.mark.parametrize("wert,erwartet", [("30", 30), ("31", 30), ("1000", 360), ("2", 6)])
def test_snap_rastet_blockgroesse_auf_gueltige_werte_ein(wert, erwartet):
    assert P._snap(P.SETTING_SPECS["b_sl"], int(wert)) == erwartet


def test_snap_begrenzt_zahlenregler_und_verwirft_unbekannte_texte():
    assert P._snap(P.SETTING_SPECS["runs_sl"], 1000) == C.RUNS_MAX and P._snap(P.SETTING_SPECS["runs_sl"], -3) == C.RUNS_MIN
    with pytest.raises(ValueError):
        P._snap(P.SETTING_SPECS["quelle_sel"], "Weibull")
    assert P._snap(P.SETTING_SPECS["xi_sl"], 0.09) == 0.0
    assert P._bool("ja") is True and P._bool("false") is False
    with pytest.raises(ValueError):
        P._bool("vielleicht")
