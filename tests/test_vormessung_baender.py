"""Die eingefrorenen Messungen (SplitMix64, `gvb_tables.json`) gegen die Zahlen der Vormessung V11 und V1 (numpy `default_rng`, anderer Zufallsstrom) als Band mit Standardfehler.

Die Zahlen der Vormessung sind Erwartungen: ein Band von ± 4 Standardfehlern beider Messungen (Standardfehler der SD: SD/√(2(R − 1)); der Anteile: √(p(1 − p)/R), mindestens 1/R;
des Mittels: SD/√R). Zellen, in denen die Messung außerhalb liegt, stehen in der README unter „Befunde und Korrekturen“, nicht hier. Zufallsanteile werden NIE auf das letzte Bit verglichen.

Spaltenkürzel der Vormessung (Plan, Befund B): ML-SD von ξ̂, ML-Rand, SD der L-Momente (mit Klammer), Anteil „Endpunkt unter dem Maximum“ der L-Momente (mit Klammer)."""

import math

import gvb_reference as REF

# (ξ, m): (ML-SD, ML-Rand, LM-SD, LM-Endpunkt<Max, LM-Mittel ξ̂); R = 1 000 in der Vormessung. Bei m = 10 steht die SD von ML nicht im Band (Ausreißer-Maß), dort gilt IQR/1,349 (Spalte 6).
V11_FIT = {
    (-0.9, 10): (None, 0.793, 0.272, 0.027, -0.757, None), (-0.9, 20): (0.135, 0.716, 0.186, 0.100, -0.826, None), (-0.9, 100): (0.068, 0.325, 0.096, 0.282, -0.884, None),
    (-0.6, 10): (None, 0.538, 0.304, 0.045, -0.581, None), (-0.6, 20): (0.213, 0.205, 0.214, 0.134, -0.606, None), (-0.6, 50): (0.110, 0.006, 0.130, 0.196, -0.594, None),
    (-0.3, 10): (None, 0.218, 0.305, 0.025, -0.310, None), (-0.3, 20): (0.218, 0.022, 0.180, 0.027, -0.310, None), (-0.3, 50): (0.103, 0.000, 0.107, 0.049, -0.303, None),
    (0.0, 10): (None, 0.085, 0.281, 0.005, -0.059, 0.394), (0.0, 20): (0.242, 0.002, 0.189, 0.002, -0.021, None), (0.0, 50): (0.115, 0.000, 0.109, 0.000, -0.006, None),
    (0.0, 100): (0.079, 0.000, 0.080, 0.001, -0.004, None),
    (0.2, 10): (None, 0.033, 0.292, 0.003, 0.121, 0.430), (0.2, 20): (0.262, 0.000, 0.205, 0.000, 0.151, None), (0.2, 50): (0.130, 0.000, 0.121, 0.000, 0.184, None),
    (0.5, 20): (0.296, 0.000, 0.220, 0.000, 0.387, None), (0.5, 50): (0.155, 0.000, 0.157, 0.000, 0.431, None),
}
R_V11 = 1000
# (Quelle, b): (Bias der Läufe, SD, lnRMSE, Gumbel-lnRMSE); R = 200 in der Vormessung
V11_BLOCK = {
    ("Normal", 6): (-0.073, 0.031, 0.079, 0.421), ("Normal", 30): (-0.029, 0.041, 0.050, 0.194), ("Normal", 60): (-0.021, 0.048, 0.052, 0.136), ("Normal", 360): (-0.001, 0.079, 0.079, 0.058),
    ("Exponential", 6): (0.142, 0.055, 0.152, 0.071), ("Exponential", 30): (0.011, 0.072, 0.073, 0.028), ("Exponential", 360): (-0.021, 0.110, 0.112, 0.056),
    ("Vehikel B", 6): (0.211, 0.098, 0.232, 1.390), ("Vehikel B", 12): (0.502, 0.145, 0.523, 1.169), ("Vehikel B", 30): (0.484, 0.187, 0.518, 0.886),
    ("Vehikel B", 60): (0.247, 0.207, 0.322, 0.703), ("Vehikel B", 360): (0.042, 0.303, 0.305, 0.375),
}
R_BLOCK_V11 = 200


def _se_anteil(p, r):
    return math.sqrt(max(p * (1 - p), 1.0 / r) / r)


def _band(a, se_a, b, se_b, k=4.0):
    return abs(a - b) <= k * math.hypot(se_a, se_b)


def test_tabelle_a_ml_und_l_momente_im_band_der_vormessung_v11():
    """Alle Zellen der Vormessung, die im Raster der eingefrorenen Tabelle liegen: ML-SD (außer m = 10), ML-Rand, L-Moment-SD und Endpunkt-Anteil (mit Klammer), L-Moment-Mittel von ξ̂."""
    n_vergleiche = 0
    for (xi, m), (ml_sd, rand, lm_sd, endp, lm_mean, iqr) in V11_FIT.items():
        c = REF.fit_cell(xi, m)
        R = c["laeufe"]
        if ml_sd is not None:
            assert _band(c["ml"]["xi"]["sd"], c["ml"]["xi"]["sd"] / math.sqrt(2 * (R - 1)), ml_sd, ml_sd / math.sqrt(2 * (R_V11 - 1))), (xi, m, "ML-SD", c["ml"]["xi"]["sd"], ml_sd)
            n_vergleiche += 1
        if iqr is not None:
            assert _band(c["ml"]["xi"]["iqr_sd"], 0.02, iqr, 0.02), (xi, m, "IQR", c["ml"]["xi"]["iqr_sd"], iqr)       # robuste Streuung, grobe Standardfehler 0,02
            n_vergleiche += 1
        assert _band(c["ml"]["rand"]["anteil"], _se_anteil(c["ml"]["rand"]["anteil"], R), rand, _se_anteil(rand, R_V11)), (xi, m, "ML-Rand", c["ml"]["rand"]["anteil"], rand)
        lm = c["lm_klammer"]
        assert _band(lm["xi"]["sd"], lm["xi"]["sd"] / math.sqrt(2 * (R - 1)), lm_sd, lm_sd / math.sqrt(2 * (R_V11 - 1))), (xi, m, "LM-SD", lm["xi"]["sd"], lm_sd)
        e = lm["endpunkt_unter_max"]["anteil"]
        assert _band(e, _se_anteil(e, R), endp, _se_anteil(endp, R_V11)), (xi, m, "LM-Endpunkt", e, endp)
        assert _band(lm["xi"]["mittel"], lm["xi"]["sd"] / math.sqrt(R), lm_mean, lm_sd / math.sqrt(R_V11)), (xi, m, "LM-Mittel", lm["xi"]["mittel"], lm_mean)
        assert c["ml"]["endpunkt_unter_max"]["anteil"] == 0.0                         # bei ML nie: die Likelihood verbietet es (alle 28 Zellen, unten gezählt)
        n_vergleiche += 5
    assert n_vergleiche == 18 * 5 + 13 + 2                                          # 18 Zellen mit je 5 Vergleichen, 13 ML-SD (nicht bei m = 10), 2 IQR (m = 10)


def test_tabelle_b_blockgroesse_im_band_der_vormessung_v11():
    """Bias der Läufe, SD von ln V, lnRMSE der freien GEV und lnRMSE von Gumbel gegen die 12 Zellen der Vormessung (R = 200) aus der Tabelle des Plans (R = 1 000 hier)."""
    n = 0
    for (source, b), (bias, sd, rmse, gum) in V11_BLOCK.items():
        c = REF.block_cell(source, b)
        R = c["ml"]["n_gueltig"]
        assert _band(c["ml"]["bias_ln"], c["ml"]["sd_ln"] / math.sqrt(R), bias, sd / math.sqrt(R_BLOCK_V11)), (source, b, "Bias", c["ml"]["bias_ln"], bias)
        assert _band(c["ml"]["sd_ln"], c["ml"]["sd_ln"] / math.sqrt(2 * R), sd, sd / math.sqrt(2 * R_BLOCK_V11)), (source, b, "SD", c["ml"]["sd_ln"], sd)
        assert _band(c["ml"]["ln_rmse"], c["ml"]["ln_rmse"] / math.sqrt(2 * R), rmse, rmse / math.sqrt(2 * R_BLOCK_V11)), (source, b, "lnRMSE", c["ml"]["ln_rmse"], rmse)
        assert _band(c["gumbel"]["ln_rmse"], c["gumbel"]["ln_rmse"] / math.sqrt(2 * R), gum, gum / math.sqrt(2 * R_BLOCK_V11)), (source, b, "Gumbel", c["gumbel"]["ln_rmse"], gum)
        n += 4
    assert n == 48


# (ξ, m, Verfahren-Anfang): (Bias, SD) der Fehler des 100er-Werts in σ aus V1 Abschnitt 5 (2 000 Läufe); nur Verfahren ohne Ausreißer-Maß
V1_POSITIONEN = {
    (0.0, 20, "Gumbel-ML"): (-0.183, 0.89), (0.0, 20, "Gumbel-Regression, Positionen nach Weibull"): (0.519, 1.17), (0.0, 20, "Gumbel-Regression, Gringorten"): (0.009, 1.08),
    (0.0, 50, "Gumbel-Regression, Positionen nach Weibull"): (0.296, 0.71), (0.0, 50, "Gumbel-Regression, Gringorten"): (0.016, 0.68),
    (0.2, 50, "Gumbel-ML"): (-2.358, 0.88), (-0.2, 50, "Gumbel-ML"): (1.267, 0.50),
}


def test_tabelle_c_handverfahren_im_band_der_vormessung_v1():
    n = 0
    for (xi, m, name), (bias, sd) in V1_POSITIONEN.items():
        r = next(z for z in REF.positions_rows() if z["xi"] == xi and z["m"] == m and z["verfahren"].startswith(name))
        assert _band(r["bias"], r["sd"] / math.sqrt(r["n_gueltig"]), bias, sd / math.sqrt(2000)), (xi, m, name, r["bias"], bias)
        assert _band(r["sd"], r["sd"] / math.sqrt(2 * r["n_gueltig"]), sd, sd / math.sqrt(4000)), (xi, m, name, r["sd"], sd)
        n += 2
    assert n == 14


def test_ml_gev_bei_m_20_sd_ist_ein_ausreisser_mass_median_und_iqr_sind_stabil():
    """V1: SD 2,98, Prüfer 9,02, V11: 8,00 (Zufallsströme!), hier eine weitere Zahl; gelesen werden Median (−0,3) und IQR/1,349 (rund 1,8): sie liegen in allen Messungen nahe beieinander."""
    r = next(z for z in REF.positions_rows() if z["xi"] == 0.0 and z["m"] == 20 and z["verfahren"] == "ML-GEV")
    assert -0.4 < r["median"] < -0.2 and 1.6 < r["iqr_sd"] < 2.1
    assert r["sd"] > r["iqr_sd"]                                                        # die SD liegt über dem robusten Maß (Ausreißer)
