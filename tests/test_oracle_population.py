"""Orakel des Populations-Bias (Quadratur über F^b) gegen eine Simulation und gegen die Max-Stabilität.

Der Populations-Bias ist ein Grenzwert m → ∞ nur für Quellen mit ξ_pop ≥ 0 (Exponential, Pareto-Verteilung, Fréchet, Vehikel B). Für die Normalverteilung gibt es keinen:
der Test prüft dort nur, dass das Ergebnis vom Abschneiden des Tails abhängt (Spannweite über 0,1)."""

import math

import numpy as np
import pytest

import gvb_constants as C
import gvb_fit as F
import gvb_population as PO
import gvb_reference as REF
import gvb_scenario as SC
import gvb_sources as S


def test_quadratur_gegen_simulation_exponential_und_vehikel_b_14_von_14_zellen():
    """Bias der 1 000 simulierten Läufe (eingefrorene Tabelle, SplitMix64) gegen den per Quadratur gerechneten Grenzwert: alle 14 Zellen (Exponential, Vehikel B; 7 Blockgrößen) innerhalb 0,03 in ln."""
    abw = []
    for source in ("Exponential", "Vehikel B"):
        for b in C.BLOCKS:
            sim = REF.block_cell(source, b)["ml"]["bias_ln"]
            pop = PO.population_bias(source, b)["ln_v"]
            abw.append(abs(sim - pop))
    assert len(abw) == 14 and max(abw) < 0.03, abw


def test_quadratur_gegen_simulation_mit_sehr_grossem_m_exponential_b24():
    """Eine einzige große Stichprobe (m = 20 000 Block-Maxima, direkt als Maximum von b = 24 gezogen) gegen die Quadratur, auf 0,03 in ln."""
    b = 24
    rng = SC.rng_for(31, b)
    u = np.maximum(SC.uniforms(rng, 20000), SC.U_MIN)
    x = S.isf("Exponential", -np.expm1(np.log(u) / b))
    f = F.fit_ml(x)
    sim = math.log(f.level(S.block_exceedance(b, C.Q_TARGET)) / S.true_level("Exponential", C.Q_TARGET))
    assert sim == pytest.approx(PO.population_bias("Exponential", b)["ln_v"], abs=0.03)


def test_exponential_b24_mit_dem_ziel_der_vormessung_1_gegen_V1():
    """Eigener Rechenweg gegen die Vormessung V1 (scipy, Ziel q = 1/(100·8 760)): Populations-Bias in Pegel-Einheiten +0,546 (Exponential, b = 24), innerhalb 0,01."""
    r = PO.population_bias("Exponential", 24, q=1.0 / (100 * 8760))
    assert r["pegel"] - r["wahr"] == pytest.approx(0.546, abs=0.01)


def test_exponential_grenzwert_konvergiert_im_logit_rand():
    """Exponential, b = 6: ab Logit-Rand S = 13,8 stimmt das Ergebnis auf 0,002 (bei +0,143); S = 6 schneidet zu viel Tail ab."""
    werte = {rand: PO.population_bias_logit("Exponential", 6, rand)[0] for rand in (6.0, 13.8, 20.7, 30.0)}
    hoch = [werte[13.8], werte[20.7], werte[30.0]]
    assert max(hoch) - min(hoch) < 0.002 and np.mean(hoch) == pytest.approx(0.143, abs=0.002)
    assert werte[6.0] < min(hoch) - 0.02


def test_normal_hat_keinen_grenzwert_das_ergebnis_haengt_am_abschneiden():
    werte = [PO.population_bias_logit("Normal", 6, rand)[0] for rand in (6.0, 9.2, 13.8, 20.7, 30.0)]
    assert max(werte) - min(werte) > 0.1
    assert werte == sorted(werte)                       # mit mehr Tail steigt der Bias monoton
    assert not S.survival_is_known_limit("Normal") and all(S.survival_is_known_limit(q) for q in ("Exponential", "Vehikel B", "Pareto-Verteilung", "Standard-Fréchet"))


def test_normal_quadratur_steigt_weiter_bei_groesserem_abschneiden_zunahme_wird_langsamer():
    """README Befund 8: S = 60 / 100 / 200 / 400 gibt ln V 0,217 / 0,264 / 0,311 / 0,344; die Zunahme je Verdopplung von S wird langsamer, ein Grenzwert ist damit nicht ausgeschlossen."""
    werte = {s: PO.population_bias_logit("Normal", 6, s)[0] for s in (30.0, 60.0, 100.0, 200.0, 400.0)}
    assert [round(werte[s], 3) for s in (60.0, 100.0, 200.0, 400.0)] == [0.217, 0.264, 0.311, 0.344]
    assert werte[200.0] - werte[100.0] > werte[400.0] - werte[200.0] > 0.0
    assert werte[60.0] - werte[30.0] > werte[100.0] - werte[60.0]


def test_normal_einzelne_grosse_stichproben_mittel_ueber_zufalls_seeds_schrumpft_mit_m():
    """README Befund 8: im Mittel über Zufalls-Seeds ln V −0,08 (m = 3 000, 30 Zufalls-Seeds) und −0,06 (m = 30 000, 15 Zufalls-Seeds); die Einzelwerte streuen mit SD 0,03 und 0,02 (Bänder, nicht auf die Stelle)."""
    k3 = [PO.normal_reference(6, 3000, seed=s)["ln_v"] for s in range(1, 31)]
    k30 = [PO.normal_reference(6, 30000, seed=s)["ln_v"] for s in range(1, 16)]
    assert -0.10 < np.mean(k3) < -0.065 and 0.015 < np.std(k3, ddof=1) < 0.04
    assert -0.08 < np.mean(k30) < -0.045 and 0.01 < np.std(k30, ddof=1) < 0.03
    assert np.mean(k30) > np.mean(k3)                  # der Betrag des Mittels schrumpft mit m


@pytest.mark.parametrize("b", [6, 60, 360])
def test_frechet_ist_max_stabil_kein_block_bias(b):
    """F^b der Standard-Fréchet ist GEV(μ = b, σ = b, ξ = 1) für jedes b: die Populations-ML findet es wieder, der Bias verschwindet."""
    r = PO.frechet_population(b)
    assert r["xi"] == pytest.approx(1.0, abs=2e-3) and r["mu_je_b"] == pytest.approx(1.0, abs=2e-3) and r["sigma_je_b"] == pytest.approx(1.0, abs=2e-3)
    assert abs(r["ln_v"]) < 0.002


def test_frechet_f_hoch_b_gleich_gev_exakt():
    """Exakte Max-Stabilität als Formel: (exp(−1/x))^b = GEV(μ = b, σ = b, ξ = 1) (die Quadratur-Probe oben ist die numerische Gegenprobe)."""
    import gvb_gev as G
    for b in (1, 6, 60, 360):
        for x in (0.5 * b, b, 5.0 * b):
            assert math.exp(-1.0 / x) ** b == pytest.approx(float(G.gev_cdf(x, b, b, 1.0)), abs=1e-14)


def test_xi_pop_der_quellen_mit_grenzwert_ist_nicht_negativ():
    """Die Quellen, für die der Populations-Bias als Grenzwert geführt wird, haben ξ_pop ≥ 0 (Exponential: nahe 0, höchstens −0,01; Vehikel B: über 0,15 bei jedem b)."""
    for b in C.BLOCKS:
        assert PO.population_bias("Exponential", b)["xi"] > -0.01
        assert PO.population_bias("Vehikel B", b)["xi"] > 0.15
