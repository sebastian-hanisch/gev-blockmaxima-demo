"""Eigenschaften (je 20 bis 50 Zufallsinstanzen, fester Zufalls-Seed) und Randfälle der Anpassungen und der Auswertung."""

import math

import numpy as np
import pytest

import gvb_constants as C
import gvb_evaluation as EV
import gvb_fit as F
import gvb_gev as G
import gvb_scenario as SC
import gvb_sources as S


def _x(xi, m, seed, idx=0):
    return SC.exact_gev_blocks(SC.rng_for(seed, idx), m, xi=xi)


# ---------------------------------------------------------------------------------------------------
# Eigenschaften
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["ML", "L-Momente", "Gumbel"])
def test_verschiebung_und_skalierung_aendern_mu_und_sigma_aber_nicht_xi(method):
    for i in range(20):
        x = _x((-0.3, 0.0, 0.3)[i % 3], 30, 40, i)
        f0 = F.fit(method, x)
        f1 = F.fit(method, x + 7.5)
        f2 = F.fit(method, 3.0 * x)
        assert f1.xi == pytest.approx(f0.xi, abs=2e-4) and f1.sigma == pytest.approx(f0.sigma, rel=2e-4) and f1.mu == pytest.approx(f0.mu + 7.5, abs=2e-3 * f0.sigma)
        assert f2.xi == pytest.approx(f0.xi, abs=2e-4) and f2.sigma == pytest.approx(3 * f0.sigma, rel=2e-4) and f2.mu == pytest.approx(3 * f0.mu, abs=2e-3 * f0.sigma)


@pytest.mark.parametrize("method", ["ML", "L-Momente", "Gumbel"])
def test_die_reihenfolge_der_maxima_aendert_nichts(method):
    for i in range(20):
        x = _x(0.1, 25, 41, i)
        y = x[::-1].copy()
        np.random.default_rng(i).shuffle(y)
        a, b = F.fit(method, x), F.fit(method, y)
        assert (a.mu, a.sigma, a.xi) == pytest.approx((b.mu, b.sigma, b.xi), rel=1e-6, abs=1e-6)


def test_pegel_waechst_mit_kleinerem_p():
    for xi in (-0.5, 0.0, 0.4):
        ps = np.array([0.5, 0.2, 0.1, 0.01, 1e-3, 1e-5])
        z = G.pegel(ps, 0.0, 1.0, xi)
        assert np.all(np.diff(z) > 0)


def test_ml_endpunkt_liegt_nie_unter_dem_maximum():
    for i in range(50):
        xi = (-0.9, -0.6, -0.3, 0.0, 0.3)[i % 5]
        x = _x(xi, 10 + 10 * (i % 4), 42, i)
        f = F.fit_ml(x)
        assert f.endpoint >= x.max() - 1e-9


def test_mehr_bloecke_verkleinern_die_streuung_von_xi_im_mittel_ueber_zufalls_seeds():
    def sd(m):
        return float(np.std([F.fit_ml(_x(0.0, m, 43, i)).xi for i in range(60)], ddof=1))
    s20, s50, s100 = sd(20), sd(50), sd(100)
    assert s20 > s50 > s100


def test_ln_rmse_quadrat_gleich_bias_quadrat_plus_sd_quadrat_in_jeder_zelle():
    import gvb_reference as REF
    n = 0
    for source in C.SOURCES4:
        for b in C.BLOCKS:
            for key in ("ml", "gumbel"):
                c = REF.block_cell(source, b)[key]
                assert c["ln_rmse"] ** 2 == pytest.approx(c["bias_ln"] ** 2 + c["sd_ln"] ** 2, rel=1e-9)
                n += 1
    assert n == 42


def test_gleiche_zufalls_seeds_gleiche_ausgabe():
    s = EV.Settings(quelle="Vehikel B", b=12, seed=77)
    a1, a2 = EV.analyse(s), EV.analyse(s)
    assert a1 == a2
    assert EV.analyse(EV.Settings(quelle="Vehikel B", b=12, seed=78)) != a1
    A1 = EV.fit_cell_runs(0.0, 20, range(5), 9)
    assert np.array_equal(A1, EV.fit_cell_runs(0.0, 20, range(5), 9), equal_nan=True)


def test_gumbel_regime_ml_kann_xi_nicht_schaetzen_aber_pegel_ist_die_gumbel_formel():
    x = _x(0.0, 40, 44)
    f = F.fit("Gumbel", x)
    assert f.xi == 0.0 and f.level(0.01) == pytest.approx(f.mu - f.sigma * math.log(-math.log(0.99)), rel=1e-12)


def test_ml_gumbel_nll_nie_besser_als_freie_gev():
    """Die freie GEV enthält das Gumbel-Modell (ξ = 0): ihre minimale nll ist nicht größer (kleine Toleranz für den Optimierer)."""
    for i in range(20):
        x = _x(0.0, 30, 45, i)
        g = F.fit("Gumbel", x)
        f = F.fit_ml(x)
        assert G.nll((f.mu, math.log(f.sigma), f.xi), x) <= G.nll((g.mu, math.log(g.sigma), 0.0), x) + 1e-6


# ---------------------------------------------------------------------------------------------------
# Randfälle
# ---------------------------------------------------------------------------------------------------
def test_m_gleich_3_ist_das_kleinste_m_der_l_momente():
    mu, sg, xi = G.fit_lmoments([1.0, 2.0, 6.0])
    assert math.isfinite(mu) and sg > 0 and -0.999 <= xi <= 0.99


@pytest.mark.parametrize("n", [0, 1, 2])
def test_zu_wenige_blockmaxima_geben_eine_meldung_statt_einer_zahl(n):
    x = np.arange(1.0, n + 1.0)
    with pytest.raises(G.FitError):
        G.fit_lmoments(x)
    if n < 2:
        with pytest.raises(G.FitError):
            F.fit_ml(x)
        with pytest.raises(G.FitError):
            F.gumbel_ml(x)


@pytest.mark.parametrize("method", ["ML", "L-Momente", "Gumbel"])
def test_alle_maxima_gleich_meldung(method):
    with pytest.raises(G.FitError, match="gleich"):
        F.fit(method, np.full(10, 3.0))


def test_xi_gleich_minus_eins_genau_liegt_am_rand_des_ml_bereichs():
    """ξ = −1 (Gleich-Fall): die Block-Maxima sind GEV(0, 1, −1); ML endet am Rand ξ = −1 (Optimierer), das ist kein Anpassungsbeispiel."""
    kleben = [F.fit_ml(_x(-1.0, 20, 46, i)).edge for i in range(30)]
    assert sum(kleben) >= 15
    f = F.fit_ml(_x(-1.0, 20, 46, 0))
    assert f.xi >= G.XI_MIN


def test_xi_unter_minus_eins_kein_ml_aber_die_app_rechnet_weiter():
    x = _x(-1.2, 30, 47)
    f = F.fit_ml(x)
    assert f.xi >= -1.0
    a = EV.analyse(EV.Settings(quelle="Exakte GEV", xi=-1.2, m=30))
    assert a.fits["ML"] is not None and a.ratios["ML"] > 0
    assert G.nll((0.0, 0.0, -1.2), x) == G.HUGE


def test_xi_gegen_null_gumbel_grenzfall_der_formeln():
    for p in (0.1, 0.01):
        g = float(G.pegel(p, 1.0, 2.0, 0.0))
        assert float(G.pegel(p, 1.0, 2.0, 1e-12)) == pytest.approx(g, abs=1e-9)                    # unter der Umschaltgrenze: Gumbel-Formel
        assert float(G.pegel(p, 1.0, 2.0, 2e-9)) == pytest.approx(g, abs=1e-5)                     # knapp über der Umschaltgrenze: GEV-Formel, stetig
        assert float(G.pegel(p, 1.0, 2.0, -2e-9)) == pytest.approx(g, abs=1e-5)
    assert float(G.gev_cdf(1.0, 0.0, 1.0, 0.0)) == pytest.approx(math.exp(-math.exp(-1.0)))
    # τ3 und L-Moment-Rückrechnung gehen bei ξ → 0 stetig über
    assert G.tau3_of_xi(1e-7) == pytest.approx(G.tau3_of_xi(0.0), abs=1e-6)


def test_p_gleich_0_und_1_und_wertebereich_des_pegels():
    assert float(G.pegel(0.0, 0.0, 1.0, 0.5)) == math.inf and float(G.pegel(0.0, 0.0, 1.0, 0.0)) == math.inf
    assert float(G.pegel(0.0, 0.0, 1.0, -0.5)) == pytest.approx(2.0)                      # Endpunkt μ − σ/ξ
    assert float(G.pegel(1.0, 0.0, 1.0, 0.0)) == -math.inf
    assert np.isfinite(G.pegel(np.array([1e-12, 0.5, 1 - 1e-12]), 0.0, 1.0, 0.2)).all()


def test_zahl_der_laeufe_1_gibt_meldung_statt_streuung():
    with pytest.raises(G.FitError, match="Zu wenige"):
        EV.live_cell(EV.Settings(xi=0.0, m=20, runs=1))
    s, _ = EV.live_cell(EV.Settings(xi=0.0, m=20, runs=2))
    assert s["laeufe"] == 2


def test_b_gleich_n_ein_block_meldung_statt_zahl():
    a = EV.analyse(EV.Settings(quelle="Normal", b=C.N_OBS))
    assert a.m == 1 and a.ratios == {} and set(a.errors) == {"ML", "L-Momente", "Gumbel"}
    kind, text = EV.verdict_fit(a, "ML", None)
    assert kind == "warning" and "keine Anpassung" in text and "mindestens 2" in text


def test_b_das_n_nicht_teilt_in_analyse():
    with pytest.raises(ValueError):
        EV.analyse(EV.Settings(quelle="Normal", b=7))


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.5])
def test_pareto_verteilung_mit_alpha_bis_1_hat_kein_mittel_aber_das_maximum_ist_erklaert(alpha):
    """α ≤ 1: kein Erwartungswert, das Block-Maximum und der Pegel sind trotzdem definiert (F^b), und die Anpassung rechnet (ξ = 1/α ≥ 1: ML mit offener oberer Grenze)."""
    q = C.Q_TARGET
    assert S.true_level("Pareto-Verteilung", q, alpha) == pytest.approx(q ** (-1 / alpha), rel=1e-14)
    v = SC.series_v(SC.rng_for(5, 9), 3600)
    x = S.isf("Pareto-Verteilung", SC.block_minima_of_v(v, 30), alpha)
    assert np.all(x >= 1.0) and F.fit_ml(x).xi > 0.3


def test_ungueltige_eingaben_der_quellen():
    with pytest.raises(ValueError):
        S.isf("Exakte GEV", np.array([0.5]))
    with pytest.raises(ValueError):
        S.true_level("Exakte GEV", 0.01)
    with pytest.raises(ValueError):
        F.fit("Hand", np.arange(5.0))
    assert np.all(np.isfinite(S.isf("Normal", np.array([1.0]))))                         # v = 1 (u = 0) wird auf 1 − 2^-53 begrenzt
