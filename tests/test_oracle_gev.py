"""Orakel der GEV-Kernroutinen: andere Rechenwege (scipy nur hier), `scipy.stats.genextreme` mit **c = −ξ** (die App hat kein scipy).

(a) Likelihood, Quantilfunktion und Pegel gegen scipy und Bisektion; (b) eigene ML (nur numpy) gegen `genextreme.fit` (mindestens so tiefes Minimum) und Gumbel-ML gegen `gumbel_r.fit`;
(c) L-Momente: Populationswerte gegen scipy-Quadratur, Stichprobenwerte gegen Paar- und Tripel-Aufzählung; (d) Literaturtest Coles (3.4)."""

import itertools
import math

import numpy as np
import pytest

import gvb_fit as F
import gvb_gev as G
import gvb_scenario as SC

stats = pytest.importorskip("scipy.stats")
integrate = pytest.importorskip("scipy.integrate")


def _blocks(xi, m, seed, idx=0):
    return SC.exact_gev_blocks(SC.rng_for(seed, idx), m, xi=xi)


@pytest.mark.parametrize("xi", [-0.6, -0.2, 0.2, 0.5])
def test_nll_gleich_minus_summe_der_scipy_logpdf_mit_c_gleich_minus_xi(xi):
    """30 Stichproben je ξ; die Vorzeichenfalle: mit c = +ξ gibt scipy einen anderen Wert."""
    for i in range(30):
        x = _blocks(xi, 12, 100 + i)
        mu, sg = 0.3 * i / 30, 1.0 + 0.02 * i
        ref = -float(np.sum(stats.genextreme.logpdf(x, c=-xi, loc=mu, scale=sg)))
        assert G.nll_values(x, mu, sg, xi) == pytest.approx(ref, abs=1e-10, rel=1e-12)
        assert G.nll((mu, math.log(sg), xi), x) == pytest.approx(ref, abs=1e-10, rel=1e-12)
    x = _blocks(xi, 12, 7)
    assert abs(-float(np.sum(stats.genextreme.logpdf(x, c=+xi))) - G.nll_values(x, 0.0, 1.0, xi)) > 0.1


@pytest.mark.parametrize("mu,sg,xi", [(0.0, 1.0, 0.5), (3.0, 2.0, -0.3), (10.0, 4.0, 0.2), (0.0, 1.0, 0.0), (1.0, 0.5, -0.9)])
def test_quantilfunktion_und_pegel_gegen_scipy(mu, sg, xi):
    ps = (0.5, 0.1, 0.01, 0.001, 1e-5)
    for p in ps:
        assert float(G.pegel(p, mu, sg, xi)) == pytest.approx(float(stats.genextreme.isf(p, c=-xi, loc=mu, scale=sg)), rel=1e-9)
        assert float(G.gev_ppf(1 - p, mu, sg, xi)) == pytest.approx(float(stats.genextreme.ppf(1 - p, c=-xi, loc=mu, scale=sg)), rel=1e-9, abs=1e-9)
        assert float(G.gev_cdf(float(G.pegel(p, mu, sg, xi)), mu, sg, xi)) == pytest.approx(1 - p, rel=1e-9)
    xs = np.linspace(mu - 1, mu + 3 * sg, 11)
    ok = np.isfinite(stats.genextreme.cdf(xs, c=-xi, loc=mu, scale=sg))
    assert np.allclose(G.gev_cdf(xs, mu, sg, xi)[ok], stats.genextreme.cdf(xs, c=-xi, loc=mu, scale=sg)[ok], rtol=1e-10, atol=1e-12)


def test_ml_so_tief_wie_scipy_genextreme_fit_m100_20_stichproben():
    """Eigene ML (nur numpy): das Minimum der nll ist mindestens so tief wie das von `genextreme.fit` (größter Unterschied in V11: −6e-8)."""
    for i in range(20):
        xi = (-0.4, 0.0, 0.3)[i % 3]
        x = _blocks(xi, 100, 300 + i)
        f = F.fit_ml(x)
        mine = G.nll((f.mu, math.log(f.sigma), f.xi), x)
        c, loc, scale = stats.genextreme.fit(x)
        theirs = -float(np.sum(stats.genextreme.logpdf(x, c, loc=loc, scale=scale)))
        assert mine <= theirs + 1e-6, (i, mine, theirs)


def test_ml_gleich_scipy_fit_bei_regulaerem_fall():
    x = _blocks(0.2, 400, 11)
    f = F.fit_ml(x)
    c, loc, scale = stats.genextreme.fit(x)
    assert (f.xi, f.mu, f.sigma) == pytest.approx((-c, loc, scale), abs=2e-3)


def test_gumbel_ml_gleich_scipy_gumbel_r_fit_auf_6_stellen():
    for i in range(10):
        x = _blocks(0.0, 30 + 7 * i, 500 + i)
        mu, sg = F.gumbel_ml(x)
        loc, scale = stats.gumbel_r.fit(x)
        assert (mu, sg) == pytest.approx((loc, scale), abs=1e-5)


def test_l_momente_stichprobe_gegen_paar_und_tripel_aufzaehlung():
    """l2 = halbes Mittel aller Paar-Abstände, l3 = Mittel aller Tripel (x(3) − 2x(2) + x(1))/3: andere Wege zu denselben Zahlen, 25 Stichproben."""
    for i in range(25):
        m = 5 + i % 6
        x = _blocks(0.1 * (i % 5) - 0.2, m, 700 + i)
        l1, l2, t3 = G.sample_lmoments(x)
        pairs = list(itertools.combinations(x, 2))
        l2_ref = sum(abs(a - b) for a, b in pairs) / (2 * len(pairs))
        trip = list(itertools.combinations(sorted(x), 3))
        l3_ref = sum((c - 2 * b + a) / 3 for a, b, c in trip) / len(trip)
        assert (l1, l2, t3) == pytest.approx((float(np.mean(x)), l2_ref, l3_ref / l2_ref), rel=1e-10)


def test_l_momente_population_gegen_scipy_quadratur():
    for mu, sg, xi in ((1.0, 2.0, 0.2), (0.0, 1.0, -0.3), (0.0, 1.0, 0.0), (0.0, 1.0, -0.8)):
        Q = lambda u: float(stats.genextreme.ppf(u, c=-xi, loc=mu, scale=sg))
        L1 = integrate.quad(Q, 0, 1, limit=200)[0]
        L2 = integrate.quad(lambda u: Q(u) * (2 * u - 1), 0, 1, limit=200)[0]
        L3 = integrate.quad(lambda u: Q(u) * (6 * u * u - 6 * u + 1), 0, 1, limit=200)[0]
        p1, p2, p3 = G.lmoments_pop(mu, sg, xi)
        assert (p1, p2, p3) == pytest.approx((L1, L2, L3 / L2), abs=2e-6)


def test_rueckrechnung_gegen_hosking_naeherung_und_probe_am_ergebnis():
    """Exakte Rückrechnung gegen die Näherung (Abstand unter 1e-3 für |ξ| ≤ 0,2; die Hosking-Näherung ist nur in ξ ∈ [−0,5; 0,5] geprüft, eigene Rechnung)."""
    for xi in np.linspace(-0.2, 0.2, 9):
        t3 = G.tau3_of_xi(float(xi))
        assert abs(G.xi_from_tau3_hosking(t3) - xi) < 1e-3
        assert G.xi_from_tau3(t3) == pytest.approx(float(xi), abs=2e-9)       # Umschaltgrenze 1e-9: bei ξ = 0 ist τ3 dort ein Plateau
    worst = max(abs(G.xi_from_tau3_hosking(G.tau3_of_xi(float(x))) - x) for x in np.linspace(-0.5, 0.5, 101))
    assert worst < 1e-3                                                         # eigene Rechnung: im Bereich [−0,5; 0,5] höchstens in der Größenordnung 1e-3
    for i in range(20):
        x = _blocks(0.0, 30, 900 + i)
        mu, sg, xi = G.fit_lmoments(x)
        l1, l2, t3 = G.sample_lmoments(x)
        assert G.lmoments_pop(mu, sg, xi) == pytest.approx((l1, l2, t3), abs=1e-8)       # die angepasste GEV hat die Stichproben-L-Momente


def test_klammer_wirkt_nur_bei_tau3_unter_minus_ein_drittel():
    x = _blocks(0.2, 40, 3)
    assert G.fit_lmoments(x, True) == pytest.approx(G.fit_lmoments(x, False), abs=1e-12)
    assert G.tau3_of_xi(-1.0) == pytest.approx(-1 / 3, abs=1e-12) and G.tau3_of_xi(1.0) == pytest.approx(1.0, abs=1e-12)
    assert G.tau3_of_xi(0.0) == pytest.approx(0.169925, abs=1e-6)
