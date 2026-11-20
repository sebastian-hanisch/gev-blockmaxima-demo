"""Literaturtest (ORACLE-PLAYBOOK §6): Aussagen aus gelesenen Quellen als Test. Gelesen (Stufe W): Coles (2001), Kap. 3 (Gleichungen 3.2 und 3.4, Beispiele 3.1 bis 3.3, S. 47 bis 57,
Regularität nach Smith S. 55) und Embrechts/Klüppelberg/Mikosch (1997), 6.3; hier nur Gleichungen und Befunde mit Fundstelle, nichts wörtlich. **Kein Test** aus Smith 1985, Hosking/Wallis/Wood 1985,
Hosking 1990, Cunnane 1978, Fisher/Tippett 1928 (nicht gelesen, in der README „offen“)."""

import math

import pytest

import gvb_fit as F
import gvb_gev as G
import gvb_scenario as SC

stats = pytest.importorskip("scipy.stats")


def _blocks(xi, m, seed, idx=0):
    return SC.exact_gev_blocks(SC.rng_for(seed, idx), m, xi=xi)


def test_literatur_coles_wiederkehrwert_3_4_gegen_scipy_isf():
    """Coles (2001), Gl. 3.4: x_p = μ − σ/ξ (1 − y^(−ξ)), y = −ln(1 − p); gelesen (W), kein Wortlaut. Gegen scipy isf (c = −ξ) auf 1e-14 relativ."""
    for mu, sg, xi in ((0.0, 1.0, 0.3), (2.0, 1.5, -0.4)):
        for p in (0.1, 0.01, 0.001):
            y = -math.log(1 - p)
            coles = mu - sg / xi * (1 - y ** (-xi))
            assert coles == pytest.approx(float(stats.genextreme.isf(p, c=-xi, loc=mu, scale=sg)), rel=1e-13)
    # Coles Bsp. 3.1 bis 3.3 (Exponential, Standard-Fréchet, Gleich): die Max-Verteilung ist F^b; die GEV-Passung steht in den Handbeispielen
    assert float(G.gev_cdf(2.0, 1.0, 1.0, 1.0)) == pytest.approx(math.exp(-1 / 2.0), rel=1e-14)        # Standard-Fréchet exp(−1/x) = GEV(μ = 1, σ = 1, ξ = 1)
    b, x = 1000, 0.7                                                                                  # Exponential: (1 − e^−(x + ln b))^b → Gumbel exp(−e^−x)
    assert (1 - math.exp(-(x + math.log(b)))) ** b == pytest.approx(float(G.gev_cdf(x, 0.0, 1.0, 0.0)), abs=1e-3)


def test_smith_regime_der_likelihood():
    """Smith (1985) nach Coles S. 55 (W): ξ < −1 nicht berechenbar. Die Optimierer-Nebenbedingung dieser Demo ist ξ ≥ −1 (scipy kennt sie nicht): ML auf ξ = −1,2 klebt am Rand."""
    x = _blocks(-1.2, 40, 5)
    f = F.fit_ml(x)
    assert f.xi >= -1.0 and f.edge == (f.xi <= -0.999)
