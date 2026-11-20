"""Handbeispiele H1 bis H7 des Plans als exakte Tests: Brüche, Ganzzahlen, Tafelwerte. Die Rechnung steht im Test (nicht der Wert des Codes abgeschrieben); nur Standardbibliothek und numpy."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

import gvb_fit as F
import gvb_gev as G

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------------------------------
# H1 und H2: Likelihood von zwei Werten, Pegel
# ---------------------------------------------------------------------------------------------------
def test_h1_nll_von_zwei_werten_von_hand():
    """x = (0, 1), μ = 0, σ = 1, ξ = 0,5: t = (1; 1,5), nll = m ln σ + (1 + 1/ξ) Σ ln tᵢ + Σ tᵢ^(−1/ξ) = 3 ln 1,5 + 1 + 4/9."""
    von_hand = 3 * math.log(1.5) + 1 + 4 / 9
    assert G.nll_values([0.0, 1.0], 0.0, 1.0, 0.5) == pytest.approx(von_hand, abs=1e-14)
    assert von_hand == pytest.approx(2.66084, abs=1e-5)
    # dieselbe Zahl über die Optimierer-Form (μ, ln σ, ξ)
    assert G.nll((0.0, 0.0, 0.5), np.array([0.0, 1.0])) == pytest.approx(von_hand, abs=1e-14)


def test_h2_pegel_von_hand():
    """x_p = 2((−ln 0,99)^(−0,5) − 1) = 17,949853 für ξ = 0,5; Gumbel −ln(−ln 0,99) = 4,600149; Lage-Skalen-Verhalten; ξ → 0 geht stetig über."""
    assert float(G.pegel(0.01, 0.0, 1.0, 0.5)) == pytest.approx(17.949853, abs=1e-6)
    assert float(G.pegel(0.01, 0.0, 1.0, 0.5)) == pytest.approx(2 * ((-math.log(0.99)) ** -0.5 - 1), rel=1e-14)
    assert float(G.pegel(0.01, 0.0, 1.0, 0.0)) == pytest.approx(4.600149, abs=1e-6)
    assert float(G.pegel(0.01, 0.0, 1.0, 0.0)) == pytest.approx(-math.log(-math.log(0.99)), rel=1e-14)
    assert float(G.pegel(0.01, 0.0, 1.0, 1e-8)) == pytest.approx(4.600149, abs=1e-5)       # gleich neben der Umschaltgrenze 1e-9, aber auf der GEV-Seite
    assert float(G.pegel(0.01, 5.0, 3.0, 0.2)) == pytest.approx(5.0 + 3.0 * float(G.pegel(0.01, 0.0, 1.0, 0.2)), rel=1e-14)
    assert float(G.pegel(0.001, 0.0, 1.0, -0.3)) < 1 / 0.3 and G.endpoint(0.0, 1.0, -0.3) == pytest.approx(1 / 0.3)   # Pegel bleibt unter dem Endpunkt μ − σ/ξ


def test_h2_pegel_gegen_bisektion_auf_der_verteilungsfunktion():
    """Zweiter Rechenweg: Bisektion auf 1 − G(x) = p für drei Parameter und drei p."""
    for mu, sg, xi in ((0.0, 1.0, 0.5), (3.0, 2.0, -0.3), (10.0, 4.0, 0.2)):
        for p in (0.1, 0.01, 0.001):
            lo = mu - 1000.0
            hi = mu + 1e9 if xi > 0 else mu - sg / xi - 1e-12
            for _ in range(300):
                mid = 0.5 * (lo + hi)
                if 1 - float(G.gev_cdf(mid, mu, sg, xi)) > p:
                    lo = mid
                else:
                    hi = mid
            x = float(G.pegel(p, mu, sg, xi))
            assert x == pytest.approx(0.5 * (lo + hi), rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------------------------------
# H3: fünf Block-Maxima, L-Momente von Hand
# ---------------------------------------------------------------------------------------------------
def _pwm_fractions(x):
    m = len(x)
    xs = sorted(x)
    b0 = Fraction(sum(xs), m)
    b1 = sum(Fraction(j - 1, m - 1) * xs[j - 1] for j in range(1, m + 1)) / m
    b2 = sum(Fraction((j - 1) * (j - 2), (m - 1) * (m - 2)) * xs[j - 1] for j in range(1, m + 1)) / m
    return b0, b1, b2


def test_h3_l_momente_als_brueche_und_ueber_paare_und_tripel():
    x = [2, 3, 5, 6, 9]
    b0, b1, b2 = _pwm_fractions(x)
    assert (b0, b1, b2) == (5, Fraction(67, 20), Fraction(77, 30))
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    assert (l1, l2, l3) == (5, Fraction(17, 10), Fraction(3, 10))
    assert l3 / l2 == Fraction(3, 17)
    # zwei unabhängige Wege: l2 = halbes Mittel aller 10 Paar-Abstände, l3 = Mittel aller 10 Tripel (x(3) − 2x(2) + x(1))/3
    paare = list(itertools.combinations(x, 2))
    assert Fraction(sum(abs(a - b) for a, b in paare), 2 * len(paare)) == l2
    tripel = list(itertools.combinations(sorted(x), 3))
    assert sum(Fraction(c - 2 * b + a, 3) for a, b, c in tripel) / len(tripel) == l3
    # der Code gibt dieselben Zahlen
    c1, c2, t3 = G.sample_lmoments(x)
    assert (c1, c2) == (5.0, pytest.approx(1.7, abs=1e-15)) and t3 == pytest.approx(3 / 17, rel=1e-14)
    assert float(Fraction(3, 17)) == pytest.approx(0.176471, abs=1e-6)
    assert 2 * math.log(3) / math.log(2) - 3 == pytest.approx(0.169925, abs=1e-6)       # τ3 der Gumbel-Verteilung


def test_h3_rueckrechnung_reproduziert_die_l_momente_und_drei_pegel():
    x = [2, 3, 5, 6, 9]
    l1, l2, t3 = G.sample_lmoments(x)
    mu, sg, xi = G.fit_lmoments(x)
    assert 0.0 < xi < 0.05 and xi == pytest.approx(0.0102, abs=1e-4)
    p1, p2, p3 = G.lmoments_pop(mu, sg, xi)
    assert (p1, p2, p3) == (pytest.approx(5.0, abs=1e-9), pytest.approx(1.7, abs=1e-9), pytest.approx(3 / 17, abs=1e-9))
    assert (mu, sg) == (pytest.approx(3.5730, abs=1e-4), pytest.approx(2.4295, abs=1e-4))
    assert float(G.pegel(0.01, mu, sg, xi)) == pytest.approx(15.01, abs=0.01)
    # Hosking-Näherung gegen die exakte Rückrechnung (Abstand unter 1e-3 für |ξ| klein)
    assert abs(G.xi_from_tau3_hosking(t3) - xi) < 1e-3
    # Gumbel-L-Momente in geschlossener Form: σ = l2/ln 2, μ = l1 − γσ
    sg_g = l2 / math.log(2)
    mu_g = l1 - G.EULER * sg_g
    assert (sg_g, mu_g) == (pytest.approx(2.452582, abs=1e-5), pytest.approx(3.584331, abs=1e-5))
    assert float(G.pegel(0.01, mu_g, sg_g, 0.0)) == pytest.approx(14.87, abs=0.01)
    # Gumbel-ML auf denselben fünf Werten: σ̂ = 1,989, μ̂ = 3,833, 100er-Wert 12,98; alle drei Pegel liegen über dem größten der fünf Werte (9)
    mu_m, sg_m = F.gumbel_ml(np.array(x, float))
    assert (mu_m, sg_m) == (pytest.approx(3.833, abs=1e-3), pytest.approx(1.989, abs=1e-3))
    assert float(G.pegel(0.01, mu_m, sg_m, 0.0)) == pytest.approx(12.98, abs=0.01)
    assert min(15.01, 14.87, 12.98) > max(x)
    # Gumbel-ML erfüllt die Likelihood-Gleichung s = x̄ − Σ x e^(−x/s)/Σ e^(−x/s) (zweiter Weg zur selben Zahl)
    xa = np.array(x, float)
    w = np.exp(-xa / sg_m)
    assert sg_m == pytest.approx(xa.mean() - float(np.sum(xa * w) / np.sum(w)), abs=1e-9)


def test_h3_m3_l3_gleich_dem_einzigen_tripel_und_m2_hat_null_im_nenner():
    x = [1, 2, 6]
    b0, b1, b2 = _pwm_fractions(x)
    assert b2 == Fraction(6, 3)                                   # nur j = 3 hat Gewicht: x(3)/3
    assert 6 * b2 - 6 * b1 + b0 == Fraction(6 - 2 * 2 + 1, 3)       # gleich dem einzigen Tripel
    with pytest.raises(ZeroDivisionError):
        Fraction(1 * 0, (2 - 1) * (2 - 2))                          # m = 2: (m − 1)(m − 2) = 0
    with pytest.raises(G.FitError):
        G.sample_lmoments([1.0, 2.0])                               # der Code meldet es als Fehler statt Zahl


def test_h3b_die_klammer_am_beispiel_0_3_3_3_4():
    x = [0, 3, 3, 3, 4]
    b0, b1, b2 = _pwm_fractions(x)
    t3 = (6 * b2 - 6 * b1 + b0) / (2 * b1 - b0)
    assert t3 == Fraction(-1, 2) and t3 < Fraction(-1, 3)          # unter −1/3 verlangt die exakte Rückrechnung ξ̂ < −1
    mu, sg, xi = G.fit_lmoments(x, klammer=False)
    assert xi == pytest.approx(-1.4946, abs=1e-3) and xi < -1.0
    assert G.endpoint(mu, sg, xi) == pytest.approx(3.84, abs=1e-2) and G.endpoint(mu, sg, xi) < max(x)    # unmögliche Anpassung
    assert G.lmoments_pop(0.0, 1.0, xi)[2] == pytest.approx(-0.5, abs=1e-9)                               # zweiter Weg: τ3 der angepassten GEV
    mu_c, sg_c, xi_c = G.fit_lmoments(x, klammer=True)
    assert xi_c == pytest.approx(-0.999, abs=1e-12)
    assert G.endpoint(mu_c, sg_c, xi_c) == pytest.approx(4.20, abs=1e-2) and G.endpoint(mu_c, sg_c, xi_c) > max(x)
    assert G.lmoments_pop(0.0, 1.0, -0.999)[2] > -0.5                # die geklammerte Anpassung reproduziert τ3 nicht: Wahl der Demo


def test_h3_populations_l_momente_gegen_quadratur():
    """Populations-L-Momente geschlossen gegen Quadratur über die Quantilfunktion (Gauß-Legendre, 4000 Knoten) auf 2e-6."""
    xg, wg = np.polynomial.legendre.leggauss(2000)
    th, wth = (xg + 1) / 2, wg / 2
    u = np.sin(np.pi * th / 2) ** 2                       # Abbildung auf (0, 1), die beide Ränder verdichtet (der Tail hat dort eine Potenz-Singularität)
    w = wth * (np.pi / 2) * np.sin(np.pi * th)
    for mu, sg, xi in ((1.0, 2.0, 0.2), (0.0, 1.0, -0.3), (0.0, 1.0, 0.0)):
        q = np.array([float(G.pegel(1 - t, mu, sg, xi)) for t in u])
        L1 = float(np.sum(w * q))
        L2 = float(np.sum(w * q * (2 * u - 1)))
        L3 = float(np.sum(w * q * (6 * u * u - 6 * u + 1)))
        p1, p2, p3 = G.lmoments_pop(mu, sg, xi)
        assert (p1, p2, p3) == (pytest.approx(L1, abs=2e-6), pytest.approx(L2, abs=2e-6), pytest.approx(L3 / L2, abs=2e-6))


# ---------------------------------------------------------------------------------------------------
# H4 bis H7
# ---------------------------------------------------------------------------------------------------
def test_h4_blockgroesse_exponential_harmonische_zahlen():
    """E[M_b] = H_b gegen den Gumbel-Grenzwert ln b + γ: Differenz 0,2296 (b = 2), 0,0810 (b = 6), 0,0492 (b = 10), 0,00137 (b = 365); Var[M_2] = 5/4 gegen π²/6."""
    h = lambda n: sum(Fraction(1, k) for k in range(1, n + 1))
    assert (h(2), h(6), h(10)) == (Fraction(3, 2), Fraction(49, 20), Fraction(7381, 2520))
    for b, diff in ((2, 0.2296), (6, 0.0810), (10, 0.04917)):
        assert float(h(b)) - (math.log(b) + G.EULER) == pytest.approx(diff, abs=1e-4)
    assert float(h(365)) - (math.log(365) + G.EULER) == pytest.approx(0.001369, abs=1e-5)
    # Var[M_2] = 5/4 (zwei Exponentialwerte: Max = Min + Rest, Var = 1/4 + 1)
    assert Fraction(1, 4) + Fraction(1) == Fraction(5, 4) and math.pi ** 2 / 6 - 1.25 == pytest.approx(0.394934, abs=1e-6)
    # Verschachtelung (F^b)² = F^(2b): zwei Blöcke zu drei Würfen sind ein Block zu sechs, als Brüche; Aufzählung für b = 4
    for k in range(1, 7):
        assert Fraction(k, 6) ** 3 * Fraction(k, 6) ** 3 == Fraction(k, 6) ** 6
    assert len([o for o in itertools.product(range(1, 7), repeat=4) if max(o) <= 4]) == 4 ** 4


def test_h5_frechet_ist_max_stabil_und_gleich_nicht():
    for b in (1, 6, 60, 360):
        for xv in (0.5 * b, 1.0 * b, 3.0 * b, 20.0 * b):
            assert math.exp(-1 / xv) ** b == pytest.approx(float(G.gev_cdf(xv, b, b, 1.0)), abs=1e-14)
    for b, dist in ((2, 0.1620), (10, 0.0281)):
        for xv in (0.5, 0.8, 0.95, 1.0):
            assert abs(xv ** b - float(G.gev_cdf(xv, 1 - 1 / b, 1 / b, -1.0))) <= dist     # Abstand zum Grenzgesetz bleibt unter dem Abstand aus „Das Größte von n“
    assert 0.95 ** 10 == pytest.approx(0.598737, abs=1e-6)
    assert float(G.gev_cdf(0.95, 0.9, 0.1, -1.0)) == pytest.approx(math.exp(-0.5), abs=1e-14)
    assert float(G.gev_cdf(1.0, 0.9, 0.1, -1.0)) == pytest.approx(1.0, abs=1e-12) and float(G.gev_cdf(1.0001, 0.9, 0.1, -1.0)) == 1.0


def test_h6_likelihood_ohne_grenze():
    """Drei Werte (1, 2, 3), ξ = −1,5, σ = 1, Endpunkt 3 + ε: je Zehnerpotenz Abstand sinkt die nll um (1 + 1/ξ) ln 10 = 0,7675."""
    x3 = [1.0, 2.0, 3.0]
    vals = [G.nll_values(x3, 3.0 + e - 1 / 1.5, 1.0, -1.5) for e in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]
    step = (1 + 1 / -1.5) * math.log(10)
    assert step == pytest.approx(0.767528, abs=1e-6)
    for a, b in zip(vals[2:], vals[3:]):
        assert a - b == pytest.approx(step, abs=0.02)
    schritte = [a - b for a, b in zip(vals, vals[1:])]
    assert [round(s, 2) for s in schritte[:4]] == [1.17, 0.83, 0.78, 0.77]                 # Schritte je Zehnerpotenz (README Befund 5), nähern sich 0,7675 nur an
    assert all(abs(s - step) < 0.02 for s in schritte[2:])
    assert vals[0] - vals[-1] > 3.5                                       # über fünf Zehnerpotenzen sinkt die nll um mehr als 3,5
    # ξ = −0,8: die nll steigt, wenn der Endpunkt an das größte Beobachtete heranrückt
    v2 = [G.nll_values(x3, 3.0 + e - 1.25, 1.0, -0.8) for e in (1e-2, 1e-4, 1e-6)]
    assert v2[0] < v2[1] < v2[2]
    # die Optimierer-Form verweigert ξ < −1
    assert G.nll((0.0, 0.0, -1.5), np.array(x3)) == G.HUGE


def test_h7_plotting_positionen_weibull_gringorten():
    mm = 10
    wb = [Fraction(i, mm + 1) for i in range(1, mm + 1)]
    assert wb[-1] == Fraction(10, 11)
    gr = [(i - 0.44) / (mm + 0.12) for i in range(1, mm + 1)]
    assert gr[-1] == pytest.approx(0.9447, abs=1e-4) and gr[0] == pytest.approx(0.0553, abs=1e-4)
    # G(E X_(i)) der Gumbel-Verteilung bei m = 10 per Quadratur (trapezförmig auf feinem Gitter); K: 0,0678 / 0,1663 / 0,4616 / 0,8513 / 0,9454
    t = np.linspace(-12, 40, 400001)
    Gt = np.exp(-np.exp(-t))
    for i, ref in ((1, 0.0678), (2, 0.1663), (5, 0.4616), (9, 0.8513), (10, 0.9454)):
        c = math.factorial(mm) / (math.factorial(i - 1) * math.factorial(mm - i))
        f = c * Gt ** (i - 1) * (1 - Gt) ** (mm - i) * np.exp(-t) * Gt
        mean_i = float(_trapz(t * f, t))
        assert math.exp(-math.exp(-mean_i)) == pytest.approx(ref, abs=6e-4)
        if i == 10:
            assert float(wb[-1]) - math.exp(-math.exp(-mean_i)) < -0.03     # Position nach Weibull des Größten liegt deutlich unter G(E X_(10))
    # E[U_(3)] = 3/6 bei m = 5 (Beta-Mittel)
    u = np.linspace(0, 1, 200001)
    ci = math.factorial(5) / (math.factorial(2) * math.factorial(2))
    assert float(_trapz(u * ci * u ** 2 * (1 - u) ** 2, u)) == pytest.approx(0.5, abs=1e-12)
