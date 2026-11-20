"""Populations-Bias per Quadratur: die ML-Anpassung an die EXAKTE Verteilung F^b der Block-Maxima (Grenzwert m → ∞, kein Zufall).

Die Quantilknoten u_i = (i + 1/2)/K der Max-Verteilung F^b geben x_i = isf(1 - u_i^(1/b)); die Populations-ML minimiert die mittlere negative Log-Likelihood über diese Knoten.
Das ist ein GRENZWERT nur für Quellen mit ξ_pop ≥ 0 (Exponential, Pareto-Verteilung, Standard-Fréchet, Vehikel B). Für die Normalverteilung gibt es keinen: die GEV mit ξ̂ < 0
hat einen oberen Endpunkt, die Normalverteilung hat darüber noch Wahrscheinlichkeit, die erwartete Log-Likelihood ist nicht beschränkt, und das Ergebnis hängt davon ab, wie viel
Tail die Quadratur mitnimmt (`logit_nodes`, Logit-Rand S). `normal_reference` zeigt stattdessen den Bias einer einzigen sehr großen Stichprobe."""

import math

import numpy as np

import gvb_constants as C
import gvb_fit as F
import gvb_gev as G
import gvb_scenario as SC
import gvb_sources as S

NODES = 20000


def max_nodes(source, b, nodes=NODES):
    """Quantilknoten der Max-Verteilung F^b (gleiches Gewicht)."""
    u = (np.arange(nodes) + 0.5) / nodes
    sf = -np.expm1(np.log(u) / b)       # 1 - u^(1/b)
    return S.isf(source, sf)


def logit_nodes(source, b, rand, npts=20001):
    """Quantilknoten im Logit-Raster s ∈ [-S, S] (u = 1/(1 + e^-s), Gewicht u (1 - u)): symmetrisch abgeschnitten, Tail-Masse e^-S."""
    s = np.linspace(-rand, rand, npts)
    logu = -np.log1p(np.exp(-s))
    sf = -np.expm1(logu / b)
    u = np.exp(logu)
    return S.isf(source, sf), u * (1 - u)


def wnll(theta, x, w):
    """Gewichtete mittlere negative Log-Likelihood je Masse; 1e300 für einen Knoten außerhalb des Trägers oder ξ < -1."""
    mu, ls, xi = theta
    if xi < G.XI_MIN:
        return G.HUGE
    z = (x - mu) * math.exp(-ls)
    if abs(xi) < 1e-7:
        terms = ls + z + np.exp(-z)
    else:
        t = 1 + xi * z
        if np.any(t <= 0):
            return G.HUGE
        lt = np.log(t)
        terms = ls + (1 + 1 / xi) * lt + np.exp(-lt / xi)
    return float(np.sum(w * terms) / np.sum(w))


def population_fit(x, w=None):
    """Populations-ML (μ, σ, ξ) und Populations-Gumbel (μ, σ) an Quantilknoten x mit Gewichten w."""
    w = np.ones_like(x) if w is None else w
    try:
        f0 = F.fit_ml(x[::50])
        start = [f0.mu, math.log(f0.sigma), f0.xi]
    except G.FitError:
        start = [float(x.mean()), math.log(float(x.std())), 0.0]
    scale = np.array([float(x.std()), 0.1, 0.1])
    best = None
    for dxi in (0.0, -0.1, 0.1):
        p0 = np.array([start[0], start[1], start[2] + dxi])
        if wnll(p0, x, w) >= G.HUGE:
            continue
        p, f = F.nm_minimize(lambda t: wnll(t, x, w), p0, scale, 1e-8, 1e-12, 4000)
        if best is None or f < best[1]:
            best = (p, f)
    best = F.nm_minimize(lambda t: wnll(t, x, w), best[0], scale * 0.05, 1e-11, 1e-15, 4000)
    mu, ls, xi = best[0]
    pg, _ = F.nm_minimize(lambda t: wnll((t[0], t[1], 0.0), x, w), np.array([float(x.mean()), math.log(float(x.std()))]), scale[:2], 1e-10, 1e-14, 4000)
    return (float(mu), math.exp(float(ls)), float(xi)), (float(pg[0]), math.exp(float(pg[1])))


def population_bias(source, b, q=C.Q_TARGET, nodes=NODES):
    """Bias des Grenzwerts m → ∞ in ln: ln(Pegel der Populations-Anpassung / wahrer Pegel), für ML-GEV und Gumbel; nur sinnvoll für Quellen mit ξ_pop ≥ 0."""
    x = max_nodes(source, b, nodes)
    (mu, s, xi), (mg, sg) = population_fit(x)
    pb = S.block_exceedance(b, q)
    zt = S.true_level(source, q)
    z = float(G.pegel(pb, mu, s, xi))
    return {
        "ln_v": math.log(z / zt), "xi": xi, "pegel": z, "wahr": zt,
        "ln_v_gumbel": math.log(float(G.pegel(pb, mg, sg, 0.0)) / zt),
    }


def population_bias_logit(source, b, rand, q=C.Q_TARGET):
    """Wie `population_bias`, aber mit Logit-Rand S (zeigt, dass Normal vom Abschneiden abhängt)."""
    x, w = logit_nodes(source, b, rand)
    (mu, s, xi), _ = population_fit(x, w)
    return math.log(float(G.pegel(S.block_exceedance(b, q), mu, s, xi)) / S.true_level(source, q)), xi


def frechet_population(b, nodes=NODES):
    """Standard-Fréchet: F^b ist GEV(μ = b, σ = b, ξ = 1) für jedes b (max-stabil); die Populations-ML gibt (μ/b, σ/b, ξ) und den Bias in ln."""
    x = max_nodes("Standard-Fréchet", b, nodes)
    (mu, s, xi), _ = population_fit(x)
    pb = S.block_exceedance(b, C.Q_TARGET)
    return {"mu_je_b": mu / b, "sigma_je_b": s / b, "xi": xi,
            "ln_v": math.log(float(G.pegel(pb, mu, s, xi)) / S.true_level("Standard-Fréchet", C.Q_TARGET))}


def normal_reference(b, m, seed=2026):
    """Bias einer EINZIGEN sehr großen Stichprobe von m Block-Maxima der Normalverteilung (nur als Referenz: kein Grenzwert).
    Jedes Block-Maximum wird direkt als isf(1 - u^(1/b)) mit einer gleichverteilten Zahl u gezogen (die Verteilung des Maximums von b Normalwerten)."""
    rng = SC.rng_for(seed, 9, b, m)
    u = np.maximum(SC.uniforms(rng, m), SC.U_MIN)
    x = S.isf("Normal", -np.expm1(np.log(u) / b))
    f = F.fit_ml(x)
    zt = S.true_level("Normal", C.Q_TARGET)
    return {"ln_v": math.log(f.level(S.block_exceedance(b, C.Q_TARGET)) / zt), "xi": f.xi, "endpunkt": f.endpoint, "groesster_wert": float(x.max())}
