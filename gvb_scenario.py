"""Reihen und Block-Maxima aus dem SplitMix64 der Vorlage (`gvb_rng.py`): Zufalls-Seeds geben auf jeder Plattform dieselben Zufallszahlen (die daraus gerechneten Ergebnisse stimmen bis auf Rundungsstellen überein).

Ein Lauf braucht N gleichverteilte Zahlen (nicht N Normal- oder Lognormal-Ziehungen): die Beobachtungen werden nie einzeln erzeugt. Mit v = 1 - u gehört das Maximum eines Blocks
zum Minimum seiner v, und EINE Umkehrung `isf(min v)` gibt das Block-Maximum. Verschiedene Blockgrößen auf derselben Reihe sind damit Minima von Gruppen-Minima.
Für die Quelle „Exakte GEV“ gibt es keine Einzelbeobachtungen: jeder der m Blöcke ist eine Ziehung `gev_ppf(u)`."""

from dataclasses import dataclass

import numpy as np

import gvb_constants as C
import gvb_gev as G
import gvb_sources as S
from gvb_rng import MASK, SplitMix64

U_MIN = 2.0 ** -53


def rng_for(seed, *idx):
    """Eigener Strom je (Zufalls-Seed, Indizes): die Indizes werden über SplitMix64 gemischt, nie über hash() (pro Prozess gesalzen)."""
    r = SplitMix64(int(seed))
    for i in idx:
        r = SplitMix64(r.next_u64() ^ (((int(i) + 1) * 0xD1B54A32D192ED03) & MASK))
    return r


def uniforms(rng, n):
    rnd = rng.random
    return np.fromiter((rnd() for _ in range(n)), float, n)


def block_minima_of_v(v, b):
    """Minima der v über aufeinanderfolgende Blöcke zu b (b muss die Länge teilen)."""
    v = np.asarray(v, float)
    if b < 1 or v.size % b:
        raise ValueError(f"Blockgröße b = {b} teilt die Reihenlänge {v.size} nicht")
    return v.reshape(-1, b).min(axis=1)


def series_v(rng, n=C.N_OBS):
    """N gleichverteilte v = 1 - u in (0, 1]: die Reihe als Überlebenswahrscheinlichkeiten."""
    return 1.0 - uniforms(rng, n)


def block_maxima(source, v, b):
    """Block-Maxima einer Reihe (als v-Feld) für eine Quelle mit Einzelbeobachtungen."""
    return S.isf(source, block_minima_of_v(v, b))


def exact_gev_blocks(rng, m, mu=0.0, sigma=1.0, xi=0.0):
    """m Block-Maxima, exakt GEV(μ, σ, ξ), durch Umkehrung je einer gleichverteilten Zahl."""
    u = np.maximum(uniforms(rng, m), U_MIN)
    return G.gev_ppf(u, mu, sigma, xi)


@dataclass(frozen=True)
class Series:
    source: str
    b: int
    m: int
    maxima: tuple          # die m Block-Maxima in Reihenfolge der Blöcke


def make_series(source, b, seed, n=C.N_OBS, xi=0.0, m=None, index=0):
    """Eine Reihe: bei „Exakte GEV“ m Blöcke GEV(0, 1, ξ) (b und n zählen dort nicht), sonst n Beobachtungen in Blöcken zu b."""
    rng = rng_for(seed, 1, index)
    if source == "Exakte GEV":
        mm = int(m if m is not None else 50)
        return Series(source, 0, mm, tuple(exact_gev_blocks(rng, mm, xi=xi).tolist()))
    v = series_v(rng, n)
    mx = block_maxima(source, v, b)
    return Series(source, b, mx.size, tuple(mx.tolist()))
