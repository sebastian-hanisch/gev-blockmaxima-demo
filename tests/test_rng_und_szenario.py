"""Zufallsgenerator (SplitMix64 der Vorlage) und Erzeugung der Reihen: feste Werte, Bestimmtheit, Gleichverteilung, Block-Maxima als Minimum der v, Quellen aus einer Zahl je Block."""

import math

import numpy as np
import pytest

import gvb_constants as C
import gvb_scenario as SC
import gvb_sources as S
from gvb_rng import MASK, SplitMix64


def test_splitmix64_bekannte_werte():
    """Der erste Ausgang von SplitMix64 mit Zufalls-Seed 0 ist 0xE220A8397B1DCDAF (Referenzwert des Verfahrens); Zustand springt um die goldene Konstante."""
    r = SplitMix64(0)
    assert r.next_u64() == 0xE220A8397B1DCDAF
    assert r.state == 0x9E3779B97F4A7C15 & MASK
    assert 0.0 <= SplitMix64(1).random() < 1.0


def test_gleiche_zufalls_seeds_gleiche_folgen_und_getrennte_stroeme():
    a = SC.uniforms(SC.rng_for(5, 1, 2), 50)
    b = SC.uniforms(SC.rng_for(5, 1, 2), 50)
    c = SC.uniforms(SC.rng_for(5, 1, 3), 50)
    d = SC.uniforms(SC.rng_for(6, 1, 2), 50)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, d)
    assert len({tuple(SC.uniforms(SC.rng_for(9, i), 3)) for i in range(200)}) == 200


def test_uniforms_sind_gleichverteilt_im_toleranzband():
    """200 000 Zahlen: Mittel 0,5 ± 4 Standardfehler (σ = 1/√12), zehn Klassen je 10 % ± 4 Standardfehler (nur ein Band, kein Gleichverteilungsbeweis)."""
    u = SC.uniforms(SC.rng_for(3), 200000)
    assert abs(u.mean() - 0.5) < 4 * math.sqrt(1 / 12 / 200000)
    freq = np.bincount((u * 10).astype(int), minlength=10) / u.size
    assert np.all(np.abs(freq - 0.1) < 4 * math.sqrt(0.1 * 0.9 / 200000))
    assert u.min() >= 0.0 and u.max() < 1.0


def test_blockmaxima_sind_das_minimum_der_ueberlebenswahrscheinlichkeiten():
    """Brute Force: Maximum der b einzelnen Beobachtungen = isf(min v) für jede monotone Umkehrung; kleine Reihe, alle vier Quellen mit Einzelbeobachtungen."""
    v = SC.series_v(SC.rng_for(8, 1), 360)
    for source in ("Exponential", "Normal", "Pareto-Verteilung", "Vehikel B", "Standard-Fréchet"):
        einzeln = S.isf(source, v)                       # jede Beobachtung einzeln
        for b in (6, 30, 360):
            brute = einzeln.reshape(-1, b).max(axis=1)
            assert np.allclose(SC.block_maxima(source, v, b), brute, rtol=1e-12)


def test_verschachtelte_blockgroessen_sind_minima_von_gruppen_minima():
    v = SC.series_v(SC.rng_for(8, 2), 360)
    m6 = SC.block_minima_of_v(v, 6)
    assert np.array_equal(SC.block_minima_of_v(v, 60), m6.reshape(-1, 10).min(axis=1))
    assert np.array_equal(SC.block_minima_of_v(v, 360), SC.block_minima_of_v(m6, 60))


def test_block_maximum_hat_die_verteilung_f_hoch_b():
    """Band: Anteil der Block-Maxima der Exponentialquelle unter ln(b) + x gleich Gumbel-nah (1 − e^−(x+ln b))^b; 2 000 Blöcke zu b = 30 aus einer Reihe."""
    v = SC.series_v(SC.rng_for(12), 60000)
    x = SC.block_maxima("Exponential", v, 30)
    for t in (0.0, 1.0, 2.0):
        p = (1 - math.exp(-(t + math.log(30)))) ** 30
        assert abs(float(np.mean(x <= t + math.log(30))) - p) < 4 * math.sqrt(p * (1 - p) / x.size)


def test_exakte_gev_bloecke_haben_die_verteilung_g():
    import gvb_gev as G
    x = SC.exact_gev_blocks(SC.rng_for(4), 20000, xi=0.2)
    for p in (0.5, 0.9, 0.99):
        q = float(G.pegel(1 - p, 0.0, 1.0, 0.2))
        assert abs(float(np.mean(x <= q)) - p) < 4 * math.sqrt(p * (1 - p) / x.size)


def test_make_series_bestimmt_und_bei_exakter_gev_ohne_beobachtungen():
    a = SC.make_series("Normal", 60, 35)
    assert a == SC.make_series("Normal", 60, 35) and a.m == C.N_OBS // 60 and a.b == 60
    assert SC.make_series("Normal", 60, 36) != a
    e = SC.make_series("Exakte GEV", 60, 35, xi=0.0, m=10)
    assert e.m == 10 and e.b == 0 and len(e.maxima) == 10


def test_b_das_n_nicht_teilt_ist_ein_fehler_mit_meldung():
    v = SC.series_v(SC.rng_for(1), 100)
    with pytest.raises(ValueError, match="teilt"):
        SC.block_minima_of_v(v, 7)
    with pytest.raises(ValueError):
        SC.block_minima_of_v(v, 0)
