"""Orakel Vehikel B: eigene Aufzählung aus der Definition gegen die kopierte Tabelle, exakte Überlebensfunktion gegen scipy, Pegel und Umkehrtabelle.

Definition (gemeinsam für die Linie, Quelle: Demo „Überschreitungen und die GPD“, Abschnitt Vehikel B): Art (Import 0,35 / Export 0,30 / Umschlag 0,25 / Leer 0,10) × Kunde (12, gleich wahrscheinlich, Effekt UK·U_SCALE)
× Zoll (nur Import, Anteil 0,25) × Wochenende (Anteil 2/7) × Vorlauf (Export gleichverteilt (1, 10), Umschlag (0,5, 6), sonst 0; Gauß-Legendre mit 48 Knoten);
je Komponente ln(Verweilzeit) ~ N(μ, s²) mit μ = ln(Median) + Effekt, s = S_BASE[Art]."""

import math

import numpy as np
import pytest
from vehikel_b_def import ART_P, S_BASE, U_SCALE, UK, aufzaehlung  # noqa: E402,F401

import gvb_sources as S


def test_aufzaehlung_4704_schritte_2376_mit_gewicht_1224_komponenten():
    schritte, mit_gewicht, summe = aufzaehlung()
    assert (schritte, mit_gewicht, len(summe)) == (4704, 2376, 1224)
    assert sum(summe.values()) == pytest.approx(1.0, abs=1e-12)
    from collections import Counter
    n_art = Counter(s for (_, s) in summe)
    assert dict(n_art) == {0.55: 48, 0.25: 576, 0.35: 576, 0.9: 24}                     # Import 48, Export 576, Umschlag 576, Leer 24


def test_kopierte_tabelle_gleich_eigener_aufzaehlung():
    """Jede Komponente der kopierten Tabelle (Gewicht, μ, s) stimmt mit der eigenen Aufzählung überein (Gewichte auf 1e-12 relativ, μ auf 1e-10)."""
    _, _, summe = aufzaehlung()
    w, mu, s = S.mixture_components()
    assert len(w) == len(summe) == 1224
    for wi, mi, si in zip(w, mu, s):
        key = (round(float(mi), 10), float(si))
        assert key in summe, key
        assert float(wi) == pytest.approx(summe[key], rel=1e-12)


def test_ueberlebensfunktion_gegen_scipy_und_die_benannten_zielgroessen():
    sp = pytest.importorskip("scipy.special")
    w, mu, s = S.mixture_components()

    def sf_ref(x):
        z = (math.log(x) - mu) / s
        return float(np.sum(w * np.exp(sp.log_ndtr(-z))))
    for x in (1.0, 5.0, 30.0, 100.0, 287.0):
        assert S.mixture_sf(x) == pytest.approx(sf_ref(x), rel=1e-11)
    assert S.mixture_sf(30.0) == pytest.approx(0.009830, abs=5e-7)                     # P(D > 30 Tage)
    assert S.mixture_level(1e-5) == pytest.approx(287.158, abs=5e-4)                   # Zielgröße (a): je Container p = 1e-5
    p_tag = 1 - (1 - 1e-3) ** (1 / 100)                                                # Zielgröße (b): Tagesmaximum aus 100 Containern, 0,999-Quantil
    assert p_tag == pytest.approx(1.0005e-5, rel=1e-4)
    assert S.mixture_level(p_tag) == pytest.approx(287.123, abs=5e-4)
    # Tagesmaximum-Quantile zweiter Weg: 100·ln(1 − sf(x)) = ln q per Bisektion auf dem Tagesmaximum selbst
    def tagesmax_quantil(q):
        lo, hi = -2.0, 14.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 100 * math.log1p(-S.mixture_sf(math.exp(mid))) < math.log(q):
                lo = mid
            else:
                hi = mid
        return math.exp(0.5 * (lo + hi))
    assert tagesmax_quantil(0.999) == pytest.approx(287.123, abs=5e-4)
    assert tagesmax_quantil(0.99) == pytest.approx(155.128, abs=5e-4)                  # „155,128 bei T = 100“: einmal in 100 Tagen


def test_pegel_der_quellen_aus_f():
    assert S.true_level("Normal", 1 / 36000) == pytest.approx(4.0309, abs=5e-5)
    assert S.true_level("Exponential", 1 / 36000) == pytest.approx(10.4913, abs=5e-5)
    assert S.true_level("Vehikel B", 1 / 36000) == pytest.approx(221.36, abs=5e-3)
    assert S.true_level("Pareto-Verteilung", 1 / 36000) == pytest.approx(36000 ** 0.25, rel=1e-14)
    assert -math.expm1(-1.0 / S.true_level("Standard-Fréchet", 1 / 36000)) == pytest.approx(1 / 36000, rel=1e-12)     # Probe an der Überlebensfunktion 1 − exp(−1/x)


def test_umkehrtabelle_gegen_wurzelsuche():
    """Die eingefrorene Umkehrtabelle (Interpolation in ln sf) gegen die direkte Wurzelsuche auf der exakten Überlebensfunktion: relativ innerhalb 1e-6 für p = 1e-2 bis 1e-8."""
    for p in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8):
        assert float(S.isf("Vehikel B", np.array([p]))[0]) == pytest.approx(S.mixture_level(p), rel=1e-6)


def test_umkehr_ist_die_umkehrung_der_ueberlebensfunktion_je_quelle():
    sp = pytest.importorskip("scipy.stats")
    v = np.array([0.5, 0.1, 1e-3, 1e-5])
    assert np.allclose(S.isf("Normal", v), sp.norm.isf(v), rtol=1e-12)
    assert np.allclose(S.isf("Exponential", v), sp.expon.isf(v), rtol=1e-12)
    assert np.allclose(S.isf("Pareto-Verteilung", v), sp.pareto.isf(v, 4.0), rtol=1e-12)
    assert np.allclose(S.isf("Standard-Fréchet", v), sp.invweibull.isf(v, 1.0), rtol=1e-12)
