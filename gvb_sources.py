"""Quellen der Beobachtungen: alle ausgedacht, alle mit exakter Verteilung (die Wahrheit ist F^b; nur die Quelle „Exakte GEV“ in gvb_scenario hat kein F, dort ist es die erzeugende GEV).

Jede Quelle liefert die Umkehrung der Überlebensfunktion `isf(v)` (v = P(X > x)): ein Block-Maximum von b Beobachtungen ist `isf(min v)` über die b gleichverteilten
v des Blocks (das Maximum der Beobachtungen gehört zum Minimum der v). Für Vehikel B ist die Umkehrung die eingefrorene Tabelle `gvb_mixture_inv.json` (Interpolation
in ln sf), der wahre Pegel kommt dagegen aus der exakten Überlebensfunktion der Mischung (Bisektion, `math.erfc`).

Vehikel B = die Verweilzeit-Mischung der `verweilzeit-demo` als EXAKTE Aufzählung (gemeinsame Definition der Linie, 1 224 Komponenten (Gewicht, μ, s) der logarithmierten
Verweilzeit, Quelle und Stand in `gvb_mixture_quelle.json`; `gvb_mixture.json` ist eine unveränderte Kopie, Kopier-Test in `tests/test_copies.py`)."""

import json
import math
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np

import gvb_constants as C

HERE = Path(__file__).resolve().parent
_ND = NormalDist()
V_MAX = 1.0 - 2.0 ** -53       # v = 1 - u liegt in (0, 1]; v = 1 hieße u = 0 und hat keine endliche Umkehrung für Normal


@lru_cache(maxsize=1)
def mixture_components():
    """(Gewichte, μ, s) der 1 224 Komponenten als numpy-Felder."""
    d = json.loads((HERE / "gvb_mixture.json").read_text(encoding="utf-8"))
    a = np.array(d["komponenten"], float)
    return a[:, 0], a[:, 1], a[:, 2]


@lru_cache(maxsize=1)
def mixture_inverse_table():
    """Eingefrorene Umkehrtabelle: ln x (aufsteigend, gleichmäßiger Schritt) und ln sf(x) (fallend)."""
    d = json.loads((HERE / "gvb_mixture_inv.json").read_text(encoding="utf-8"))
    lnx = d["lnx_min"] + d["schritt"] * np.arange(len(d["lnsf"]))
    return lnx, np.array(d["lnsf"], float)


@lru_cache(maxsize=1)
def _mix_lists():
    w, mu, s = mixture_components()
    return tuple(w.tolist()), tuple(mu.tolist()), tuple((s * math.sqrt(2)).tolist())


def mixture_sf(x):
    """Exakte Überlebensfunktion P(D > x) der Mischung (x > 0), reine Standardbibliothek je Komponente."""
    w, mu, s2 = _mix_lists()
    lx = math.log(x)
    return math.fsum(wi * 0.5 * math.erfc((lx - m) / s) for wi, m, s in zip(w, mu, s2))


def mixture_level(p):
    """Pegel x mit P(D > x) = p, exakt per Bisektion auf ln x (0 < p < 1)."""
    lo, hi = -3.0, 14.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mixture_sf(math.exp(mid)) > p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return math.exp(0.5 * (lo + hi))


def isf(source, v, alpha=C.PARETO_ALPHA):
    """Umkehrung der Überlebensfunktion: x mit P(X > x) = v, für ein Feld v in (0, 1]. `source` ohne „Exakte GEV“ (dort gibt es keine Einzelbeobachtung)."""
    v = np.minimum(np.asarray(v, float), V_MAX)
    if source == "Exponential":
        return -np.log(v)
    if source == "Normal":
        return np.array([-_ND.inv_cdf(float(t)) for t in v.ravel()]).reshape(v.shape)
    if source == "Pareto-Verteilung":
        return np.power(v, -1.0 / alpha)
    if source == "Standard-Fréchet":
        return -1.0 / np.log1p(-v)
    if source == "Vehikel B":
        lnx, lnsf = mixture_inverse_table()
        return np.exp(np.interp(np.log(v), lnsf[::-1], lnx[::-1]))
    raise ValueError(f"Quelle {source!r} hat keine Einzelbeobachtungen")


def true_level(source, q, alpha=C.PARETO_ALPHA):
    """Wahrer Pegel: der Wert, den EINE Beobachtung mit Wahrscheinlichkeit q überschreitet (exakt aus F, keine GEV-Annahme)."""
    if source == "Exponential":
        return -math.log(q)
    if source == "Normal":
        return -_ND.inv_cdf(q)
    if source == "Pareto-Verteilung":
        return q ** (-1.0 / alpha)
    if source == "Standard-Fréchet":
        return -1.0 / math.log1p(-q)
    if source == "Vehikel B":
        return mixture_level(q)
    raise ValueError(f"Quelle {source!r} hat keinen Beobachtungspegel")


def block_exceedance(b, q):
    """p_b = 1 - (1 - q)^b: Wahrscheinlichkeit, dass das Maximum eines Blocks den Wert überschreitet, den eine Beobachtung mit q überschreitet."""
    return -math.expm1(b * math.log1p(-q))


def tail_index(source, alpha=C.PARETO_ALPHA):
    """ξ der Quelle im Grenzwert (Normal 0 im Sinn des Gumbel-Bereichs, aber ohne Grenzwert des Bias; Vehikel B: lokales ξ der Mischung nur als Näherung, hier nicht gesetzt)."""
    return {"Exponential": 0.0, "Normal": 0.0, "Pareto-Verteilung": 1.0 / alpha, "Standard-Fréchet": 1.0}.get(source)


def survival_is_known_limit(source):
    """Hat die Quelle einen Grenzwert des Populations-Bias für m → ∞ (ξ_pop ≥ 0)? Normal: nein (hängt am Abschneiden des Tails, README Befund E)."""
    return source in ("Exponential", "Pareto-Verteilung", "Standard-Fréchet", "Vehikel B")
