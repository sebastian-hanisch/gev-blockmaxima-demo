"""Kennzahlen, Läufe und Verdikt-Logik. Jedes Verdikt ist eine reine Funktion `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`.

Zähleinheiten: m = Block-Maxima je Anpassung, b = Beobachtungen je Block, Läufe = unabhängige Wiederholungen einer ganzen Reihe (jede Streuung ist eine Streuung über Läufe).
V = geschätzter Pegel / wahrer Pegel; lnRMSE = Wurzel des Mittels von (ln V)²; es gilt lnRMSE² = Bias² + SD² (SD mit Nenner R) in jeder Zelle."""

import math
from dataclasses import dataclass

import numpy as np

import gvb_constants as C
import gvb_fit as F
import gvb_gev as G
import gvb_reference as REF
import gvb_scenario as SC
import gvb_sources as S
from gvb_texts import de, pct


@dataclass(frozen=True)
class Settings:
    quelle: str = C.DEFAULTS["quelle_sel"]
    b: int = C.DEFAULTS["b_sl"]
    verfahren: str = C.DEFAULTS["verfahren_sel"]
    seed: int = C.DEFAULT_SEED
    xi: float = C.DEFAULTS["xi_sl"]
    m: int = C.DEFAULTS["m_sl"]
    runs: int = C.DEFAULT_RUNS
    klammer: bool = C.DEFAULTS["klammer_tg"]
    quelle4: str = C.DEFAULTS["quelle4_sel"]


def settings_from_state(state):
    """Settings aus einem Session-State-ähnlichen Wörterbuch (fehlende Schlüssel: Standard)."""
    g = lambda k: state.get(k, C.DEFAULTS[k])
    return Settings(g("quelle_sel"), int(g("b_sl")), g("verfahren_sel"), int(g("seed_input")) if "seed_input" in state else C.DEFAULT_SEED,
                    float(g("xi_sl")), int(g("m_sl")), int(g("runs_sl")), bool(g("klammer_tg")), g("quelle4_sel"))


# ---------------------------------------------------------------------------------------------------
# Experiment 1 und 4 (live): eine Reihe anpassen
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Analysis:
    source: str
    b: int
    m: int
    n_obs: int                  # Beobachtungen der Reihe (bei „Exakte GEV“: 0, es gibt keine Einzelbeobachtungen)
    maxima: tuple
    p: float                    # Überschreitungswahrscheinlichkeit je Block, zu der der Pegel gehört
    truth: float                # wahrer Pegel (exakt aus F^b bzw. aus der GEV der Quelle)
    xi_true: float | None       # nur bei „Exakte GEV“
    fits: dict                  # Verfahren -> Fit oder None
    errors: dict                # Verfahren -> Meldung, wenn die Anpassung nicht möglich war
    ratios: dict                # Verfahren -> V

    def v(self, method):
        return self.ratios.get(method)


def analyse(settings: Settings) -> Analysis:
    """Eine Reihe erzeugen und mit allen drei Verfahren anpassen. Quelle „Exakte GEV“: m Blöcke GEV(0, 1, ξ), Ziel p = 1/(2m) je Block; sonst N = 18 000 Beobachtungen in Blöcken zu b,
    Ziel q = 1/(2N) je Beobachtung (p_b = 1 - (1 - q)^b je Block)."""
    ser = SC.make_series(settings.quelle, settings.b, settings.seed, xi=settings.xi, m=settings.m)
    x = np.array(ser.maxima)
    if settings.quelle == "Exakte GEV":
        p = 1.0 / (2 * ser.m)
        truth = float(G.pegel(p, 0.0, 1.0, settings.xi))
        n_obs, xi_true, b = 0, settings.xi, 0
    else:
        p = S.block_exceedance(ser.b, C.Q_TARGET)
        truth = S.true_level(settings.quelle, C.Q_TARGET)
        n_obs, xi_true, b = C.N_OBS, None, ser.b
    fits, errors, ratios = {}, {}, {}
    for method in F.METHODS:
        try:
            f = F.fit(method, x, settings.klammer)
            fits[method] = f
            ratios[method] = f.level(p) / truth
        except G.FitError as e:
            fits[method], errors[method] = None, str(e)
    return Analysis(settings.quelle, b, ser.m, n_obs, ser.maxima, p, truth, xi_true, fits, errors, ratios)


def true_curve(a: Analysis, ps):
    """Wahre Pegel zu den Block-Überschreitungswahrscheinlichkeiten ps (aus F^b bzw. der GEV der Quelle)."""
    ps = np.asarray(ps, float)
    if a.source == "Exakte GEV":
        return G.pegel(ps, 0.0, 1.0, a.xi_true)
    q = -np.expm1(np.log1p(-ps) / a.b)           # 1 - (1 - p)^(1/b)
    return np.array([S.true_level(a.source, float(t)) for t in q])


# ---------------------------------------------------------------------------------------------------
# Experimente 2 und 3 (live und eingefroren): viele Läufe mit exakten GEV(0, 1, ξ)-Blöcken
# ---------------------------------------------------------------------------------------------------
COLS = ("ml_xi", "ml_rand", "ml_endpunkt", "ml_pegelfehler", "lm_xi", "lm_endpunkt", "lm_pegelfehler", "lmo_xi", "lmo_endpunkt", "lmo_pegelfehler")


def fit_cell_runs(xi, m, run_ids, seed):
    """Läufe einer Zelle (ξ, m): je Lauf m exakte GEV(0, 1, ξ)-Blöcke; Spalten wie COLS (ML; L-Momente mit Klammer; L-Momente ohne Klammer); Pegel zu p = 0,01, Fehler in σ."""
    ztrue = float(G.pegel(C.P_HUNDERT, 0.0, 1.0, xi))
    out = []
    for r in run_ids:
        x = SC.exact_gev_blocks(SC.rng_for(seed, int(round(xi * 100)) + 200, m, r), m, xi=xi)
        row = [np.nan] * 10
        try:
            f = F.fit_ml(x)
            row[0], row[1], row[2], row[3] = f.xi, float(f.edge), float(f.endpoint < x.max() - 1e-9), f.level(C.P_HUNDERT) - ztrue
        except G.FitError:
            pass
        for off, klammer in ((4, True), (7, False)):
            try:
                mu, s, xl = G.fit_lmoments(x, klammer)
                row[off], row[off + 1], row[off + 2] = xl, float(G.endpoint(mu, s, xl) < x.max() - 1e-9), float(G.pegel(C.P_HUNDERT, mu, s, xl)) - ztrue
            except G.FitError:
                pass
        out.append(row)
    return np.array(out, float)


def robust(a):
    """Mittel, SD (Nenner R - 1), Median, robuste Streuung IQR/1,349, Kleinstes, Größtes: die SD allein ist bei kleinem m ein Ausreißer-Maß."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        raise G.FitError("Zu wenige gültige Läufe.")
    q25, q75 = np.quantile(a, [0.25, 0.75])
    return {"n": int(a.size), "mittel": float(a.mean()), "sd": float(a.std(ddof=1)), "median": float(np.median(a)), "iqr_sd": float((q75 - q25) / 1.349),
            "min": float(a.min()), "max": float(a.max())}


def share(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    p = float(a.mean()) if a.size else float("nan")
    return {"anteil": p, "se": math.sqrt(p * (1 - p) / max(a.size, 1)), "n": int(a.size)}


def cell_summary(A):
    """Statistik einer Zelle aus dem Feld der Läufe (Spalten COLS)."""
    err = lambda col: {"median": float(np.nanmedian(A[:, col])), "mittel_abs": float(np.nanmean(np.abs(A[:, col])))}
    return {
        "laeufe": int(A.shape[0]),
        "ml": {"xi": robust(A[:, 0]), "rand": share(A[:, 1]), "endpunkt_unter_max": share(A[:, 2]), "pegelfehler": err(3)},
        "lm_klammer": {"xi": robust(A[:, 4]), "endpunkt_unter_max": share(A[:, 5]), "pegelfehler": err(6)},
        "lm_ohne": {"xi": robust(A[:, 7]), "endpunkt_unter_max": share(A[:, 8])},
    }


def live_cell(settings: Settings):
    """Live gerechnete Zelle (ξ, m) mit `runs` Läufen; derselbe Lauf-Code wie in der eingefrorenen Tabelle, anderer Zufalls-Seed (der aus den Reglern)."""
    A = fit_cell_runs(settings.xi, settings.m, range(settings.runs), settings.seed)
    return cell_summary(A), A


def lm_key(klammer):
    return "lm_klammer" if klammer else "lm_ohne"


# ---------------------------------------------------------------------------------------------------
# Experiment 4: Zerlegung je Blockgröße aus der eingefrorenen Tabelle, Populations-Bias live
# ---------------------------------------------------------------------------------------------------
def block_rows(source):
    """Zeilen der Blockgrößen-Tabelle (eingefroren) für eine Quelle: je b ML und Gumbel mit Median V, Bias, SD, lnRMSE."""
    rows = []
    for b in C.BLOCKS:
        c = REF.block_cell(source, b)
        if c is None:
            return []
        rows.append(c)
    return rows


def best_block(rows, key="ml"):
    """Zelle mit dem kleinsten lnRMSE der Methode `key` (ml oder gumbel)."""
    return min(rows, key=lambda c: c[key]["ln_rmse"])


# ---------------------------------------------------------------------------------------------------
# Verdikte (rein)
# ---------------------------------------------------------------------------------------------------
def _band(v):
    """Wort zur Abweichung des Pegel-Verhältnisses V von 1 (nur für Verdikt-Art und Satz, kein Maß)."""
    if v is None or not math.isfinite(v):
        return "warning"
    return "success" if abs(math.log(v)) < 0.1 else ("info" if abs(math.log(v)) < 0.3 else "warning")


def verdict_fit(a: Analysis, method: str, ref: dict | None):
    """Verdikt zu Experiment 1: dieses Verhältnis V einer Reihe, daneben die eingefrorene Streuung über 1 000 Läufe (nur für Normal, Exponential, Vehikel B und ML/Gumbel)."""
    v = a.v(method)
    if v is None:
        return "warning", f"Mit dem Verfahren „{method}“ ist bei m = {a.m} Block-Maxima keine Anpassung möglich: {a.errors.get(method, 'unbekannter Grund')}"
    wo = f"{a.source}, b = {a.b} (m = {a.m})" if a.source != "Exakte GEV" else f"Exakte GEV mit ξ = {de(a.xi_true, 1)} (m = {a.m})"
    wahr = "wahrer Pegel der erzeugenden GEV" if a.source == "Exakte GEV" else "wahrer Pegel aus F^b"
    text = f"{wo}: der geschätzte Pegel ist das **{de(v, 3)}-Fache** des wahren ({method}; eine Reihe, V = Schätzwert / {wahr})."
    if ref is not None and method in ("ML", "Gumbel"):
        key = "ml" if method == "ML" else "gumbel"
        r = ref[key]
        text += (f" Über {T_n(r)} Läufe dieser Quelle und Blockgröße (eingefroren): V im Median {de(r['median'], 3)} "
                 f"(Q10/Q90 {de(r['q10'], 3)}/{de(r['q90'], 3)}); das ist die Streuung bei dieser Blockgröße, kein Streuband gegen T und m.")
        if method == "ML" and ref.get("gumbel") is not None:
            text += f" Gumbel mit ξ = 0 fest: Median {de(ref['gumbel']['median'], 3)}."
    return _band(v), text


def T_n(r):
    return str(r["n_gueltig"])


def verdict_xi_spread(live: dict, ref: dict | None, xi: float, m: int, klammer: bool):
    """Verdikt zu Experiment 2: Streuung von ξ̂ über die Läufe, ML und L-Momente nebeneinander; der Vergleich der beiden steht nur für die eingefrorenen 1 000 Läufe (live ist er Zufall);
    bei m = 10 steht die SD von ML neben Median und IQR/1,349."""
    ml, lm = live["ml"]["xi"], live[lm_key(klammer)]["xi"]
    n = live["laeufe"]
    kl = "mit Klammer ξ̂ ≥ −0,999" if klammer else "ohne Klammer"
    text = (f"ξ = {de(xi, 1)}, m = {m}, {n} Läufe live: ML hat SD von ξ̂ {de(ml['sd'], 3)} (Median {de(ml['median'], 3)}, IQR/1,349 {de(ml['iqr_sd'], 3)}), "
            f"die L-Momente ({kl}) {de(lm['sd'], 3)} (Median {de(lm['median'], 3)}, IQR/1,349 {de(lm['iqr_sd'], 3)}).")
    if ref is not None:
        r, rl = ref["ml"]["xi"], ref[lm_key(klammer)]["xi"]
        text += (f" Eingefroren ({ref['laeufe']} Läufe): ML SD {de(r['sd'], 3)} (IQR/1,349 {de(r['iqr_sd'], 3)}), L-Momente {de(rl['sd'], 3)} (IQR/1,349 {de(rl['iqr_sd'], 3)}); "
                 f"die kleinere SD hat dort {'ML' if r['sd'] < rl['sd'] else 'die L-Momente'}.")
    else:
        text += " Für diese Kombination (ξ, m) gibt es keine eingefrorene Zelle, ein Vergleich der beiden SD aus so wenigen Läufen wäre Zufall."
    if m <= 10:
        text += (" Bei m = 10 hängt die SD von ML an wenigen Läufen mit riesigem ξ̂ (der Optimierer hat keine obere Grenze für ξ): "
                 f"das größte ξ̂ liegt hier bei {de(ml['max'], 1)}; Median und IQR/1,349 lesen.")
    return "info", text


def lm_endpoint_range(xis=(-0.6, -0.9), ms=(20, 50, 100)):
    """Spanne des Anteils „Endpunkt unter dem größten Wert“ der L-Momente über die eingefrorenen Zellen (ξ-Raster und m genannt): (mit Klammer min, max, ohne Klammer min, max)."""
    mit, ohne = [], []
    for xi in xis:
        for m in ms:
            c = REF.fit_cell(xi, m)
            mit.append(c["lm_klammer"]["endpunkt_unter_max"]["anteil"])
            ohne.append(c["lm_ohne"]["endpunkt_unter_max"]["anteil"])
    return min(mit), max(mit), min(ohne), max(ohne)


def verdict_edge(live: dict, xi: float, m: int, klammer: bool):
    """Verdikt zu Experiment 3: ML am Rand ξ = −1 (Optimierer) gegen L-Momente mit Endpunkt unter dem größten Wert; benennt die Klammer."""
    n = live["laeufe"]
    rand = live["ml"]["rand"]["anteil"]
    emp = live[lm_key(klammer)]["endpunkt_unter_max"]["anteil"]
    kl = "mit der Klammer ξ̂ ≥ −0,999" if klammer else "ohne Klammer (ξ̂ bis −60 zugelassen)"
    text = (f"ξ = {de(xi, 1)}, m = {m}, {n} Läufe: ML klebt in {pct(rand)} der Läufe an der Nebenbedingung ξ = −1 (Wahl dieser Demo), "
            f"die L-Momente enden {kl} in {pct(emp)} der Läufe unter dem größten beobachteten Wert (unmögliche Anpassung; bei ML nie, die Likelihood verbietet es).")
    if xi <= -1.0:
        text += " Der Fall ξ ≤ −1 liegt am oder jenseits des Randes des ML-Bereichs: die Likelihood ist dort unbeschränkt, ML taugt nicht als Anpassungsbeispiel."
    lo1, hi1, lo2, hi2 = lm_endpoint_range()
    if xi < -0.5 and xi > -1.0:
        text += (f" Eingefroren (1 000 Läufe je Zelle) liegt der Anteil unmöglicher L-Moment-Anpassungen für ξ = −0,6 und ξ = −0,9 bei m = 20, 50 und 100 zwischen {pct(lo1)} und {pct(hi1)} mit Klammer "
                 f"({pct(lo2)} bis {pct(hi2)} ohne; die Ränder der Spanne sind Zufallsextreme über sechs Zellen und gelten auf etwa ± 2 Punkte); die Spanne gehört zu diesem ξ-Raster (bei ξ = −0,8, m = 100: {pct(REF.fit_cell(-0.8, 100)['lm_klammer']['endpunkt_unter_max']['anteil'])} mit Klammer), "
                 "sie wächst mit m und liegt im eingefrorenen Raster (m ≤ 100) bei bis zu einem Drittel; für m = 500 gibt es keine eingefrorene Zelle, der Anteil kann dort höher liegen (er steht oben live).")
    return ("warning" if emp > 0.05 or rand > 0.05 else "info"), text


def verdict_blocksize(source: str, rows: list, pop: dict | None):
    """Verdikt zu Experiment 4 aus der eingefrorenen Tabelle (1 000 Läufe je Zelle)."""
    if not rows:
        return "info", "Für diese Quelle gibt es keine eingefrorene Blockgrößen-Tabelle."
    best, bestg = best_block(rows, "ml"), best_block(rows, "gumbel")
    first, last = rows[0], rows[-1]
    t = (f"{source}: kleinster lnRMSE der freien GEV (ML) bei b = {best['b']} (m = {best['m']}): {de(best['ml']['ln_rmse'], 3)}; "
         f"b = {first['b']} hat Bias {de(first['ml']['bias_ln'], 3)} bei SD {de(first['ml']['sd_ln'], 3)} (in ln, Läufe), b = {last['b']} Bias {de(last['ml']['bias_ln'], 3)} bei SD {de(last['ml']['sd_ln'], 3)}.")
    n_better = sum(1 for c in rows if c["gumbel"]["ln_rmse"] < c["ml"]["ln_rmse"])
    t += f" Gumbel mit ξ = 0 fest hat den kleineren lnRMSE in {n_better} von {len(rows)} Blockgrößen (kleinster {de(bestg['gumbel']['ln_rmse'], 3)} bei b = {bestg['b']}); das gilt nur, wenn ξ = 0 vorab richtig bekannt ist."
    if best["b"] in (rows[0]["b"], rows[-1]["b"]):
        t += f" Das globale Minimum liegt am Rand des gemessenen Bereichs (b = {best['b']}); kleinere Blöcke sind nicht gemessen."
        for i in range(1, len(rows) - 1):
            r = [rows[j]["ml"]["ln_rmse"] for j in (i - 1, i, i + 1)]
            if r[1] < r[0] and r[1] < r[2]:
                t += (f" Daneben hat die Kurve ein lokales Minimum bei b = {rows[i]['b']} (lnRMSE {de(r[1], 3)}, die Nachbarn b = {rows[i - 1]['b']} und b = {rows[i + 1]['b']} haben "
                      f"{de(r[0], 3)} und {de(r[2], 3)}).")
    return "info", t + f" (eingefroren, {best['ml']['n_gueltig']} Läufe je Zelle)"


def verdict_positions(rows: list):
    """Verdikt zu Experiment 5 (Tabelle der Handverfahren, 2 000 Läufe je Zelle, ξ = 0, m = 20 und 50)."""
    def pick(xi, m, name):
        return next(r for r in rows if r["xi"] == xi and r["m"] == m and r["verfahren"].startswith(name))
    wb = pick(0.0, 50, "Gumbel-Regression, Positionen nach Weibull")
    gr = pick(0.0, 50, "Gumbel-Regression, Gringorten")
    gm = pick(0.0, 50, "Gumbel-ML")
    wrong = pick(0.2, 50, "Gumbel-ML")
    return "info", (f"Wer ξ nicht kennt, zahlt Streuung; wer es festhält, zahlt Bias, sobald es falsch ist: bei ξ = 0 und m = 50 hat Gumbel-ML RMSE {de(gm['rmse'], 2)} σ, bei wahrem ξ = 0,2 "
                    f"(ebenfalls m = 50, 100er-Wert) aber Bias {de(wrong['bias'], 2)} σ (zu niedrig in {pct(wrong['anteil_unter_wahr'])} der Läufe). Die Gumbel-Regression mit Positionen nach Weibull ist bei ξ = 0 um {de(wb['bias'], 2)} σ verzerrt, "
                    f"mit Gringorten um {de(gr['bias'], 3)} σ (Standardfehler rund {de(gr['sd'] / math.sqrt(2000), 3)} σ, von null nicht zu unterscheiden) bei RMSE {de(gr['rmse'], 2)} σ (2 000 Läufe).")
