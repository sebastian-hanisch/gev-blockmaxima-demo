"""Erzeugt die eingefrorenen Daten der Demo und prüft sie.

  mixture-inv   Umkehrtabelle der Verweilzeit-Mischung (`gvb_mixture_inv.json`, aus der Kopie `gvb_mixture.json`, mit `math.erfc`)
  fit           Tabelle A: ML und L-Momente bei kleinem m (1 000 Läufe je Zelle), in `gvb_tables.json`
  block         Tabelle B: Blockgröße je Quelle (1 000 Läufe je Zelle), in `gvb_tables.json`
  positions     Tabelle C: Handverfahren (2 000 Läufe je Zelle), in `gvb_tables.json`
  all           alle vier nacheinander
Aufruf (aus dem Demo-Ordner):  OMP_NUM_THREADS=1 <python> -X utf8 tools/gvb_make_tables.py <Teil> [--check] [--prozesse N]
  --check   rechnet je Teil einen kleinen Ausschnitt neu und prüft als Band (± 4 Standardfehler) gegen die eingefrorene Datei; schreibt nichts
  --lang    mit --check: den ganzen Satz neu rechnen (mehrere Minuten) und gegen die Datei vergleichen
Die Zufalls-Seeds sind feste Zahlen (BASIS_*) und andere als die der Vormessung (die benutzte numpy `default_rng`)."""

import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import gvb_constants as C  # noqa: E402
import gvb_evaluation as EV  # noqa: E402
import gvb_fit as F  # noqa: E402
import gvb_gev as G  # noqa: E402
import gvb_scenario as SC  # noqa: E402
import gvb_sources as S  # noqa: E402

STAND = "2026-11-20"
BASIS_FIT, BASIS_BLOCK, BASIS_POS = 710001, 720002, 730003
R_FIT, R_BLOCK, R_POS = 1000, 1000, 2000
TABLES = ROOT / "gvb_tables.json"
INV = ROOT / "gvb_mixture_inv.json"
LNX_MIN, LNX_MAX, SCHRITT = -3.0, 10.0, 0.002


def P(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------
# Umkehrtabelle der Mischung
# ---------------------------------------------------------------------------------------------------
def make_inv():
    n = int(round((LNX_MAX - LNX_MIN) / SCHRITT)) + 1
    lnsf = []
    for i in range(n):
        s = S.mixture_sf(math.exp(LNX_MIN + SCHRITT * i))
        lnsf.append(round(math.log(s), 12))
    return {
        "beschreibung": "Umkehrtabelle der Verweilzeit-Mischung (Vehikel B): ln sf an einem gleichmäßigen Gitter in ln x; Interpolation linear in ln sf",
        "quelle": "erzeugt mit tools/gvb_make_tables.py mixture-inv aus gvb_mixture.json (Kopie der gemeinsamen Tabelle von Stück 4, Linie Extremwert) mit math.erfc",
        "stand": STAND, "lnx_min": LNX_MIN, "schritt": SCHRITT, "lnx_max": LNX_MAX, "n": n, "lnsf": lnsf,
    }


def teil_mixture_inv(check):
    obj = make_inv()
    if check:
        alt = json.loads(INV.read_text(encoding="utf-8"))
        d = max(abs(a - b) for a, b in zip(alt["lnsf"], obj["lnsf"]))
        assert alt["n"] == obj["n"] and d < 1e-9, f"Umkehrtabelle weicht ab ({d:.1e})"
        P(f"mixture-inv --check: gleich (größte Abweichung {d:.1e})")
    else:
        INV.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8", newline="\n")
        P(f"mixture-inv geschrieben: {INV.name}, {obj['n']} Punkte")


# ---------------------------------------------------------------------------------------------------
# Hilfen für die Statistik der Zellen
# ---------------------------------------------------------------------------------------------------
def run_fit_cell(args):
    """Eine Zelle (ξ, m): Läufe mit exakten GEV(0, 1, ξ)-Blöcken; derselbe Code wie die Live-Rechnung der App (gvb_evaluation.fit_cell_runs)."""
    xi, m, runs, basis = args
    return EV.fit_cell_runs(xi, m, runs, basis)


def teil_fit(check, lang, nproc):
    grid = [(xi, m) for xi in C.XI_GRID for m in C.M_GRID]
    if check and not lang:
        grid = [(0.0, 50), (-0.9, 20)]
    runs_all = list(range(R_FIT if (lang or not check) else 200))
    tasks = [(xi, m, runs_all[k:k + 50], BASIS_FIT) for xi, m in grid for k in range(0, len(runs_all), 50)]
    t0 = time.perf_counter()
    with Pool(nproc) as pool:
        res = pool.map(run_fit_cell, tasks, chunksize=1)
    by = {}
    for (xi, m, _, _), a in zip(tasks, res):
        by.setdefault((xi, m), []).append(a)
    cells = []
    for xi, m in grid:
        s = EV.cell_summary(np.vstack(by[(xi, m)]))
        s.update({"xi": xi, "m": m})
        cells.append(s)
    P(f"fit: {len(grid)} Zellen, {len(runs_all)} Läufe je Zelle, {time.perf_counter() - t0:.0f} s")
    return cells


# ---------------------------------------------------------------------------------------------------
# Tabelle B: Blockgröße
# ---------------------------------------------------------------------------------------------------
SRC_IDX = {"Normal": 1, "Exponential": 2, "Vehikel B": 3}


def run_block_runs(args):
    """Läufe einer Quelle: je Lauf eine Reihe aus N gleichverteilten Zahlen, je Blockgröße ML-GEV und Gumbel an die Block-Maxima. Spalten je b: V_ML, V_Gumbel, ξ̂, Rand."""
    source, runs, basis = args
    out = []
    for r in runs:
        v = SC.series_v(SC.rng_for(basis, SRC_IDX[source], r))
        row = []
        for b in C.BLOCKS:
            x = S.isf(source, SC.block_minima_of_v(v, b))
            pb = S.block_exceedance(b, C.Q_TARGET)
            zt = S.true_level(source, C.Q_TARGET)
            try:
                f = F.fit_ml(x)
                vg = f.level(pb) / zt
                xi, edge = f.xi, float(f.edge)
            except G.FitError:
                vg, xi, edge = np.nan, np.nan, np.nan
            try:
                mu, s = F.gumbel_ml(x)
                vgum = float(G.pegel(pb, mu, s, 0.0)) / zt
            except G.FitError:
                vgum = np.nan
            row += [vg, vgum, xi, edge]
        out.append(row)
    return np.array(out, float)


def block_summary(A, i):
    vml, vgu, xi, edge = A[:, 4 * i], A[:, 4 * i + 1], A[:, 4 * i + 2], A[:, 4 * i + 3]

    def zerleg(v):
        ok = np.isfinite(v) & (v > 0)
        lv = np.log(v[ok])
        n = int(ok.sum())
        return {
            "n_gueltig": n, "median": float(np.median(v[ok])), "q10": float(np.quantile(v[ok], 0.1)), "q90": float(np.quantile(v[ok], 0.9)),
            "anteil_unter_1": float(np.mean(v[ok] < 1)), "bias_ln": float(lv.mean()), "sd_ln": float(lv.std(ddof=0)),
            "ln_rmse": float(math.sqrt(np.mean(lv ** 2))), "se_bias": float(lv.std(ddof=0) / math.sqrt(n)),
        }
    return {"ml": zerleg(vml), "gumbel": zerleg(vgu), "xi_mittel": float(np.nanmean(xi)), "ml_rand": float(np.nanmean(edge))}


def teil_block(check, lang, nproc):
    sources = list(SRC_IDX)
    n_runs = R_BLOCK if (lang or not check) else 40
    t0 = time.perf_counter()
    out = {}
    for source in (sources if (lang or not check) else ["Normal"]):
        tasks = [(source, list(range(k, min(n_runs, k + 5))), BASIS_BLOCK) for k in range(0, n_runs, 5)]
        with Pool(nproc) as pool:
            res = pool.map(run_block_runs, tasks, chunksize=1)
        A = np.vstack(res)
        out[source] = {
            "wahrer_pegel": S.true_level(source, C.Q_TARGET), "laeufe": int(A.shape[0]),
            "zellen": {str(b): {**block_summary(A, i), "b": b, "m": C.N_OBS // b} for i, b in enumerate(C.BLOCKS)},
        }
    P(f"block: {len(out)} Quellen, {n_runs} Läufe je Zelle, {time.perf_counter() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------------------------------
# Tabelle C: Handverfahren (Fehler des 100er-Werts in σ)
# ---------------------------------------------------------------------------------------------------
POS_METHODS = ("ML-GEV", "L-Momente", "Gumbel-ML (ξ = 0 fest)", "Gumbel-Regression, Positionen nach Weibull", "Gumbel-Regression, Gringorten",
               "GEV-Kleinste-Quadrate auf Positionen nach Weibull")
POS_CELLS = (  # (ξ, m, Verfahrensindizes)
    (0.0, 20, (0, 1, 2, 3, 4)), (0.0, 50, (0, 1, 2, 3, 4, 5)), (0.2, 50, (2,)), (-0.2, 50, (2,)),
)


def gev_ls_weibull(x, p0):
    """GEV-Anpassung durch Kleinste Quadrate der geordneten Werte gegen die Quantile an den Positionen nach Weibull i/(m + 1), ungewichtet (nach V1 Abschnitt 5)."""
    xs = np.sort(x)
    m = xs.size
    pos = np.arange(1, m + 1) / (m + 1)

    def f(th):
        if th[1] <= 0 or abs(th[2]) > 0.9:
            return G.HUGE
        return float(np.sum((xs - G.gev_ppf(pos, th[0], th[1], th[2])) ** 2))
    th, fv = F.nm_minimize(f, np.array(p0, float), np.array([xs.std(), 0.1 * xs.std(), 0.1]), 1e-7, 1e-10, 1500)
    return th if fv < G.HUGE / 10 else None


def estimates(x):
    """Die sechs Schätzer des 100er-Werts aus einer Stichprobe (nan, wenn keine Anpassung)."""
    out = [np.nan] * 6
    xs = np.sort(x)
    m = x.size
    i = np.arange(1, m + 1)
    try:
        out[0] = F.fit_ml(x).level(C.P_HUNDERT)
    except G.FitError:
        pass
    try:
        mu, s, xi = G.fit_lmoments(x, True)
        out[1] = float(G.pegel(C.P_HUNDERT, mu, s, xi))
    except G.FitError:
        pass
    try:
        mu, s = F.gumbel_ml(x)
        out[2] = float(G.pegel(C.P_HUNDERT, mu, s, 0.0))
    except G.FitError:
        pass
    for k, pos in ((3, i / (m + 1)), (4, (i - 0.44) / (m + 0.12))):
        yv = -np.log(-np.log(pos))
        sl, ic = np.polyfit(yv, xs, 1)
        if sl > 0:
            out[k] = float(G.pegel(C.P_HUNDERT, ic, sl, 0.0))
    try:
        l1, l2, t3 = G.sample_lmoments(x)
        xi0 = float(np.clip(G.xi_from_tau3_hosking(t3), -0.5, 0.5))
        mu0, s0 = G.gev_from_lmoments(l1, l2, xi0)
        if not (s0 > 0 and math.isfinite(s0)):
            raise G.FitError("start")
    except (G.FitError, ValueError):
        s0, mu0, xi0 = float(x.std()) * 0.78, float(x.mean()) - 0.45 * float(x.std()), 0.0
    th = gev_ls_weibull(x, [mu0, s0, xi0])
    if th is not None:
        out[5] = float(G.pegel(C.P_HUNDERT, *th))
    return out


def run_pos_cell(args):
    xi, m, runs, basis = args
    out = []
    for r in runs:
        x = SC.exact_gev_blocks(SC.rng_for(basis, int(round(xi * 100)) + 200, m, r), m, xi=xi)
        out.append(estimates(x))
    return np.array(out, float)


def teil_positions(check, lang, nproc):
    n_runs = R_POS if (lang or not check) else 200
    cells = POS_CELLS if (lang or not check) else POS_CELLS[:1]
    out = []
    t0 = time.perf_counter()
    for xi, m, idx in cells:
        tasks = [(xi, m, list(range(k, min(n_runs, k + 100))), BASIS_POS) for k in range(0, n_runs, 100)]
        with Pool(nproc) as pool:
            A = np.vstack(pool.map(run_pos_cell, tasks, chunksize=1))
        zt = float(G.pegel(C.P_HUNDERT, 0.0, 1.0, xi))
        for j in idx:
            e = A[:, j] - zt
            ok = np.isfinite(e)
            e = e[ok]
            q25, q75 = np.quantile(e, [0.25, 0.75])
            out.append({
                "xi": xi, "m": m, "verfahren": POS_METHODS[j], "laeufe": n_runs, "n_gueltig": int(ok.sum()), "bias": float(e.mean()), "median": float(np.median(e)),
                "sd": float(e.std(ddof=1)), "rmse": float(math.sqrt(np.mean(e ** 2))), "iqr_sd": float((q75 - q25) / 1.349),
                "anteil_unter_wahr": float(np.mean(e < 0)), "se_bias": float(e.std(ddof=1) / math.sqrt(e.size)),
            })
    P(f"positions: {len(out)} Zeilen, {n_runs} Läufe je Zelle, {time.perf_counter() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------------------------------
# Schreiben und Prüfen
# ---------------------------------------------------------------------------------------------------
def meta(teil, laeufe, basis):
    import platform
    return {
        "quelle": f"gemessen mit tools/gvb_make_tables.py {teil} (SplitMix64 aus gvb_rng.py, kein scipy)", "stand": STAND, "laeufe_je_zelle": laeufe,
        "zufalls_seed_basis": basis, "python": platform.python_version(), "numpy": np.__version__, "plattform": platform.system(),
        "hinweis": "Zufallsanteile sind Messungen mit Standardfehler; Neu-Rechnen-Tests prüfen als Band",
    }


def load_tables():
    return json.loads(TABLES.read_text(encoding="utf-8")) if TABLES.exists() else {}


def write_part(key, laeufe, basis, data, teil):
    t = load_tables()
    t[key] = {"meta": meta(teil, laeufe, basis), "daten": data}
    TABLES.write_text(json.dumps(t, ensure_ascii=False, indent=None), encoding="utf-8", newline="\n")
    P(f"{key} nach {TABLES.name} geschrieben")


def check_fit(cells_new):
    t = load_tables()["fit"]["daten"]
    alt = {(c["xi"], c["m"]): c for c in t}
    n_ok = 0
    for c in cells_new:
        a = alt[(c["xi"], c["m"])]
        for who in ("ml", "lm_klammer"):
            sd_new, sd_alt = c[who]["xi"]["sd"], a[who]["xi"]["sd"]
            se = math.hypot(sd_new / math.sqrt(2 * (c["laeufe"] - 1)), sd_alt / math.sqrt(2 * (a["laeufe"] - 1)))
            assert abs(sd_new - sd_alt) <= C.BAND_SE * se + 1e-9 or c[who]["xi"]["iqr_sd"] > 0 and abs(c[who]["xi"]["iqr_sd"] - a[who]["xi"]["iqr_sd"]) < 0.08, \
                (c["xi"], c["m"], who, sd_new, sd_alt)
            n_ok += 1
        pn, pa = c["ml"]["rand"]["anteil"], a["ml"]["rand"]["anteil"]
        se = math.hypot(c["ml"]["rand"]["se"], a["ml"]["rand"]["se"])
        assert abs(pn - pa) <= C.BAND_SE * se + 1e-9, (c["xi"], c["m"], "Rand", pn, pa)
        n_ok += 1
    P(f"fit --check: {n_ok} Vergleiche im Band")


def check_block(new):
    t = load_tables()["block"]["daten"]
    n_ok = 0
    for source, d in new.items():
        for b, c in d["zellen"].items():
            a = t[source]["zellen"][b]
            for who in ("ml", "gumbel"):
                se = math.hypot(c[who]["sd_ln"] / math.sqrt(c[who]["n_gueltig"]), a[who]["sd_ln"] / math.sqrt(a[who]["n_gueltig"]))
                assert abs(c[who]["bias_ln"] - a[who]["bias_ln"]) <= C.BAND_SE * se + 1e-9, (source, b, who, c[who]["bias_ln"], a[who]["bias_ln"])
                n_ok += 1
    P(f"block --check: {n_ok} Vergleiche im Band")


def check_positions(new):
    t = load_tables()["positionen"]["daten"]
    alt = {(r["xi"], r["m"], r["verfahren"]): r for r in t}
    n_ok = 0
    for r in new:
        a = alt[(r["xi"], r["m"], r["verfahren"])]
        pa, pr = a["anteil_unter_wahr"], r["anteil_unter_wahr"]
        se = math.hypot(math.sqrt(pa * (1 - pa) / a["n_gueltig"]), math.sqrt(pr * (1 - pr) / r["n_gueltig"]))
        se_med = 1.253 * math.hypot(r["iqr_sd"] / math.sqrt(r["n_gueltig"]), a["iqr_sd"] / math.sqrt(a["n_gueltig"]))   # Standardfehler eines Medians ≈ 1,253 σ/√n (robustes σ)
        assert abs(r["median"] - a["median"]) <= C.BAND_SE * se_med and abs(pr - pa) <= C.BAND_SE * se, (r, a)   # Bias und SD sind bei kleinem m Ausreißer-Maße und stehen nicht im Band
        n_ok += 1
    P(f"positions --check: {n_ok} Vergleiche im Band")


def main(argv):
    if not argv or argv[0] not in ("mixture-inv", "fit", "block", "positions", "all"):
        print(__doc__)
        return 2
    teil, check, lang = argv[0], "--check" in argv, "--lang" in argv
    nproc = 4
    if "--prozesse" in argv:
        nproc = int(argv[argv.index("--prozesse") + 1])
    teile = ["mixture-inv", "fit", "block", "positions"] if teil == "all" else [teil]
    for t in teile:
        if t == "mixture-inv":
            teil_mixture_inv(check)
        elif t == "fit":
            d = teil_fit(check, lang, nproc)
            check_fit(d) if check else write_part("fit", R_FIT, BASIS_FIT, d, "fit")
        elif t == "block":
            d = teil_block(check, lang, nproc)
            check_block(d) if check else write_part("block", R_BLOCK, BASIS_BLOCK, d, "block")
        elif t == "positions":
            d = teil_positions(check, lang, nproc)
            check_positions(d) if check else write_part("positionen", R_POS, BASIS_POS, d, "positions")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
