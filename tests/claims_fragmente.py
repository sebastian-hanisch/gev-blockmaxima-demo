"""Zahlen der README als Text-Bausteine, berechnet aus den eingefrorenen Tabellen und den deterministischen Rechnungen (Quadratur, Mischung).

`tests/test_claims.py` prüft, dass jeder Baustein wörtlich in der README steht (und damit jede README-Zahl eine Herkunft hat); der Satzbau der README ist Handarbeit."""

import math
from functools import lru_cache

import gvb_constants as C
import gvb_evaluation as EV
import gvb_population as PO
import gvb_reference as REF
import gvb_sources as S
from gvb_texts import de, pct


def _c(xi, m):
    return REF.fit_cell(xi, m)


def _trio(f, fmt):
    return " / ".join(fmt(f(m)) for m in (20, 50, 100))


@lru_cache(maxsize=1)
def _build():
    f = {}
    d3 = lambda v: de(v, 3)
    f["sd_ml_xi0"] = _trio(lambda m: _c(0.0, m)["ml"]["xi"]["sd"], d3)
    f["sd_ml_xi02"] = _trio(lambda m: _c(0.2, m)["ml"]["xi"]["sd"], d3)
    f["sd_lm_xi0_m50"] = d3(_c(0.0, 50)["lm_klammer"]["xi"]["sd"])
    f["sd_ml_xi0_m50"] = d3(_c(0.0, 50)["ml"]["xi"]["sd"])
    f["m10_sd"] = " / ".join(d3(_c(xi, 10)["ml"]["xi"]["sd"]) for xi in (0.0, 0.2, 0.5))
    f["m10_median"] = " / ".join(d3(_c(xi, 10)["ml"]["xi"]["median"]) for xi in (0.0, 0.2, 0.5))
    f["m10_iqr"] = " / ".join(d3(_c(xi, 10)["ml"]["xi"]["iqr_sd"]) for xi in (0.0, 0.2, 0.5))
    f["m10_max"] = de(max(_c(xi, 10)["ml"]["xi"]["max"] for xi in (0.0, 0.2, 0.5)), 1)
    six = (-0.9, -0.6, -0.3, 0.0, 0.2, 0.5)
    kleiner = lambda m: sum(1 for xi in six if _c(xi, m)["lm_klammer"]["xi"]["sd"] < _c(xi, m)["ml"]["xi"]["sd"])
    f["lm_kleiner_m_le_20"] = f"{kleiner(10) + kleiner(20)} von 12"
    f["lm_kleiner_m100"] = f"{kleiner(100)} von 6"
    f["ausnahmen_m_le_20"] = "ξ = −0,9 bei m = 10 und m = 20"
    assert [(xi, m) for xi in six for m in (10, 20) if not _c(xi, m)["lm_klammer"]["xi"]["sd"] < _c(xi, m)["ml"]["xi"]["sd"]] == [(-0.9, 10), (-0.9, 20)]
    f["rand_06"] = " / ".join(pct(_c(-0.6, m)["ml"]["rand"]["anteil"]) for m in (10, 20, 50))
    f["rand_09"] = " / ".join(pct(_c(-0.9, m)["ml"]["rand"]["anteil"]) for m in (10, 20, 50, 100))
    lo1, hi1, lo2, hi2 = EV.lm_endpoint_range()
    f["lm_span_mit"] = f"{pct(lo1)} bis {pct(hi1)}"
    f["lm_span_ohne"] = f"{pct(lo2)} bis {pct(hi2)}"
    f["lm_08_m100"] = pct(_c(-0.8, 100)["lm_klammer"]["endpunkt_unter_max"]["anteil"])
    f["lm_08_m100_ohne"] = pct(_c(-0.8, 100)["lm_ohne"]["endpunkt_unter_max"]["anteil"])
    f["lm_09_m20_mit"] = pct(_c(-0.9, 20)["lm_klammer"]["endpunkt_unter_max"]["anteil"])
    f["lm_09_m20_ohne"] = pct(_c(-0.9, 20)["lm_ohne"]["endpunkt_unter_max"]["anteil"])
    f["sd_09_m20"] = f"bei ML {d3(_c(-0.9, 20)['ml']['xi']['sd'])}, bei den L-Momenten {d3(_c(-0.9, 20)['lm_klammer']['xi']['sd'])} mit und {d3(_c(-0.9, 20)['lm_ohne']['xi']['sd'])} ohne Klammer"
    assert _c(-0.9, 20)["ml"]["xi"]["sd"] < _c(-0.9, 20)["lm_klammer"]["xi"]["sd"] < _c(-0.9, 20)["lm_ohne"]["xi"]["sd"]          # „L-Momente sind stabiler“ gilt dort auch mit der Klammer nicht
    f["sd_lm_ml_xi0_m100"] = f"{d3(_c(0.0, 100)['lm_klammer']['xi']['sd'])} gegen {d3(_c(0.0, 100)['ml']['xi']['sd'])}"
    cells = REF.tables()["fit"]["daten"]
    cmp = [(z["lm_ohne"]["endpunkt_unter_max"]["anteil"] > z["lm_klammer"]["endpunkt_unter_max"]["anteil"], z["lm_ohne"]["endpunkt_unter_max"]["anteil"] == z["lm_klammer"]["endpunkt_unter_max"]["anteil"]) for z in cells]
    f["klammer_mehr_gleich"] = f"in {sum(1 for a, _ in cmp if a)} von 28 Zellen mehr unmögliche Anpassungen und in {sum(1 for _, g in cmp if g)} gleich viele, in keiner weniger"
    assert sum(1 for a, g in cmp if not a and not g) == 0
    f["lm_mean_05"] = f"{d3(_c(0.5, 50)['lm_klammer']['xi']['mittel'])} gegen {d3(_c(0.5, 50)['ml']['xi']['mittel'])}"
    f["sd_06_m20"] = f"{d3(_c(-0.6, 20)['lm_klammer']['xi']['sd'])} gegen {d3(_c(-0.6, 20)['ml']['xi']['sd'])}"
    f["se_08"] = "rund " + de(100 * _c(-0.8, 100)["lm_klammer"]["endpunkt_unter_max"]["se"], 1) + " Punkte"
    f["ml_endpkt_nie"] = "0 von 28 Zellen"
    assert all(c["ml"]["endpunkt_unter_max"]["anteil"] == 0 for c in REF.tables()["fit"]["daten"]) and len(REF.tables()["fit"]["daten"]) == 28

    def blk(src, b, key="ml"):
        return REF.block_cell(src, b)[key]
    f["n6_bias_sd"] = f"Bias {d3(blk('Normal', 6)['bias_ln'])} bei SD {d3(blk('Normal', 6)['sd_ln'])}"
    f["n360_bias_sd"] = f"Bias {d3(blk('Normal', 360)['bias_ln'])} bei SD {d3(blk('Normal', 360)['sd_ln'])}"
    f["n30_rmse"] = d3(blk("Normal", 30)["ln_rmse"])
    f["n_best"] = f"b = {EV.best_block(EV.block_rows('Normal'))['b']}"
    f["n_gumbel_360"] = f"{d3(blk('Normal', 360, 'gumbel')['ln_rmse'])} gegen {d3(blk('Normal', 360)['ln_rmse'])}"
    f["n_gumbel_better"] = f"{sum(1 for c in EV.block_rows('Normal') if c['gumbel']['ln_rmse'] < c['ml']['ln_rmse'])} von 7"
    f["e_gumbel_better"] = f"{sum(1 for c in EV.block_rows('Exponential') if c['gumbel']['ln_rmse'] < c['ml']['ln_rmse'])} von 7"
    f["e30_rmse"] = f"{d3(blk('Exponential', 30, 'gumbel')['ln_rmse'])} gegen {d3(blk('Exponential', 30)['ln_rmse'])}"
    f["e6_bias"] = d3(blk("Exponential", 6)["bias_ln"])
    f["b_best"] = f"b = {EV.best_block(EV.block_rows('Vehikel B'))['b']} ({d3(EV.best_block(EV.block_rows('Vehikel B'))['ml']['ln_rmse'])})"
    f["b12_30_rmse"] = f"{d3(blk('Vehikel B', 12)['ln_rmse'])} und {d3(blk('Vehikel B', 30)['ln_rmse'])}"
    gm = [c["gumbel"]["median"] for c in EV.block_rows("Vehikel B")]
    f["b_gumbel_faktor"] = f"das {de(min(gm), 2)}-Fache (b = 6) bis {de(max(gm), 2)}-Fache (b = 360)"
    f["b_gumbel_b12_30"] = f"das {de(blk('Vehikel B', 12, 'gumbel')['median'], 2)}- bis {de(blk('Vehikel B', 30, 'gumbel')['median'], 2)}-Fache"
    vb = EV.block_rows("Vehikel B")
    inner = [vb[i]["b"] for i in range(1, 6) if vb[i]["ml"]["ln_rmse"] < vb[i - 1]["ml"]["ln_rmse"] and vb[i]["ml"]["ln_rmse"] < vb[i + 1]["ml"]["ln_rmse"]]
    assert inner == [180]                                   # genau ein lokales Minimum im Inneren, das globale liegt am Rand b = 6
    f["b180_360"] = f"bei b = 180 ({d3(blk('Vehikel B', 180)['ln_rmse'])} gegen {d3(blk('Vehikel B', 360)['ln_rmse'])} bei b = 360)"
    f["n60_rmse"] = f"praktisch gleich bei b = 60 ({d3(blk('Normal', 60)['ln_rmse'])};"
    f["e6_gumbel_bias"] = f"Gumbel mit festem ξ = 0 dort {de(blk('Exponential', 6, 'gumbel')['bias_ln'], 3)}"
    f["pop_e360"] = f"Exponentialquelle und b = 360: {de(PO.population_bias('Exponential', 360)['ln_v'], 4)} in ln"
    f["pop_b360"] = f"noch {de(PO.population_bias('Vehikel B', 360)['ln_v'], 3)}"
    f["n_logit_hi"] = " / ".join(de(PO.population_bias_logit("Normal", 6, S_)[0], 3) for S_ in (60.0, 100.0, 200.0, 400.0))
    f["b_sd_monoton"] = "monoton" if all(blk("Vehikel B", C.BLOCKS[i])["sd_ln"] < blk("Vehikel B", C.BLOCKS[i + 1])["sd_ln"] for i in range(6)) else "nicht monoton"
    pb = {b: PO.population_bias("Vehikel B", b)["ln_v"] for b in C.BLOCKS}
    f["pop_b12"] = f"{de(math.exp(pb[12]), 2)}"
    f["pop_b12_pct"] = f"{de(100 * (math.exp(pb[12]) - 1), 0)} %"
    f["pop_b30"] = f"{de(100 * (math.exp(pb[30]) - 1), 0)} %"
    pe = PO.population_bias("Exponential", 6)["ln_v"]
    f["pop_e6"] = de(pe, 2)
    abw = [abs(REF.block_cell(s, b)["ml"]["bias_ln"] - PO.population_bias(s, b)["ln_v"]) for s in ("Exponential", "Vehikel B") for b in C.BLOCKS]
    assert max(abw) < 0.02
    f["pop_14"] = f"{sum(1 for a in abw if a < 0.03)} von 14 Zellen innerhalb 0,03 (größte Abweichung unter 0,02)"
    f["n_logit"] = " / ".join(de(PO.population_bias_logit("Normal", 6, S_)[0], 3) for S_ in (6.0, 9.2, 13.8, 20.7, 30.0))
    f["pos_weibull"] = d3(next(r for r in REF.positions_rows() if r["xi"] == 0.0 and r["m"] == 50 and r["verfahren"].startswith("Gumbel-Regression, Positionen nach Weibull"))["bias"])
    f["pos_gring"] = d3(next(r for r in REF.positions_rows() if r["xi"] == 0.0 and r["m"] == 50 and r["verfahren"].startswith("Gumbel-Regression, Gringorten"))["bias"])
    g50 = next(r for r in REF.positions_rows() if r["xi"] == 0.0 and r["m"] == 50 and r["verfahren"].startswith("Gumbel-ML"))
    gr50 = next(r for r in REF.positions_rows() if r["xi"] == 0.0 and r["m"] == 50 and r["verfahren"].startswith("Gumbel-Regression, Gringorten"))
    f["pos_rmse"] = f"{de(gr50['rmse'], 2)} gegen {de(g50['rmse'], 2)}"
    f["pos_gring_se"] = f"Standardfehler {de(gr50['sd'] / math.sqrt(2000), 3)} σ, von null nicht zu unterscheiden"
    assert abs(gr50["bias"]) < 2 * gr50["sd"] / math.sqrt(2000)
    w02 = next(r for r in REF.positions_rows() if r["xi"] == 0.2 and r["verfahren"].startswith("Gumbel-ML"))
    wm02 = next(r for r in REF.positions_rows() if r["xi"] == -0.2 and r["verfahren"].startswith("Gumbel-ML"))
    f["pos_g02"] = f"{de(w02['bias'], 2)} σ, in {pct(w02['anteil_unter_wahr'])} der Läufe zu niedrig"
    f["pos_gm02"] = f"{de(wm02['bias'], 2)} σ, in {pct(1 - wm02['anteil_unter_wahr'], 1)} der Läufe zu hoch"
    ml20 = next(r for r in REF.positions_rows() if r["xi"] == 0.0 and r["m"] == 20 and r["verfahren"] == "ML-GEV")
    f["ml20_med_iqr"] = f"Median {de(ml20['median'], 2)}, IQR/1,349 {de(ml20['iqr_sd'], 2)}"
    f["ml20_sd"] = de(ml20["sd"], 2)
    fr = PO.frechet_population(60)
    f["frechet_xi"] = de(fr["xi"], 4)
    f["frechet_lnv"] = "unter 0,002"
    assert all(abs(PO.frechet_population(b)["ln_v"]) < 0.002 for b in (6, 60, 360))
    f["mix_a"] = de(S.mixture_level(1e-5), 3)
    f["mix_b"] = de(S.mixture_level(1 - (1 - 1e-3) ** (1 / 100)), 3)
    f["mix_p30"] = de(S.mixture_sf(30.0), 6)
    f["wahr_n"] = de(S.true_level("Normal", C.Q_TARGET), 4)
    f["wahr_e"] = de(S.true_level("Exponential", C.Q_TARGET), 4)
    f["wahr_b"] = de(S.true_level("Vehikel B", C.Q_TARGET), 2)
    f["q"] = de(C.Q_TARGET * 1e5, 2) + "·10⁻⁵"
    for m in (3000, 30000, 300000):
        r = PO.normal_reference(6, m)
        f[f"nref_{m}"] = de(r["ln_v"], 2)
    f["cell_laeufe"] = "1 000"
    return f


def build():
    return dict(_build())
