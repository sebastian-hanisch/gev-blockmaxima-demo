"""GEV an Blockmaxima: wie viel eine kurze Reihe verrät - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Demo der Linie „Extremwerte erklärt“: aus m Block-Maxima die Parameter der GEV schätzen und den Pegel zu q = 1/(2N) ablesen; wahr ist der exakte Pegel der erzeugenden Verteilung (F^b; bei „Exakte GEV“ die erzeugende GEV).
Rechnet nur mit numpy und der Standardbibliothek (kein scipy); die schweren Läufe sind eingefroren (gvb_tables.json, Skript tools/gvb_make_tables.py).

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

import gvb_constants as C
import gvb_evaluation as EV
import gvb_fit as F
import gvb_gev as G
import gvb_population as PO
import gvb_reference as REF
import gvb_sources as S
import gvb_texts as T
import gvb_visualization as V
from gvb_layout import dunkles_design, kopf_css
from gvb_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from gvb_texts import de, pct

st.set_page_config(page_title="GEV an Blockmaxima – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return EV.analyse(settings)


@st.cache_data(show_spinner=False)
def _live_cell(xi, m, runs, seed):
    t0 = time.perf_counter()
    summary, A = EV.live_cell(EV.Settings(xi=xi, m=m, runs=runs, seed=seed))
    return summary, A, time.perf_counter() - t0


@st.cache_data(show_spinner=False)
def _pop_table(source):
    return {b: PO.population_bias(source, b) for b in C.BLOCKS}


@st.cache_data(show_spinner=False)
def _normal_reference(b, m):
    t0 = time.perf_counter()
    r = PO.normal_reference(b, m)
    return r, time.perf_counter() - t0


@st.cache_data(show_spinner=False)
def _frechet(bs):
    return {b: PO.frechet_population(b) for b in bs}


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in gvb_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 GEV an Blockmaxima: wie viel eine kurze Reihe verrät")
st.markdown(T.INTRO)

with st.expander("So funktioniert die Block-Maxima-Methode", expanded=True):
    st.markdown(T.HOW)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Eine Reihe anpassen (Experiment 1)**")
    quelle = st.selectbox("Quelle der Beobachtungen", C.SOURCES, key="quelle_sel",
                          help="Ausgedachte Quellen mit exakter Verteilung. „Exakte GEV“: die Block-Maxima sind direkt GEV(0, 1, ξ)-verteilt (ξ und m unten).")
    b = st.select_slider("Blockgröße b", C.BLOCKS, key="b_sl", help="Beobachtungen je Block; N = 18 000, also m = 18 000 / b Block-Maxima. Gilt nicht bei „Exakte GEV“ (dort zählt m).")
    verfahren = st.selectbox("Verfahren", F.METHODS, key="verfahren_sel", format_func=lambda k: F.METHOD_LABELS[k],
                             help="ML: Maximum Likelihood mit ξ ≥ −1. L-Momente: ξ̂ exakt aus τ₃, geklammert. Gumbel: ξ = 0 fest, nur richtig, wenn ξ wirklich 0 ist.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt die gezogene Reihe fest; derselbe Zufalls-Seed gibt auf jeder Plattform dieselben Zufallszahlen; die angezeigten Ergebnisse stimmen bis auf Rundungsstellen überein.")
    st.button("🎲 Neue Reihe ziehen", width="stretch", on_click=randomize_seed)
    st.markdown("**Streuung von ξ̂ und Rand (Experimente 2 und 3, auch „Exakte GEV“ in Experiment 1)**")
    xi = st.select_slider("Wahres ξ der Block-Maxima", C.XI_OPTIONS, key="xi_sl", format_func=lambda v: de(v, 1),
                          help="ξ > 0 Fréchet-Typ, 0 Gumbel, < 0 Weibull-Typ. Eingefroren sind −0,9, −0,8, −0,6, −0,3, 0, 0,2 und 0,5; −1,2 gibt es nur live.")
    m = st.select_slider("Zahl der Block-Maxima m", C.M_OPTIONS, key="m_sl", help="Eingefroren sind m = 10, 20, 50 und 100; m = 500 gibt es nur live.")
    runs = st.slider("Läufe (live)", C.RUNS_MIN, C.RUNS_MAX, key="runs_sl", step=C.RUNS_STEP,
                     help="Unabhängige Wiederholungen je Zelle. Die eingefrorene Tabelle hat 1 000 Läufe; live ist die Zufallsschwankung der Anteile deutlich größer.")
    klammer = st.toggle("Klammer ξ̂ ≥ −0,999 bei L-Momenten", key="klammer_tg", help="Ja: ξ̂ ∈ [−0,999; 0,99] (Wahl dieser Demo). Nein: ξ̂ bis −60 zugelassen, mehr oder gleich viele unmögliche Anpassungen (die Klammer greift nur bei ξ nahe −1 oder kleinem m).")
    st.markdown("**Blockgröße (Experiment 4)**")
    quelle4 = st.selectbox("Quelle der Blockgrößen-Tabelle", C.SOURCES4, key="quelle4_sel", help="Eingefroren: 1 000 Läufe je Quelle und Blockgröße.")

sync_query_params({k: st.session_state[k] for k in (
    "quelle_sel", "b_sl", "verfahren_sel", "seed_input", "xi_sl", "m_sl", "runs_sl", "klammer_tg", "quelle4_sel")})

settings = EV.Settings(quelle, int(b), verfahren, int(seed), float(xi), int(m), int(runs), bool(klammer), quelle4)

# --- Experiment 1: eine Reihe anpassen -----------------------------------------------------------------------------------------------------

a = _analysis(settings)
st.markdown("## 🎯 Experiment 1: Eine Reihe anpassen")
st.caption(T.SOURCE_NOTES[settings.quelle])
if settings.quelle == "Exakte GEV":
    st.caption(f"Die Block-Maxima sind direkt GEV(0, 1, ξ = {de(settings.xi, 1)})-verteilt, m = {a.m}; der Pegel gehört zur Überschreitungswahrscheinlichkeit p = 1/(2m) je Block "
               "(so selten, dass er im Mittel einmal in der doppelten Reihe vorkommt). Die Blockgröße gilt dort nicht.")
else:
    st.caption(f"N = 18 000 Beobachtungen in Blöcken zu b = {a.b}: m = {a.m} Block-Maxima ({T.pct(a.m / C.N_OBS, 1)} der Beobachtungen gehen in die Anpassung). "
               f"Ziel: der Wert, den eine Beobachtung mit q = 1/(2N) = {de(C.Q_TARGET * 1e5, 2)}·10⁻⁵ überschreitet; je Block entspricht das p_b = {de(a.p * 100, 3)} %.")
fit = a.fits.get(settings.verfahren)
if fit is None:
    st.warning(f"Mit dem Verfahren „{settings.verfahren}“ ist bei m = {a.m} Block-Maxima keine Anpassung möglich: {a.errors.get(settings.verfahren, '')}")
else:
    c1, c2, c3, c4 = st.columns(4)
    diff = fit.level(a.p) - a.truth
    c1.metric("Wahrer Pegel (aus F^b)" if settings.quelle != "Exakte GEV" else "Wahrer Pegel (erzeugende GEV)", de(a.truth, 2))
    # Delta = „mein Wert minus Referenz“; neutrale Farbe, weil Abweichung von der Wahrheit nach beiden Seiten Fehler ist; ASCII-Minus, damit Streamlit den Pfeil richtig setzt
    c2.metric("Geschätzter Pegel", de(fit.level(a.p), 2), delta=de(diff, 2).replace("−", "-") + " gegen wahr", delta_color="off")
    c3.metric("V = geschätzt / wahr", de(a.v(settings.verfahren), 3), help="Pegel-Verhältnis; 1 heißt: der geschätzte Pegel trifft den wahren.")
    c4.metric("ξ̂", de(fit.xi, 3), help="Geschätzter Formparameter dieser einen Reihe; die Streuung von ξ̂ über Läufe zeigt Experiment 2.")
    st.caption(f"Angepasst: μ̂ = {de(fit.mu, 2)}, σ̂ = {de(fit.sigma, 2)}, ξ̂ = {de(fit.xi, 3)}" + (f"; geschätzter oberer Endpunkt {de(fit.endpoint, 2)}" if fit.xi < -1e-9 else "") + ".")
    p_grid = np.logspace(np.log10(min(a.p / 5, 0.01)), np.log10(0.9), 120)
    curves = [("wahre Kurve (F^b)", C.COLOR_TRUE, EV.true_curve(a, p_grid), "solid"),
              (f"Anpassung: {F.METHOD_LABELS[settings.verfahren]}", C.COLOR_FIT, G.pegel(p_grid, fit.mu, fit.sigma, fit.xi), "solid")]
    st.plotly_chart(V.fig_pegel(a.maxima, p_grid, curves, a.p, a.truth), width="stretch", key="pegel_chart")
    st.caption("Die Punkte stehen an Positionen nach Gringorten (i − 0,44)/(m + 0,12), nur zur Anzeige; angepasst wird nicht an diese Positionen. Die senkrechte Linie ist das Ziel.")
    others = [f"{F.METHOD_LABELS[k]}: V = {de(a.ratios[k], 3)}" for k in F.METHODS if k in a.ratios and k != settings.verfahren]
    if others:
        st.caption("Dieselben Block-Maxima mit den anderen Verfahren: " + "; ".join(others) + ".")

st.markdown("## ❓ Kernfragen")
st.markdown(
    "1. **Was schätzt die Anpassung aus diesen Blöcken, und wo liegt die Wahrheit aus F^b?** (hier)\n"
    "2. **Wie viele Blöcke braucht man, bis ξ̂ etwas sagt?** (Experiment 2)\n"
    "3. **Wo versagt die Anpassung?** (Experiment 3)\n"
    "4. **Welche Blockgröße ist am besten, und woran hängt das?** (Experiment 4)"
)
ref1 = None
if settings.quelle in C.SOURCES4:
    ref1 = REF.block_cell(settings.quelle, a.b)
kind, text = EV.verdict_fit(a, settings.verfahren, ref1)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente 2 und 3: viele Läufe -------------------------------------------------------------------------------------------------------

with st.spinner(f"Rechne {settings.runs} Läufe mit m = {settings.m} Block-Maxima (ML und L-Momente) …"):
    summary, A_runs, secs = _live_cell(settings.xi, settings.m, settings.runs, settings.seed)
frozen = REF.fit_cell(settings.xi, settings.m)
lm_label = "L-Momente (mit Klammer)" if settings.klammer else "L-Momente (ohne Klammer)"

st.subheader("🔬 Experiment 2: Wie viele Blöcke braucht ξ̂?")
st.caption(f"{settings.runs} Läufe live ({de(secs, 1)} s auf diesem Rechner, bei der Streamlit Cloud langsamer); Blöcke exakt GEV(0, 1, ξ = {de(settings.xi, 1)}), m = {settings.m}. "
           "Die Tabelle mit 1 000 Läufen je Zelle ist eingefroren und steht darunter.")
st.plotly_chart(V.fig_xi_hist(A_runs[:, 0], A_runs[:, 4 if settings.klammer else 7], settings.xi, lm_label), width="stretch", key="xi_hist")
live_ml, live_lm = summary["ml"]["xi"], summary[EV.lm_key(settings.klammer)]["xi"]
rows = [{"Verfahren": "ML (live)", "Mittel ξ̂": de(live_ml["mittel"], 3), "SD": de(live_ml["sd"], 3), "Median": de(live_ml["median"], 3), "IQR/1,349": de(live_ml["iqr_sd"], 3), "größtes ξ̂": de(live_ml["max"], 2)},
        {"Verfahren": lm_label + " (live)", "Mittel ξ̂": de(live_lm["mittel"], 3), "SD": de(live_lm["sd"], 3), "Median": de(live_lm["median"], 3), "IQR/1,349": de(live_lm["iqr_sd"], 3), "größtes ξ̂": de(live_lm["max"], 2)}]
if frozen is not None:
    fm, fl = frozen["ml"]["xi"], frozen[EV.lm_key(settings.klammer)]["xi"]
    rows += [{"Verfahren": "ML (eingefroren, 1 000 Läufe)", "Mittel ξ̂": de(fm["mittel"], 3), "SD": de(fm["sd"], 3), "Median": de(fm["median"], 3), "IQR/1,349": de(fm["iqr_sd"], 3), "größtes ξ̂": de(fm["max"], 2)},
             {"Verfahren": lm_label + " (eingefroren, 1 000 Läufe)", "Mittel ξ̂": de(fl["mittel"], 3), "SD": de(fl["sd"], 3), "Median": de(fl["median"], 3), "IQR/1,349": de(fl["iqr_sd"], 3), "größtes ξ̂": de(fl["max"], 2)}]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
if frozen is None:
    st.caption("Für diese Kombination (ξ, m) gibt es keine eingefrorene Zelle (m = 500 und ξ = −1,2 nur live).")
kind, text = EV.verdict_xi_spread(summary, frozen, settings.xi, settings.m, settings.klammer)
getattr(st, kind)(text)
with st.expander("Eingefrorene Tabelle: SD von ξ̂ je ξ und m (1 000 Läufe, Blöcke exakt GEV)"):
    fr = []
    for c in REF.tables()["fit"]["daten"]:
        fr.append({"ξ": de(c["xi"], 1), "m": c["m"], "ML: SD": de(c["ml"]["xi"]["sd"], 3), "ML: IQR/1,349": de(c["ml"]["xi"]["iqr_sd"], 3),
                   "L-Momente (Klammer): SD": de(c["lm_klammer"]["xi"]["sd"], 3), "L-Momente (ohne): SD": de(c["lm_ohne"]["xi"]["sd"], 3)})
    st.dataframe(pd.DataFrame(fr), hide_index=True, width="stretch")
    st.caption(f"Quelle: {REF.meta('fit')['quelle']}, Stand {REF.meta('fit')['stand']}, {REF.meta('fit')['laeufe_je_zelle']} Läufe je Zelle.")

st.markdown("---")

st.subheader("🔬 Experiment 3: Wo versagt die Anpassung?")
st.caption(f"Dieselben {settings.runs} Läufe wie oben (ξ = {de(settings.xi, 1)}, m = {settings.m}). Der Schalter „Klammer“ in der Seitenleiste gilt für die L-Momente hier und in Experiment 2.")
e1, e2, e3 = st.columns(3)
rand = summary["ml"]["rand"]
lmk = summary["lm_klammer"]["endpunkt_unter_max"]
lmo = summary["lm_ohne"]["endpunkt_unter_max"]
e1.metric("ML klebt an ξ = −1 (Optimierer)", pct(rand["anteil"]))
e2.metric("L-Momente: Endpunkt unter dem Maximum, mit Klammer", pct(lmk["anteil"]))
e3.metric("L-Momente: Endpunkt unter dem Maximum, ohne Klammer", pct(lmo["anteil"]))
st.caption(f"Standardfehler der drei Anteile bei {settings.runs} Läufen: ± {de(100 * rand['se'], 1)} / ± {de(100 * lmk['se'], 1)} / ± {de(100 * lmo['se'], 1)} Prozentpunkte (live ist die Zufallsschwankung deutlich größer als in der eingefrorenen Tabelle).")
if frozen is not None:
    st.caption(f"Eingefroren (1 000 Läufe): ML am Rand {pct(frozen['ml']['rand']['anteil'])}; L-Momente unter dem Maximum mit Klammer {pct(frozen['lm_klammer']['endpunkt_unter_max']['anteil'])}, "
               f"ohne Klammer {pct(frozen['lm_ohne']['endpunkt_unter_max']['anteil'])}; ML nie ({pct(frozen['ml']['endpunkt_unter_max']['anteil'])}), weil die Likelihood es verbietet.")
kind, text = EV.verdict_edge(summary, settings.xi, settings.m, settings.klammer)
getattr(st, kind)(text)
g1, g2 = st.columns(2)
with g1:
    st.markdown("**Beispiel: der Endpunkt unter einem beobachteten Wert** (fünf Block-Maxima 0, 3, 3, 3, 4; τ₃ = −1/2)")
    ex = np.array([0.0, 3.0, 3.0, 3.0, 4.0])
    f_mit, f_ohne = G.fit_lmoments(ex, True), G.fit_lmoments(ex, False)
    pg = np.logspace(-3, np.log10(0.9), 100)
    cur = [(f"ohne Klammer: ξ̂ = {de(f_ohne[2], 3)}, Endpunkt {de(G.endpoint(*f_ohne), 2)} < 4", C.COLOR_FIT, G.pegel(pg, *f_ohne), "solid"),
           (f"mit Klammer: ξ̂ = {de(f_mit[2], 3)}, Endpunkt {de(G.endpoint(*f_mit), 2)}", C.COLOR_ALT, G.pegel(pg, *f_mit), "dash")]
    st.plotly_chart(V.fig_pegel(ex, pg, cur, 0.01, G.pegel(0.01, *f_mit), title="Fünf Werte, zwei Fassungen der L-Moment-Anpassung"), width="stretch", key="endpoint_example")
    st.caption("Ohne Klammer endet die angepasste Kurve bei 3,84 und damit unter dem beobachteten Wert 4 (unmögliche Anpassung); die Klammer gibt den Endpunkt 4,20. Die Raute ist hier der Pegel der Fassung mit Klammer, keine Wahrheit.")
with g2:
    st.markdown("**Die Likelihood ohne Grenze** (Handbeispiel, drei Werte 1, 2, 3, σ = 1)")
    st.plotly_chart(V.fig_nll_unbounded(), width="stretch", key="nll_unbounded")
    st.caption("Für ξ < −1 sinkt die negative Log-Likelihood ohne untere Grenze, wenn der Endpunkt je Zehnerpotenz näher an den größten Wert rückt: zuletzt um (1 + 1/ξ)·ln 10 (ξ = −1,5: 0,7675; ab Abstand 10⁻³ weniger als 0,02 daneben); deshalb die Nebenbedingung ξ ≥ −1, eine Wahl dieser Demo. Bei ξ = −0,8 steigt sie.")
st.caption("Bekannt ist, dass ML für ξ ≤ −0,5 irregulär wird (Smith 1985, nach Coles 2001, S. 55); hier steht nur der gemessene Befund, ohne Erklärung des Randes.")

st.markdown("---")

# --- Experiment 4: Blockgröße -----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Blockgröße, Bias gegen Streuung")
st.caption(f"Quelle {settings.quelle4}: N = 18 000, Ziel q = 1/(2N), ML-GEV und Gumbel (ξ = 0 fest) an die m = N/b Block-Maxima; 1 000 Läufe je Blockgröße sind eingefroren, "
           "dazu ein Lauf live (Zufalls-Seed der Seitenleiste). V = geschätzter / wahrer Pegel; lnRMSE² = Bias² + SD².")
brows = EV.block_rows(settings.quelle4)
has_limit = S.survival_is_known_limit(settings.quelle4)
if has_limit:
    with st.spinner("Rechne den Populations-Bias (Quadratur, 20 000 Knoten) …"):
        pop = _pop_table(settings.quelle4)
else:
    pop = None
live4 = _analysis(replace(settings, quelle=settings.quelle4))
l1_, l2_ = live4.v("ML"), live4.v("Gumbel")
st.caption(f"Live, ein Lauf (b = {settings.b}, m = {live4.m}): V der freien GEV (ML) {de(l1_, 3)}, V Gumbel {de(l2_, 3)}. Eine Reihe ist ein Punkt der Streuung unten, kein Befund.")
st.plotly_chart(V.fig_blocksize(brows, pop, f"{settings.quelle4}: Fehler gegen Blockgröße (1 000 Läufe je Zelle)"), width="stretch", key="block_chart")
tab_rows = []
for c in brows:
    row = {"b": c["b"], "m": c["m"], "m/N": pct(c["m"] / C.N_OBS, 1)}
    row["Bias Grenzwert m → ∞ (ln)"] = de(pop[c["b"]]["ln_v"], 3) if pop else "kein Grenzwert ablesbar"
    row.update({"Bias der Läufe (ln)": de(c["ml"]["bias_ln"], 3), "SD der Läufe (ln)": de(c["ml"]["sd_ln"], 3), "lnRMSE ML": de(c["ml"]["ln_rmse"], 3), "lnRMSE Gumbel": de(c["gumbel"]["ln_rmse"], 3),
                "V Median (ML)": de(c["ml"]["median"], 3), "V Q10 / Q90 (ML)": f"{de(c['ml']['q10'], 3)} / {de(c['ml']['q90'], 3)}", "V < 1 (ML)": pct(c["ml"]["anteil_unter_1"])})
    tab_rows.append(row)
st.dataframe(pd.DataFrame(tab_rows), hide_index=True, width="stretch")
st.caption(f"Wahrer Pegel dieser Quelle: {de(REF.true_level_frozen(settings.quelle4), 4)} ({'Tage' if settings.quelle4 == 'Vehikel B' else 'in Einheiten der Quelle'}). "
           "Der Bias der Läufe ist das Mittel von ln V, die SD die Standardabweichung von ln V über die Läufe (Nenner R).")
if not has_limit:
    st.info("Für die Normalverteilung lässt sich auf dieser Quadratur kein Grenzwert des Bias für m → ∞ ablesen: die GEV mit ξ̂ < 0 hat einen oberen Endpunkt, die Normalverteilung hat darüber noch Wahrscheinlichkeit, und das Populations-Optimum hängt davon ab, "
            "wie viel vom Tail die Quadratur mitnimmt (Deutung; dass es für m → ∞ gar keinen Grenzwert gibt, ist damit nicht gezeigt). Stattdessen der Bias einer einzigen sehr großen Stichprobe, im Mittel über mehrere Zufalls-Seeds kleiner werdend (siehe README, Befund 8):")
    ref_b = st.selectbox("Blockgröße der Referenz", (6, 30), key="normref_b")
    ref_m = st.selectbox("Zahl m der Block-Maxima der Referenz", (3000, 30000, 300000), index=1, key="normref_m",
                         help="300 000 Block-Maxima rechneten bei der Messung der README meist 3 bis 8 s, bei starker Last bis zu rund 1 Minute (Cloud länger).")
    if st.button("Referenz rechnen", key="normref_go"):
        with st.spinner(f"Rechne die ML-Anpassung an {T.pint(ref_m)} Block-Maxima …"):
            r, secs_r = _normal_reference(int(ref_b), int(ref_m))
        st.write(f"b = {ref_b}, m = {T.pint(ref_m)} (eine Stichprobe, {de(secs_r, 1)} s): ln V = {de(r['ln_v'], 3)}, ξ̂ = {de(r['xi'], 3)}, geschätzter Endpunkt {de(r['endpunkt'], 2)}, größter Wert {de(r['groesster_wert'], 2)}.")
kind, text = EV.verdict_blocksize(settings.quelle4, brows, pop)
getattr(st, kind)(text)
with st.expander("Kontrolle: Standard-Fréchet ist max-stabil (kein Block-Bias bei jedem b)"):
    st.markdown("F^b der Standard-Fréchet-Verteilung ist für jedes b eine GEV mit μ = σ = b und ξ = 1. Die Populations-ML-Anpassung (Quadratur) muss das wiederfinden: der Bias verschwindet, und die Blockgröße spielt keine Rolle.")
    if st.button("Fréchet-Kontrolle rechnen (b = 6, 60, 360)", key="frechet_go"):
        with st.spinner("Rechne die Quadratur für drei Blockgrößen …"):
            fr = _frechet((6, 60, 360))
        st.dataframe(pd.DataFrame([{"b": k, "ξ̂": de(v["xi"], 4), "μ̂ / b": de(v["mu_je_b"], 4), "σ̂ / b": de(v["sigma_je_b"], 4), "Bias (ln V)": de(v["ln_v"], 4)} for k, v in fr.items()]),
                     hide_index=True, width="stretch")

st.markdown("---")

# --- Experiment 5: Handverfahren ------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Alte Handverfahren und was sie kosten")
with st.expander("Alte Handverfahren und was sie kosten (eingefroren, 2 000 Läufe je Zelle)", expanded=False):
    st.caption("Block-Maxima exakt GEV(0, 1, ξ); Fehler = geschätzter minus wahrer 100er-Wert (Überschreitungswahrscheinlichkeit 0,01 je Block) in Einheiten σ. Alle Verfahren sehen dieselben Stichproben. "
               "Die SD von ML-GEV bei m = 20 ist ein Ausreißer-Maß (wenige Läufe mit riesigem ξ̂): Median und IQR/1,349 lesen.")
    pr = REF.positions_rows()
    st.dataframe(pd.DataFrame([{"ξ": de(r["xi"], 1), "m": r["m"], "Verfahren": r["verfahren"], "Bias": de(r["bias"], 3), "Median": de(r["median"], 3), "SD": de(r["sd"], 2), "IQR/1,349": de(r["iqr_sd"], 2),
                                "RMSE": de(r["rmse"], 2), "zu niedrig": pct(r["anteil_unter_wahr"])} for r in pr]), hide_index=True, width="stretch")
    st.caption("„Positionen nach Weibull“ i/(m + 1) bezeichnet nur diese Rangformel und hat mit dem Weibull-Typ der GEV (ξ < 0) nichts zu tun. Gringorten: (i − 0,44)/(m + 0,12). "
               "Die Kleinste-Quadrate-Anpassung ist in dieser Umsetzung (ungewichtet, Positionen nach Weibull, ξ auf |ξ| ≤ 0,9 begrenzt) schlecht; das sagt nichts über die Methodenklasse.")
kind, text = EV.verdict_positions(REF.positions_rows())
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS)
st.markdown(T.CONVENTIONS)
st.info(T.VEHIKEL_TEXT)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.FORMULAS)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html)."
)
