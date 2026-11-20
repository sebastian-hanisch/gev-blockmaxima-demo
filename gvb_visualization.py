"""Plotly-Abbildungen (alle Achsen fest, `fixedrange`)."""


import numpy as np
import plotly.graph_objects as go

import gvb_constants as C
import gvb_gev as G
from gvb_texts import de


def _layout(fig, title=None, height=380, xtitle=None, ytitle=None):
    fig.update_layout(
        title=title, height=height, margin=dict(l=55, r=15, t=45 if title else 15, b=55), legend=dict(orientation="h", y=-0.22),
        xaxis=dict(fixedrange=True, title=xtitle), yaxis=dict(fixedrange=True, title=ytitle),
    )
    return fig


def reduced(p):
    """Gumbel-reduzierte Variable zur Überschreitungswahrscheinlichkeit p je Block: -ln(-ln(1 - p)) (für kleines p ≈ ln(1/p))."""
    return -np.log(-np.log1p(-np.asarray(p, float)))


def fig_pegel(maxima, p_grid, curves, p_target, truth, title="Pegel-Diagramm: Block-Maxima, angepasste Kurve, Wahrheit (F^b bzw. erzeugende GEV)"):
    """Block-Maxima (Positionen nach Gringorten, nur zur Anzeige) gegen die reduzierte Variable, angepasste Kurven und die wahre Kurve.
    `curves`: Liste (Name, Farbe, Feld der Pegel zu p_grid, Strichart)."""
    x = np.sort(np.asarray(maxima, float))
    m = x.size
    pos = (np.arange(1, m + 1) - 0.44) / (m + 0.12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reduced(1 - pos), y=x, mode="markers", name=f"{m} Block-Maxima", marker=dict(color=C.COLOR_PTS, size=6)))
    for name, color, ys, dash in curves:
        fig.add_trace(go.Scatter(x=reduced(p_grid), y=ys, mode="lines", name=name, line=dict(color=color, width=2.5, dash=dash)))
    ticks = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ticks = [t for t in ticks if 1.0 / t >= float(np.min(p_grid)) * 0.9]
    fig.add_vline(x=float(reduced(p_target)), line=dict(color="rgba(120,120,120,0.8)", dash="dot"))
    fig.add_trace(go.Scatter(x=[float(reduced(p_target))], y=[truth], mode="markers", name="wahrer Pegel", marker=dict(color=C.COLOR_TRUE, size=12, symbol="diamond")))
    _layout(fig, title, 400, "Wiederkehrperiode in Blöcken (1/p, Gumbel-Skala)", "Wert")
    fig.update_xaxes(tickvals=[float(reduced(1.0 / t)) for t in ticks], ticktext=[str(t) for t in ticks])
    return fig


def fig_xi_hist(A_ml, A_lm, xi_true, lm_label):
    """Verteilung von ξ̂ über die Läufe: ML und L-Momente übereinander, Wahrheit als Linie; Werte über 2 werden am Rand gesammelt (Anzahl im Titel)."""
    fig = go.Figure()
    out = 0
    for name, a, color in (("ML", A_ml, C.COLOR_FIT), (lm_label, A_lm, C.COLOR_ALT)):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        out += int(np.sum(a > 2.0))
        fig.add_trace(go.Histogram(x=np.clip(a, -1.2, 2.0), name=name, xbins=dict(start=-1.2, end=2.0, size=0.1), marker_color=color, opacity=0.6))
    fig.add_vline(x=xi_true, line=dict(color=C.COLOR_TRUE, width=3))
    fig.update_layout(barmode="overlay")
    _layout(fig, "ξ̂ je Lauf (grün: wahres ξ)" + (f"; {out} Werte über 2 am Rand gesammelt" if out else ""), 330, "ξ̂", "Läufe")
    fig.update_xaxes(range=[-1.25, 2.05])
    return fig


def fig_nll_unbounded(x=(1.0, 2.0, 3.0)):
    """Handbeispiel: negative Log-Likelihood gegen den Abstand ε des Endpunkts vom größten Wert (drei Werte 1, 2, 3, σ = 1): für ξ = −1,5 ohne Grenze fallend, für ξ = −0,8 steigend."""
    xs = np.array(x, float)
    eps = 10.0 ** np.linspace(-6, -1, 60)
    fig = go.Figure()
    for xi, color in ((-1.5, C.COLOR_FIT), (-0.8, C.COLOR_ALT)):
        ys = [G.nll_values(xs, xs.max() + e + 1.0 / xi, 1.0, xi) for e in eps]    # Endpunkt μ − σ/ξ = max + ε
        fig.add_trace(go.Scatter(x=eps, y=ys, mode="lines", name=f"ξ = {de(xi, 1)}", line=dict(color=color, width=2.5)))
    _layout(fig, "Likelihood am Rand: Endpunkt rückt an das größte Beobachtete (drei Werte 1, 2, 3; σ = 1)", 330, "Abstand ε des Endpunkts vom größten Wert", "negative Log-Likelihood")
    fig.update_xaxes(type="log", autorange="reversed")
    return fig


def fig_blocksize(rows, popl, title):
    """lnRMSE (ML-GEV und Gumbel) gegen die Blockgröße b, Minimum eingezeichnet; dazu Bias und SD der ML (in ln) als Punkte; popl = Populations-Bias je b (oder None)."""
    bs = [r["b"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bs, y=[r["ml"]["ln_rmse"] for r in rows], mode="lines+markers", name="lnRMSE, freie GEV (ML)", line=dict(color=C.COLOR_FIT, width=2.5)))
    fig.add_trace(go.Scatter(x=bs, y=[r["gumbel"]["ln_rmse"] for r in rows], mode="lines+markers", name="lnRMSE, Gumbel (ξ = 0 fest)", line=dict(color=C.COLOR_ALT, width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=bs, y=[r["ml"]["sd_ln"] for r in rows], mode="markers", name="SD der ML (Läufe)", marker=dict(color=C.COLOR_FIT, symbol="triangle-up", size=9)))
    fig.add_trace(go.Scatter(x=bs, y=[abs(r["ml"]["bias_ln"]) for r in rows], mode="markers", name="|Bias| der ML (Läufe)", marker=dict(color=C.COLOR_TRUE, symbol="square", size=8)))
    if popl:
        fig.add_trace(go.Scatter(x=bs, y=[abs(popl[b]["ln_v"]) for b in bs], mode="markers", name="|Bias| Grenzwert m → ∞ (Quadratur)", marker=dict(color="rgba(0,0,0,0)", line=dict(color=C.COLOR_TRUE, width=2), symbol="circle", size=13)))
    best = min(rows, key=lambda r: r["ml"]["ln_rmse"])
    fig.add_trace(go.Scatter(x=[best["b"]], y=[best["ml"]["ln_rmse"]], mode="markers", name="Minimum der ML", marker=dict(color="black", symbol="star", size=14)))
    _layout(fig, title, 400, "Blockgröße b (m = 18 000 / b Block-Maxima)", "Fehler in ln (Pegel-Verhältnis)")
    fig.update_xaxes(type="log", tickvals=bs, ticktext=[str(b) for b in bs])
    return fig
