"""Zahlenformat (deutsches Dezimalkomma) und die festen Texte der App."""

import math


def de(x, nd=2):
    """Zahl mit Dezimalkomma und nd Nachkommastellen (U+2212 als Minuszeichen)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    s = f"{x:.{nd}f}".replace(".", ",")
    if s.startswith("-") and float(x) != 0 and float(s.replace(",", ".")) != 0:
        s = "−" + s[1:]
    elif s.startswith("-"):
        s = s[1:]
    return s


def pct(p, nd=0):
    """Anteil 0..1 als Prozent mit Dezimalkomma."""
    return de(100 * p, nd) + " %"


def pint(n):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(n):,}".replace(",", " ")


INTRO = """
Aus den **Maxima von Blöcken** lässt sich der Rand einer Verteilung schätzen: Man teilt eine Reihe von **N = 18 000** unabhängigen Beobachtungen in Blöcke der Länge **b**, behält je Block das
Maximum und passt an die **m = N/b Block-Maxima** eine **GEV** (verallgemeinerte Extremwertverteilung) mit den Parametern (μ, σ, ξ) an. Gefragt ist der **Pegel**: der Wert, den eine Beobachtung nur mit der
Wahrscheinlichkeit q = 1/(2N) überschreitet, also im Mittel einmal in der doppelten Reihe. Die **Wahrheit** ist der exakte Pegel der erzeugenden Verteilung: F^b des Blocks bei den Quellen mit Einzelbeobachtungen, die erzeugende GEV bei „Exakte GEV“ (Experimente 2, 3 und 5); die Quellen sind **ausgedacht**
(Lehrbuch/Abstract: sie sagen nichts über Logistik, Lager oder Hafen). Vier Fragen: Was schätzt die Anpassung, wie viele Blöcke braucht ξ̂, wo versagt sie, und welche Blockgröße ist am besten?

Weiter unten: die Experimente **🔬 1 bis 5**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

HOW = """
1. **Block-Maxima bilden.** Die Reihe wird in m = N/b Blöcke zu b Beobachtungen geteilt; jeder Block liefert sein größtes Element.
2. **GEV anpassen.** Drei Verfahren: **ML** (Maximum Likelihood, eigener Nelder-Mead, Nebenbedingung ξ ≥ −1), **L-Momente** (ξ̂ aus dem dritten L-Moment-Verhältnis τ₃ exakt zurückgerechnet) und **Gumbel** (ξ = 0 fest).
3. **Pegel ablesen und prüfen.** Aus (μ̂, σ̂, ξ̂) folgt der Pegel zu p_b = 1 − (1 − q)^b je Block; verglichen wird er als Verhältnis V = Schätzwert / wahrer Pegel, wobei der wahre Pegel exakt aus F gerechnet ist.
4. **Warum das nicht trivial ist:** die GEV ist bei endlichem Block nur Näherung von F^b (Block-Bias), und aus wenigen Maxima ist ξ nur grob zu lesen (Streuung). Die Blockgröße verteilt den Fehler zwischen beiden.
"""

CONVENTIONS = """
**Konvention und Begriffe.** ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ mit oberem Endpunkt μ − σ/ξ. `scipy.stats.genextreme` hat **c = −ξ** (in dieser Demo steht scipy nur in den Tests);
Hosking schreibt k = −ξ. **Pegel** heißt der Wiederkehrwert, der geschätzte Wert zu einer Überschreitungswahrscheinlichkeit; der Extremwertindex ξ ist etwas anderes als der Extremalindex θ (θ kommt hier nicht vor).
**Zähleinheiten:** m Block-Maxima je Anpassung, b Beobachtungen je Block, N Beobachtungen je Reihe, **Läufe** unabhängige Wiederholungen einer ganzen Reihe.
"""

ASSUMPTIONS = """
| Annahme | Was passiert, wenn sie verletzt ist | Hier gemessen? |
|---|---|---|
| **Unabhängige, gleich verteilte Beobachtungen** | Bei Abhängigkeit (Cluster) sind die Block-Maxima nicht mehr unabhängig. | Nein |
| **Stationär, ohne Trend** | Ein Trend verschiebt die Blöcke gegeneinander. | Nein |
| **Ausgedachte Quellen mit exakter Verteilung** | Es gibt keine Messreihe aus Logistik, Lager oder Hafen; die Zahlen sagen nichts über echten Betrieb. | – |
| **Alle Blöcke gleich lang** (b teilt N = 18 000) | Gleitende Blöcke und Blockgrößen unter 6 sind nicht gemessen. | Nein |
| **Nebenbedingung ξ ≥ −1 bei ML** | Eine Wahl dieser Demo: für ξ < −1 ist die Likelihood unbeschränkt. Andere Software kann bei kleinem m andere Ränder haben. | Ja (Experiment 3) |
| **Klammer ξ̂ ∈ [−0,999; 0,99] bei L-Momenten** | Eine Wahl dieser Demo; ohne Klammer gibt es mehr oder gleich viele unmögliche Anpassungen (die Klammer greift nur bei ξ nahe −1 oder kleinem m). | Ja (Schalter, Experiment 3) |
| **Nur Punktschätzung** | Keine Intervalle, keine Überdeckung, kein Streuband gegen Wiederkehrperiode und m. | Nein |
"""

FORMULAS = r"""
**Gegeben und gesucht.** Gegeben sind $m$ Block-Maxima $x_1,\dots,x_m$ (jedes das Maximum von $b$ unabhängigen Beobachtungen); gesucht sind die Parameter $(\mu,\sigma,\xi)$ einer GEV und der Pegel zu einer
Überschreitungswahrscheinlichkeit $p$ je Block.

| Symbol | Bedeutung |
|---|---|
| $m$, $b$, $N=mb$ | Zahl der Block-Maxima, Beobachtungen je Block, Beobachtungen je Reihe ($N=18\,000$) |
| $\mu,\sigma>0,\xi$ | Lage, Skala und Formparameter der GEV |
| $q$ | Überschreitungswahrscheinlichkeit einer Beobachtung ($q=1/(2N)$) |
| $p_b$ | dieselbe Schwelle je Block: $p_b = 1-(1-q)^b$ |
| $V$ | geschätzter Pegel geteilt durch wahren Pegel; $\ln V$ ist der Fehler in der Log-Skala |

**GEV.** Auf $1+\xi(x-\mu)/\sigma>0$ gilt
$$G(x)=\exp\!\Big(-\big[1+\xi\tfrac{x-\mu}{\sigma}\big]^{-1/\xi}\Big),$$
und für $\xi=0$ der Grenzwert $G(x)=\exp(-\exp(-(x-\mu)/\sigma))$. Es gilt $\xi>0$ Fréchet-Typ, $\xi<0$ Weibull-Typ mit oberem Endpunkt $\mu-\sigma/\xi$ (`scipy.stats.genextreme`: $c=-\xi$).

**Pegel.** Zur Überschreitungswahrscheinlichkeit $p$ je Block:
$$x_p=\mu+\frac{\sigma}{\xi}\Big((-\ln(1-p))^{-\xi}-1\Big),\qquad x_p=\mu-\sigma\ln(-\ln(1-p))\ \ (\xi=0).$$

**Negative Log-Likelihood** mit $t_i=1+\xi(x_i-\mu)/\sigma>0$:
$$\ell(\mu,\sigma,\xi)=m\ln\sigma+\Big(1+\frac1\xi\Big)\sum_i\ln t_i+\sum_i t_i^{-1/\xi},\qquad \xi\ge -1 \text{ (Nebenbedingung dieser Demo)}.$$
Für $\xi<-1$ ist $\ell$ nach unten unbeschränkt, wenn der Endpunkt an das größte $x_i$ heranrückt (Smith 1985, nach Coles 2001, S. 55; Regularität: $\xi>-0{,}5$ regulär).

**L-Momente.** Mit $b_r=\frac1m\sum_j \binom{j-1}{r}\big/\binom{m-1}{r}\,x_{(j)}$ ist $l_1=b_0$, $l_2=2b_1-b_0$, $l_3=6b_2-6b_1+b_0$, $\tau_3=l_3/l_2$ (braucht $m\ge3$). Mit $k=-\xi$:
$$\tau_3=\frac{2(1-3^{-k})}{1-2^{-k}}-3,\qquad \sigma=\frac{l_2\,k}{(1-2^{-k})\,\Gamma(1+k)},\qquad \mu=l_1-\sigma\,\frac{1-\Gamma(1+k)}{k}.$$
$\tau_3$ wächst streng mit $\xi$; $\hat\xi$ folgt aus $\tau_3$ **exakt** (Bisektion), nicht aus der Näherung, und wird auf $[-0{,}999;\,0{,}99]$ geklammert (Wahl dieser Demo).

**Block-Bias.** Die Block-Maxima haben die Verteilung $F^b$ (exakt), nicht eine GEV; der Populations-Bias ist $\ln V$ der ML-Anpassung an $F^b$ im Grenzwert $m\to\infty$ (Quadratur mit 20 000 Quantilknoten),
nur wo er ein Grenzwert ist ($\xi_{\text{pop}}\ge0$). Es gilt $\text{lnRMSE}^2=\text{Bias}^2+\text{SD}^2$ in jeder Zelle.

**Literatur.** Coles (2001), Kap. 3 (GEV, Pegel, Blockgröße; gelesen). Embrechts, Klüppelberg, Mikosch (1997), 6.3 (gelesen). Smith (1985), Hosking/Wallis/Wood (1985), Hosking (1990), Cunnane (1978), Fisher/Tippett (1928):
nicht im Original gelesen, Befunde daraus „nach Zusammenfassung, ungeprüft“. Die Näherung von Hosking für ξ̂ aus τ₃ ist hier nur als Start der ML-Anpassung benutzt.
"""

SOURCE_NOTES = {
    "Exponential": "Exponentialverteilung (wahres ξ = 0, Gumbel-Bereich).",
    "Normal": "Normalverteilung (Gumbel-Bereich, aber langsam: der Pegel der Grenzform passt bei endlichem Block nur ungefähr).",
    "Pareto-Verteilung": "Pareto-Verteilung mit α = 4 (ξ = 1/α = 0,25, Fréchet-Bereich).",
    "Vehikel B": "Verweilzeit-Mischung (ausgedacht): Art, Kunde, Zoll, Wochenende, Vorlauf; exakte Aufzählung von 1 224 lognormalen Komponenten.",
    "Standard-Fréchet": "Standard-Fréchet exp(−1/x): max-stabil, F^b ist für jedes b eine Fréchet-GEV mit μ = σ = b, ξ = 1.",
    "Exakte GEV": "Die Block-Maxima sind direkt GEV(0, 1, ξ)-verteilt (es gibt keine Einzelbeobachtungen); m aus dem Regler.",
}

VEHIKEL_TEXT = (
    "Die Reihen sind erzeugt; in den gelesenen Quellen gibt es keine Messreihe aus Logistik, Lager oder Hafen, die hier nachgebildet würde. Das Verweilzeit-Modell stammt aus der Demo "
    "„Verweilzeit lernen statt annehmen“ und ist dort ebenfalls ausgedacht. Gezählt werden Blöcke, Beobachtungen und Läufe; was die Zahlen für einen echten Betrieb heißen, wird hier nicht behauptet. "
    "Aussagen über den Nutzen einer solchen Schätzung in Logistik, Lager oder Hafen sind nicht belegt."
)
