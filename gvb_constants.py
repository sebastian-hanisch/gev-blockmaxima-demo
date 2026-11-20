"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Zähleinheiten: m = Block-Maxima je Anpassung, b = Beobachtungen je Block, N = Beobachtungen je Reihe, Läufe = unabhängige Wiederholungen einer ganzen Reihe."""

N_OBS = 18000                      # Beobachtungen je Reihe (360 je Jahr, 50 Jahre)
Q_TARGET = 1.0 / (2 * N_OBS)       # Ziel: der Wert, den eine Beobachtung mit dieser Wahrscheinlichkeit überschreitet (im Mittel einmal in der doppelten Reihe)
BLOCKS = (6, 12, 30, 60, 90, 180, 360)   # Teiler von N; m = 3 000 ... 50
P_HUNDERT = 0.01                   # Experimente 2 und 3: Pegel zur Überschreitungswahrscheinlichkeit 0,01 je Block (100er-Wert), Fehler in σ
PARETO_ALPHA = 4.0                 # Pareto-Verteilung der Quelle (Experiment 1): P(X > x) = x^(-α), x ≥ 1; ξ = 1/α = 0,25
SEED_MAX, DEFAULT_SEED = 999999, 35

SOURCES = ("Exponential", "Normal", "Pareto-Verteilung", "Vehikel B", "Standard-Fréchet", "Exakte GEV")
SOURCES4 = ("Normal", "Exponential", "Vehikel B")      # eingefrorener Satz der Blockgrößen-Tabelle
XI_OPTIONS = (-1.2, -0.9, -0.8, -0.6, -0.3, 0.0, 0.2, 0.5)
XI_GRID = (-0.9, -0.8, -0.6, -0.3, 0.0, 0.2, 0.5)     # ξ der eingefrorenen Tabelle (ξ = -1,2 nur live)
M_OPTIONS = (10, 20, 50, 100, 500)
M_GRID = (10, 20, 50, 100)
RUNS_MIN, RUNS_MAX, RUNS_STEP, DEFAULT_RUNS = 10, 100, 10, 40

BAND_SE = 4.0                      # Bänder der Tests: ± so viele Standardfehler

COLOR_TRUE = "#2F6B65"
COLOR_FIT = "#d95f02"
COLOR_ALT = "#4c78a8"
COLOR_PTS = "rgba(80,80,80,0.75)"

DEFAULTS = {
    "quelle_sel": "Normal", "b_sl": 60, "verfahren_sel": "ML", "seed_input": DEFAULT_SEED,
    "xi_sl": 0.0, "m_sl": 50, "runs_sl": DEFAULT_RUNS, "klammer_tg": True, "quelle4_sel": "Normal",
}

PRESETS = {
    "Normal, Blockgröße b = 60 (Voreinstellung)": {},
    "Exponential: Gumbel passt (b = 30)": {"quelle_sel": "Exponential", "b_sl": 30, "verfahren_sel": "Gumbel", "quelle4_sel": "Exponential"},
    "Normal, kleiner Block (b = 6)": {"quelle_sel": "Normal", "b_sl": 6, "quelle4_sel": "Normal"},
    "Normal, Jahresblock (b = 360)": {"quelle_sel": "Normal", "b_sl": 360, "quelle4_sel": "Normal"},
    "Vehikel B, b = 12: der Pegel kippt nach oben": {"quelle_sel": "Vehikel B", "b_sl": 12, "quelle4_sel": "Vehikel B"},
    "Kleine Reihe, m = 10": {"quelle_sel": "Exakte GEV", "xi_sl": 0.0, "m_sl": 10},
    "ξ = −0,9: ML klebt am Rand": {"quelle_sel": "Exakte GEV", "xi_sl": -0.9, "m_sl": 20},
    "Fréchet exakt (max-stabil)": {"quelle_sel": "Standard-Fréchet", "b_sl": 60, "quelle4_sel": "Normal"},
}
# Hinweis: die Zahlen der Hilfetexte stehen im Test `tests/test_verdicts.py::test_preset_aussagen_der_hilfetexte_gegen_die_tabelle`.
PRESET_HELP = {
    "Normal, Blockgröße b = 60 (Voreinstellung)": "Normalverteilte Beobachtungen, Blöcke zu 60 (m = 300 Block-Maxima), ML-Anpassung: der geschätzte Pegel gegen die Wahrheit aus F^b.",
    "Exponential: Gumbel passt (b = 30)": "Exponential-Quelle mit wahrem ξ = 0, Blöcke zu 30 (m = 600), Gumbel mit festem ξ = 0: schlägt die freie GEV bei allen sieben Blockgrößen, solange ξ = 0 vorab richtig bekannt ist.",
    "Normal, kleiner Block (b = 6)": "Viele kleine Blöcke (m = 3 000): kleine Streuung, aber ein Bias nach unten (−0,07 in ln bei m = 3 000), der mit wachsendem m schrumpft; die GEV ist bei b = 6 nur Näherung von F^b.",
    "Normal, Jahresblock (b = 360)": "Wenige große Blöcke (m = 50): kleiner Bias (−0,01 in ln), dafür größere Streuung (SD 0,066 gegen 0,040 bei b = 30).",
    "Vehikel B, b = 12: der Pegel kippt nach oben": "Verweilzeit-Mischung (ausgedacht): bei b = 12 liegt die freie GEV im Grenzwert m → ∞ um 67 % über dem wahren Pegel.",
    "Kleine Reihe, m = 10": "Zehn Block-Maxima, exakt GEV mit ξ = 0: ξ̂ streut so stark, dass die Standardabweichung an wenigen Läufen hängt; Median und IQR lesen.",
    "ξ = −0,9: ML klebt am Rand": "Zwanzig Block-Maxima mit Weibull-Typ ξ = −0,9: ML endet oft an der Nebenbedingung ξ = −1, die L-Momente enden manchmal unter dem größten Wert.",
    "Fréchet exakt (max-stabil)": "Standard-Fréchet ist max-stabil: F^b ist wieder eine Fréchet-GEV, die Blockgröße verursacht keinen Block-Bias (Kontrolle in Experiment 4).",
}
