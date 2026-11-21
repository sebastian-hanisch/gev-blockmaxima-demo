# GEV an Blockmaxima: wie viel eine kurze Reihe verrät – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-gev-blockmaxima-demo.streamlit.app/)**

Demo der Linie **„Extremwerte erklärt“** in der „Konzepte“-Reihe der Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Bereich Statistik, Feld „Seltenes und Extremes“). Die Linie fragt, was sich über den Rand einer Verteilung sagen lässt, also über einen Wert, den eine Beobachtung nur selten überschreitet. Diese Demo behandelt die **Block-Maxima-Methode**: Aus **m Block-Maxima** werden die Parameter (μ, σ, ξ) einer **GEV** (verallgemeinerte Extremwertverteilung) geschätzt und daraus der **Pegel** abgelesen. Die Annahme „die Einzelverteilung F ist bekannt“, bei der das Maximum von n Beobachtungen exakt F^n folgt, wird hier aufgehoben: In der Hand sind nur m Zahlen, und gefragt ist, wie viel sie über den Rand verraten.

**Vehikel.** Die Reihen sind erzeugt; in den gelesenen Quellen gibt es keine Messreihe aus Logistik, Lager oder Hafen, die hier nachgebildet würde. Das Verweilzeit-Modell stammt aus der Demo „Verweilzeit lernen statt annehmen“ und ist dort ebenfalls ausgedacht. Gezählt werden Blöcke, Beobachtungen und Läufe; was die Zahlen für einen echten Betrieb heißen, wird hier nicht behauptet. Aussagen über den Nutzen einer solchen Schätzung in Logistik, Lager oder Hafen sind nicht belegt. Die Quellen sind Exponential, Normal, Pareto-Verteilung (α = 4), Standard-Fréchet, eine exakte GEV und **Vehikel B**, die Verweilzeit-Mischung der `verweilzeit-demo` als exakte Aufzählung von 1 224 lognormalen Komponenten. Die Wahrheit ist der **exakte Pegel der erzeugenden Verteilung**: F^b bei den Quellen mit Einzelbeobachtungen, die erzeugende GEV bei „Exakte GEV“ (Experimente 2, 3 und 5). Verwandt ist `lastspitze-kappen-demo`: Dort ist das Monatsmaximum der Last ein Kostenträger bei sechs erzeugten Monaten und ohne Verteilungsform; hier ist die Anpassung an die Block-Maxima der Gegenstand. Lehrbuch-Beispiele (Deich, Versicherung) gelten nur als Lehrbuch/Abstract.

## Warum dieses Problem

Wer den Wert abschätzen will, den eine Beobachtung nur mit der Wahrscheinlichkeit q = 1/(2N) überschreitet (N = 18 000 Beobachtungen: q = 2,78·10⁻⁵, „im Mittel einmal in der doppelten Reihe“), muss über den Datenbereich hinaus **extrapolieren**. Die Block-Maxima-Methode teilt die Reihe in m = N/b Blöcke zu b Beobachtungen, behält je Block das Maximum und passt eine GEV an. Das wirft zwei Fragen auf, die sich widersprechen: Kleine Blöcke geben viele Maxima (kleine Streuung), aber die GEV ist bei kleinem b nur Näherung von F^b (**Block-Bias**); große Blöcke haben einen kleineren Block-Bias (bei der Exponentialquelle und b = 360: 0,0002 in ln; bei Vehikel B bleiben dort im Grenzwert m → ∞ noch 0,045), aber nur wenige Maxima (große Streuung). Die Demo misst das als Zerlegung in Bias und Streuung über Läufe und zeigt außerdem, wie grob ξ̂ aus wenigen Maxima zu lesen ist und wo die Anpassung versagt.

**Kernfragen.** (1) Was schätzt die Anpassung aus diesen Blöcken, und wo liegt die Wahrheit aus F^b? (2) Wie viele Blöcke braucht man, bis ξ̂ etwas sagt? (3) Wo versagt die Anpassung? (4) Welche Blockgröße ist am besten, und woran hängt das?

## Befunde und Korrekturen

Die Zahlen der Vormessung (numpy `default_rng`, Läufe: 1 000 für die Anpassung, 200 für die Blockgröße) sind Erwartungen. Im Bau wurden sie mit dem SplitMix64-Generator der Vorlage (`gvb_rng.py`) und anderen Zufalls-Seeds neu gemessen; alle verglichenen Zellen liegen im Band von vier Standardfehlern (`tests/test_vormessung_baender.py`), aber einzelne Zahlen verschieben sich:

| Aussage | Vormessung (alt) | Bau (neu, Läufe in Klammern) | Folge |
|---|---|---|---|
| L-Momente haben bei m ≤ 20 die kleinere SD von ξ̂ als ML | 9 von 12 Zellen (nicht bei ξ = −0,9 und nicht bei ξ = −0,6, m = 20) | 10 von 12 Zellen (1 000 Läufe je Zelle; Ausnahmen nur ξ = −0,9 bei m = 10 und m = 20) | bei ξ = −0,6 und m = 20 liegen beide SD innerhalb des Standardfehlers (0,209 gegen 0,216); die Zählung bleibt, der Satz nennt das Raster |
| Spanne der unmöglichen L-Moment-Anpassungen (Endpunkt unter dem größten Wert) bei ξ ≤ −0,6, m ≥ 20, mit Klammer | 10 bis 28 %, ohne Klammer 15 bis 35 % | für ξ = −0,6 und −0,9 bei m = 20, 50, 100: 8 % bis 28 % mit Klammer, 14 % bis 35 % ohne Klammer (1 000 Läufe; die Ränder sind Zufallsextreme über sechs Zellen: ein zweiter Lauf der unabhängigen Aussagenprüfung mit anderem Zufalls-Seed gab 9,9 % bis 29,8 % und 13,9 % bis 36,9 %, also etwa ± 2 Punkte) | **Ungenauigkeit U1 der Zweitprüfung**: Die Spanne gehört zum ξ-Raster, nicht zu „ξ ≤ −0,6“: bei ξ = −0,8 und m = 100 sind es 31 % mit und 31 % ohne Klammer (Zweitprüfung: 32,9 % und 33,3 % bei 400 000 Läufen, Standardfehler hier rund 1,5 Punkte); die App nennt Raster und „bis zu einem Drittel“ |
| SD von ξ̂ bei m = 10 (ML, ξ = 0 / 0,2 / 0,5) | 0,532 / 0,764 / 0,821 | 0,591 / 0,596 / 1,081; Median −0,029 / 0,202 / 0,582; IQR/1,349 0,428 / 0,410 / 0,494 (1 000 Läufe) | Die SD ist ein Ausreißer-Maß (die Reihenfolge über ξ ändert sich von Messung zu Messung, die Zweitprüfung fand 0,62 / 0,64 / 0,90, ein zweiter Lauf der Aussagenprüfung 0,488 / 0,645 / 0,915): die drei SD-Zahlen sind keine reproduzierbaren Messwerte, gelesen werden Median und IQR/1,349 |
| Blockgröße, Normal, b = 360: SD von ln V | 0,079 (200 Läufe) | Bias −0,010 bei SD 0,066 (1 000 Läufe) | im Band von vier Standardfehlern, aber die SD aus 200 Läufen ist bei großen Blöcken unsicher; die Tabelle der Blockgröße hat deshalb 1 000 statt 200 Läufe je Zelle |
| ML-Laufzeit je Anpassung | rund 28 bis 39 ms (Plan, ausgelastete Maschine) | rund 9 ms je Lauf mit ML und L-Momenten bei m = 50 beim Bau, rund 20 ms bei der Korrektur auf einem durch andere Arbeit ausgelasteten Rechner (Messungen dieser README, nicht im Test) | „200 Läufe in rund 6 s“ wird zu rund 2 bis 4 s; ein Newton-Verfahren war nicht nötig |
| Populations-Bias der Fréchet-Quadratur | ξ̂ = 1,0002, kein Block-Bias | ξ̂ = 1,0002, ln V unter 0,002 bei b = 6, 60, 360 | bestätigt; die Abweichung von null ist der Quadraturfehler |
| Normal, ein einzelnes sehr großes m (b = 6; m = 3 000 / 30 000 / 300 000) | ln V −0,054 / −0,067 / −0,031 | −0,04 / −0,05 / −0,03 (je eine Stichprobe, Zufalls-Seed 2026); im Mittel über Zufalls-Seeds −0,08 / −0,06 / −0,03 (30 / 15 / 5 Zufalls-Seeds, SD 0,03 / 0,02 / 0,02; die ersten beiden als Band im Test, die 5 Zufalls-Seeds bei m = 300 000 dauern zu lange und sind nicht im Test) | die Zahl hängt an der einzelnen Stichprobe und ist als Grenzwert nicht ablesbar (siehe Befund 8) |
| ML-GEV bei m = 20, ξ = 0: SD des Pegel-Fehlers | 2,98 (V1), 8,00 (V11), 9,02 (Prüfer) | 2,56 (2 000 Läufe); Median −0,38, IQR/1,349 1,89 (zweiter Lauf der Aussagenprüfung: 2,66 / −0,28 / 1,87) | Ausreißer-Maß: SD und Median streuen von Lauf zu Lauf (SD rund 2,6 bis 2,7, Median −0,3 bis −0,4), gelesen wird der IQR/1,349 (rund 1,9) |

Die Nebenbedingung ξ ≥ −1 bei ML und die Klammer ξ̂ ∈ [−0,999; 0,99] bei den L-Momenten sind **Wahlen dieser Demo**, keine Eigenschaften von scipy oder anderer Software; die Zahlen „ML-Rand“ und „unmögliche Anpassungen“ hängen daran.

## Modell

* **Konvention.** ξ > 0 Fréchet-Typ, ξ = 0 Gumbel, ξ < 0 Weibull-Typ mit oberem Endpunkt μ − σ/ξ. `scipy.stats.genextreme` hat **c = −ξ** (der Test `tests/test_oracle_gev.py` prüft das, mit c = +ξ gibt scipy einen anderen Wert); Hosking schreibt k = −ξ. Der Extremwertindex ξ ist nicht der Extremalindex θ (θ kommt hier nicht vor). „Pegel“ heißt der geschätzte Wiederkehrwert zu einer Überschreitungswahrscheinlichkeit.
* **GEV und Pegel.** G(x) = exp(−[1 + ξ(x − μ)/σ]^(−1/ξ)) auf 1 + ξ(x − μ)/σ > 0; Pegel zu p je Block: x_p = μ + σ/ξ((−ln(1 − p))^(−ξ) − 1). Das Ziel q = 1/(2N) je Beobachtung entspricht p_b = 1 − (1 − q)^b je Block. Wahre Pegel (exakt aus F): Normal 4,0309, Exponential 10,4913, Vehikel B 221,36 Tage.
* **Zähleinheiten.** m Block-Maxima je Anpassung, b Beobachtungen je Block, N = 18 000 Beobachtungen je Reihe, **Läufe** = unabhängige Wiederholungen einer ganzen Reihe (jede Streuung ist eine Streuung über Läufe). V = geschätzter Pegel / wahrer Pegel; lnRMSE = Wurzel des Mittels von (ln V)²; es gilt lnRMSE² = Bias² + SD² in jeder Zelle (SD mit Nenner R).
* **Verfahren.** *ML* (eigener Nelder-Mead nur mit numpy, Nebenbedingung ξ ≥ −1, **keine obere Grenze** für ξ, Start aus den L-Momenten), *L-Momente* (ξ̂ aus τ₃ **exakt** zurückgerechnet, nicht per Näherung, mit der Klammer ξ̂ ∈ [−0,999; 0,99], Schalter „Klammer“), *Gumbel* (ξ = 0 fest, ML).
* **Zufall.** Aus der reinen Vorlage `gvb_rng.py` (SplitMix64, Kopier-Test): ein Lauf braucht N = 18 000 gleichverteilte Zahlen; das Maximum eines Blocks gehört zum Minimum seiner Überlebenswahrscheinlichkeiten v = 1 − u, und **eine** Umkehrung gibt das Block-Maximum. Zufallsanteile sind Messungen mit Standardfehler; Neu-Rechnen-Tests prüfen als Band, nie auf das letzte Bit.
* **Gilt für** unabhängige, gleich verteilte Beobachtungen ohne Trend, N = 18 000, Blockgrößen b = 6 bis 360 (Teiler von N).

## Methodik

Fünf Experimente in der App (Seitenleiste: Schnellstart mit acht Szenarien und die Regler):

1. **Eine Reihe anpassen.** Quelle, Blockgröße, Verfahren, Zufalls-Seed: Pegel-Diagramm (Block-Maxima, angepasste Kurve, wahre Kurve aus F^b), V und ξ̂. Daneben für Normal, Exponential und Vehikel B die eingefrorene Streuung über 1 000 Läufe.
2. **Wie viele Blöcke braucht ξ̂?** Live bis zu 100 Läufe mit exakten GEV(0, 1, ξ)-Blöcken (Spinner mit Zeitangabe), ML und L-Momente nebeneinander; die Tabelle mit 1 000 Läufen je Zelle (ξ = −0,9, −0,8, −0,6, −0,3, 0, 0,2, 0,5; m = 10, 20, 50, 100) ist eingefroren.
3. **Wo versagt die Anpassung?** Anteil ML am Rand ξ = −1, Anteil L-Moment-Anpassungen mit Endpunkt unter dem größten Wert (Schalter Klammer ja/nein), ein Beispiel mit fünf Werten, und die Likelihood ohne Grenze (Handbeispiel).
4. **Blockgröße.** Je Quelle (Normal, Exponential, Vehikel B) und b = 6, 12, 30, 60, 90, 180, 360 Bias, SD und lnRMSE der Läufe (1 000 Läufe je Zelle, eingefroren), den Populations-Bias per Quadratur (live, 20 000 Knoten, nur für Quellen mit ξ_pop ≥ 0), ein Lauf live; dazu die Kontrolle „Standard-Fréchet ist max-stabil“.
5. **Alte Handverfahren** (Tabelle): Gumbel-Regression mit Positionen nach Weibull und nach Gringorten, Gumbel-ML, Kleinste-Quadrate-GEV, jeweils 2 000 Läufe.

**Eingefrorene Daten** (Quelle und Stand je Teil in `gvb_tables.json` unter „meta“; Erzeugung und Prüfung mit `tools/gvb_make_tables.py`, Teile `mixture-inv`, `fit`, `block`, `positions`, `--check` rechnet einen Ausschnitt neu und prüft im Band, `--lang` den ganzen Satz). Die Verweilzeit-Mischung liegt als unveränderte Kopie `gvb_mixture.json` der gemeinsamen Tabelle der Reihe vor (1 224 Komponenten, SHA-256 im Test; Quelle und Stand je Feld in `gvb_mixture_quelle.json`).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: eigene Messung mit SplitMix64, unabhängige Beobachtungen, N = 18 000 bzw. exakte GEV(0, 1, ξ)-Blöcke, **1 000 Läufe je Zelle** (Blockgröße ebenso, Handverfahren 2 000), Pegel zu q = 1/(2N) bzw. 100er-Wert (p = 0,01 je Block). Die Tabelle „Zahl → Test“ folgt darunter.

| Nr. | Frage | Befund (Läufe, Voraussetzung) |
|---|---|---|
| 1 | Streuung von ξ̂ bei exakter GEV (ML) | ξ = 0: SD von ξ̂ 0,240 / 0,114 / 0,078 bei m = 20 / 50 / 100; ξ = 0,2: 0,266 / 0,131 / 0,088 (1 000 Läufe je Zelle; die SD streut selbst, auf die dritte Stelle ist sie nicht stabil: ein zweiter Lauf mit anderem Zufalls-Seed gab für ξ = 0 0,248 / 0,116 / 0,076 und für ξ = 0,2 0,247 / 0,135 / 0,089). Bei m = 10 steht die SD (0,591 / 0,596 / 1,081 für ξ = 0 / 0,2 / 0,5; im zweiten Lauf 0,488 / 0,645 / 0,915, also keine reproduzierbare Messgröße) neben Median (−0,029 / 0,202 / 0,582) und IQR/1,349 (0,428 / 0,410 / 0,494; im zweiten Lauf 0,415 / 0,476 / 0,487), weil wenige Läufe mit riesigem ξ̂ (größtes beobachtetes ξ̂ 7,1, der Optimierer hat keine obere Grenze) sie bestimmen: im zweiten Lauf hatten 1 / 8 / 27 von 1 000 Läufen ξ̂ über 2, ohne sie sinkt die SD auf 0,482 / 0,510 / 0,535 |
| 2 | ML gegen L-Momente bei kleinem m | Die L-Momente haben bei m ≤ 20 die kleinere SD von ξ̂ in 10 von 12 Zellen (Ausnahmen: ξ = −0,9 bei m = 10 und m = 20; bei ξ = −0,6, m = 20 liegen beide SD innerhalb des Standardfehlers: 0,209 gegen 0,216), bei m = 100 in 0 von 6 (bei ξ = 0 sind beide SD gleich: 0,078 gegen 0,078); bei ξ = 0, m = 50 liegen sie nahe beieinander (0,106 gegen 0,114; die L-Momente knapp kleiner, im zweiten Lauf ebenso). Das gilt für die Klammer-Fassung der L-Momente und ML mit ξ ≥ −1. Bei ξ = 0,5 sind die L-Momente verzerrt (Mittel von ξ̂ bei m = 50, L-Momente gegen ML: 0,430 gegen 0,506) |
| 3 | ML am Rand ξ = −1 (Optimierer, Nebenbedingung dieser Demo) | ξ = −0,6 bei m = 10 / 20 / 50: 49 % / 20 % / 1 % der Läufe; ξ = −0,9 bei m = 10 / 20 / 50 / 100: 81 % / 73 % / 52 % / 30 % |
| 4 | L-Momente mit Endpunkt unter dem größten Wert (unmögliche Anpassung) | ML: 0 von 28 Zellen (die Likelihood verbietet es). L-Momente für ξ = −0,6 und −0,9 bei m = 20, 50, 100: 8 % bis 28 % mit Klammer, 14 % bis 35 % ohne Klammer (Zufallsextreme über sechs Zellen mit Standardfehler 0,9 bis 1,5 Punkte, im zweiten Lauf 9,9 % bis 29,8 % und 13,9 % bis 36,9 %); ξ = −0,9, m = 20: 8 % mit, 23 % ohne Klammer; die SD von ξ̂ ist dort bei ML 0,137, bei den L-Momenten 0,190 mit und 0,279 ohne Klammer: „L-Momente sind stabiler“ gilt dort auch mit der Klammer nicht (die Ausnahme aus Befund 2). Ohne Klammer gibt es in 13 von 28 Zellen mehr unmögliche Anpassungen und in 15 gleich viele, in keiner weniger. Die Spanne gehört zum ξ-Raster; bei ξ = −0,8, m = 100: 31 %. Sie wächst mit m und liegt für m ≤ 100 bei bis zu einem Drittel; bei m = 500 (nur live) lag der Anteil für ξ = −0,9 bei der Korrektur bei rund 42 % (300 Läufe, Standardfehler 2,8 Punkte; Messung dieser Korrektur, nicht im Test) |
| 5 | Likelihood ohne Grenze | Drei Werte (1, 2, 3), ξ = −1,5, σ = 1: rückt der Endpunkt je Zehnerpotenz näher an den größten Wert, sinkt die negative Log-Likelihood zuletzt um (1 + 1/ξ)·ln 10 = 0,7675 (Handbeispiel; die Schritte je Zehnerpotenz sind ab Abstand 10⁻¹ 1,17 / 0,83 / 0,78 / 0,77 und nähern sich dem Wert nur an, ab Abstand 10⁻³ liegen sie weniger als 0,02 daneben); bei ξ = −0,8 steigt sie dagegen |
| 6 | Blockgröße, Normal | Kleinster lnRMSE der freien GEV bei b = 30 (m = 600): 0,050, praktisch gleich bei b = 60 (0,051; der Abstand liegt im Rauschen); b = 6: Bias −0,074 bei SD 0,030 (ln V, Läufe), b = 360: Bias −0,010 bei SD 0,066. Gumbel (ξ = 0 fest) ist nur bei b = 360 besser als die freie GEV (0,058 gegen 0,067), in 1 von 7 Blockgrößen |
| 7 | Blockgröße, Exponential (wahres ξ = 0) | Gumbel hat den kleineren lnRMSE als die freie GEV in 7 von 7 Blockgrößen (0,026 gegen 0,070 bei b = 30), **vorausgesetzt ξ = 0 ist vorab richtig bekannt**. Auch bei wahrem ξ = 0 hat die freie GEV (ML) bei b = 6 einen Bias von +0,142 (ln; Grenzwert m → ∞: +0,14), Gumbel mit festem ξ = 0 dort −0,068: beide sind nur Näherungen von F^b |
| 8 | Normal: Populations-Bias | **Kein auf dieser Quadratur ablesbarer Grenzwert**: bei b = 6 gibt die Quadratur ln V −0,133 / −0,082 / −0,018 / 0,066 / 0,128 für die Logit-Ränder S = 6 / 9,2 / 13,8 / 20,7 / 30 und steigt weiter (S = 60 / 100 / 200 / 400: 0,217 / 0,264 / 0,311 / 0,344); sie hängt am Abschneiden des Tails (der angepasste Endpunkt liegt am letzten Knoten). Dass es für m → ∞ gar keinen Grenzwert gibt, ist damit nicht gezeigt: die Zunahme wird langsamer. Eine einzelne sehr große Stichprobe (b = 6, m = 3 000 / 30 000 / 300 000) gibt für Zufalls-Seed 2026 ln V −0,04 / −0,05 / −0,03; im Mittel über Zufalls-Seeds sind es −0,08 / −0,06 / −0,03 (30 / 15 / 5 Zufalls-Seeds, SD 0,03 / 0,02 / 0,02), die Einzelwerte liegen also im Rauschen und sagen weder „fällt“ noch „steigt“; nur der Betrag des Mittelwerts schrumpft mit m |
| 9 | Blockgröße, Vehikel B | Die freie GEV ist im Grenzwert m → ∞ bei b = 12 um 67 % (Faktor 1,67), bei b = 30 um 62 % zu hoch (Exponential und Vehikel B haben ξ_pop ≥ 0); das kleinste lnRMSE liegt am Rand des gemessenen Bereichs, bei b = 6 (0,236); bei b = 12 und 30 0,530 und 0,523. Daneben hat die Kurve ein flaches lokales Minimum im Inneren bei b = 180 (0,274 gegen 0,297 bei b = 360); die SD wächst monoton mit b. Gumbel mit ξ = 0 fest erreicht im Median nur das 0,25-Fache (b = 6) bis 0,71-Fache (b = 360) des wahren Pegels (bei b = 12 bis 30: das 0,31- bis 0,41-Fache). Kleinere Blöcke als b = 6 sind nicht gemessen |
| 10 | Quadratur gegen Simulation | Für Exponential und Vehikel B stimmt der Bias der Läufe mit dem Grenzwert m → ∞ überein: 14 von 14 Zellen innerhalb 0,03 (größte Abweichung unter 0,02) |
| 11 | Max-Stabilität (Standard-Fréchet) | F^b ist GEV(μ = b, σ = b, ξ = 1) für jedes b (exakt); die Populations-ML findet ξ̂ = 1,0002, kein Block-Bias (ln V unter 0,002, b = 6, 60, 360) |
| 12 | Handverfahren, ξ = 0, m = 50 | Gumbel-Regression mit Positionen nach Weibull i/(m + 1) ist um 0,288 σ verzerrt, mit Gringorten (i − 0,44)/(m + 0,12) um 0,007 σ (Standardfehler 0,015 σ, von null nicht zu unterscheiden) bei RMSE 0,68 gegen 0,56 σ (Gringorten gegen Gumbel-ML); Gumbel-ML bei wahrem ξ = 0,2 (ebenfalls m = 50, 100er-Wert, p = 0,01 je Block): Bias −2,36 σ, in 99 % der Läufe zu niedrig; bei ξ = −0,2: 1,25 σ, in 99,8 % der Läufe zu hoch (2 000 Läufe). „Positionen nach Weibull“ hat mit dem Weibull-Typ der GEV nichts zu tun |
| 13 | Vehikel B als exakte Mischung | Je Container p = 10⁻⁵: 287,158 Tage; Tagesmaximum aus 100 Containern, 0,999-Quantil (einmal in 1 000 Tagen, je Container p = 1,0005·10⁻⁵): 287,123 Tage; P(Verweilzeit > 30 Tage) = 0,009830; aus der kopierten Tabelle, geprüft gegen eine eigene Aufzählung (4 704 Schritte, 2 376 mit Gewicht, 1 224 Komponenten) |

Die **Handbeispiele** des Plans sind exakte Tests (`tests/test_handbeispiele.py`): nll(0, 1 | μ = 0, σ = 1, ξ = 0,5) = 2,66084 (mit scipy c = +ξ ein anderer Wert), Pegel 17,949853 (ξ = 0,5, p = 0,01) und 4,600149 (Gumbel); fünf Block-Maxima 2, 3, 5, 6, 9 haben l₁ = 5, l₂ = 17/10, l₃ = 3/10, τ₃ = 3/17 (Brüche, zusätzlich über alle 10 Paare und 10 Tripel) und drei Pegel aus fünf Zahlen: 15,01 (L-Momente, GEV), 14,87 (L-Momente, Gumbel), 12,98 (Gumbel-ML), alle über dem größten der fünf Werte (9).

### Zahl → Test (Band oder exakt)

| Zahl der README | Test | Art |
|---|---|---|
| nll 2,66084, Pegel 17,949853 / 4,600149, c = −ξ | `tests/test_handbeispiele.py::test_h1_nll_von_zwei_werten_von_hand`, `tests/test_handbeispiele.py::test_h2_pegel_von_hand`, `tests/test_oracle_gev.py::test_nll_gleich_minus_summe_der_scipy_logpdf_mit_c_gleich_minus_xi` | exakt / 1e-14 |
| l₁ = 5, l₂ = 17/10, l₃ = 3/10, τ₃ = 3/17; Pegel 15,01 / 14,87 / 12,98 | `tests/test_handbeispiele.py::test_h3_l_momente_als_brueche_und_ueber_paare_und_tripel`, `tests/test_handbeispiele.py::test_h3_rueckrechnung_reproduziert_die_l_momente_und_drei_pegel` | exakt (Brüche) / 0,01 |
| Klammer am Beispiel 0, 3, 3, 3, 4 (ξ̂ −1,495, Endpunkt 3,84 gegen 4,20) | `tests/test_handbeispiele.py::test_h3b_die_klammer_am_beispiel_0_3_3_3_4` | exakt / 1e-2 |
| 0,7675 je Zehnerpotenz (Likelihood ohne Grenze) | `tests/test_handbeispiele.py::test_h6_likelihood_ohne_grenze` | 0,02 |
| Plotting-Positionen (10/11, 0,9447, G(E X₍ᵢ₎)) | `tests/test_handbeispiele.py::test_h7_plotting_positionen_weibull_gringorten` | 6e-4 |
| Blockgröße beim Exponential (7381/2520 u. a.), Max-Stabilität als Formel | `tests/test_handbeispiele.py::test_h4_blockgroesse_exponential_harmonische_zahlen`, `tests/test_handbeispiele.py::test_h5_frechet_ist_max_stabil_und_gleich_nicht` | exakt |
| ML so tief wie `genextreme.fit`, Gumbel-ML gleich `gumbel_r.fit` | `tests/test_oracle_gev.py::test_ml_so_tief_wie_scipy_genextreme_fit_m100_20_stichproben`, `tests/test_oracle_gev.py::test_gumbel_ml_gleich_scipy_gumbel_r_fit_auf_6_stellen` | 1e-6 / 1e-5 |
| L-Momente gegen Quadratur, Paare und Tripel; Hosking-Näherung unter 1e-3 | `tests/test_oracle_gev.py::test_l_momente_population_gegen_scipy_quadratur`, `tests/test_oracle_gev.py::test_l_momente_stichprobe_gegen_paar_und_tripel_aufzaehlung`, `tests/test_oracle_gev.py::test_rueckrechnung_gegen_hosking_naeherung_und_probe_am_ergebnis` | 2e-6 / 1e-10 / 1e-3 |
| Pegel gegen Bisektion und scipy `isf` (Coles, Gl. (3.4)) | `tests/test_literatur_gev.py::test_literatur_coles_wiederkehrwert_3_4_gegen_scipy_isf`, `tests/test_handbeispiele.py::test_h2_pegel_gegen_bisektion_auf_der_verteilungsfunktion` | 1e-13 relativ |
| Befunde 1 bis 4 (Zellen der Tabelle A) gegen die Vormessung | `tests/test_vormessung_baender.py::test_tabelle_a_ml_und_l_momente_im_band_der_vormessung_v11`, `tests/test_tabellen_und_baender.py::test_neu_rechnen_fit_zelle_als_band_xi_0_m_50`, `tests/test_tabellen_und_baender.py::test_neu_rechnen_rand_anteile_xi_minus_0_9_m_20_als_band` | Band ± 4 Standardfehler |
| Zählungen 10 von 12, 0 von 6, 1 von 7, 7 von 7 und alle Bausteine aus den Tabellen | `tests/test_claims.py::test_jeder_messbaustein_steht_woertlich_in_der_readme`, `tests/test_claims.py::test_baustein_zaehlung_stimmt_mit_den_tabellen_ueberein` | exakt (aus der eingefrorenen Tabelle) |
| Befunde 6, 7, 9 (Blockgröße) gegen die Vormessung; lnRMSE² = Bias² + SD² | `tests/test_vormessung_baender.py::test_tabelle_b_blockgroesse_im_band_der_vormessung_v11`, `tests/test_eigenschaften_und_randfaelle.py::test_ln_rmse_quadrat_gleich_bias_quadrat_plus_sd_quadrat_in_jeder_zelle`, `tests/test_tabellen_und_baender.py::test_neu_rechnen_blockgroesse_normal_b60_als_band` | Band / 1e-9 |
| Befund 8 und 10 (Quadratur) | `tests/test_oracle_population.py::test_normal_hat_keinen_grenzwert_das_ergebnis_haengt_am_abschneiden`, `tests/test_oracle_population.py::test_quadratur_gegen_simulation_exponential_und_vehikel_b_14_von_14_zellen`, `tests/test_oracle_population.py::test_exponential_grenzwert_konvergiert_im_logit_rand`, `tests/test_oracle_population.py::test_exponential_b24_mit_dem_ziel_der_vormessung_1_gegen_V1`, `tests/test_oracle_population.py::test_xi_pop_der_quellen_mit_grenzwert_ist_nicht_negativ`, `tests/test_oracle_population.py::test_normal_quadratur_steigt_weiter_bei_groesserem_abschneiden_zunahme_wird_langsamer`, `tests/test_oracle_population.py::test_normal_einzelne_grosse_stichproben_mittel_ueber_zufalls_seeds_schrumpft_mit_m` | 0,03 / 0,002 / 0,01 |
| Befund 11 (Max-Stabilität) | `tests/test_oracle_population.py::test_frechet_ist_max_stabil_kein_block_bias`, `tests/test_oracle_population.py::test_frechet_f_hoch_b_gleich_gev_exakt` | 2e-3 / 1e-14 |
| Befund 12 (Handverfahren) | `tests/test_vormessung_baender.py::test_tabelle_c_handverfahren_im_band_der_vormessung_v1`, `tests/test_vormessung_baender.py::test_ml_gev_bei_m_20_sd_ist_ein_ausreisser_mass_median_und_iqr_sind_stabil` | Band |
| Befund 13 (Vehikel B: 287,158 / 287,123 / 0,009830 / 1 224 Komponenten) | `tests/test_oracle_mixture.py::test_aufzaehlung_4704_schritte_2376_mit_gewicht_1224_komponenten`, `tests/test_oracle_mixture.py::test_kopierte_tabelle_gleich_eigener_aufzaehlung`, `tests/test_oracle_mixture.py::test_ueberlebensfunktion_gegen_scipy_und_die_benannten_zielgroessen`, `tests/test_oracle_mixture.py::test_umkehrtabelle_gegen_wurzelsuche` | 5e-4 / 1e-12 / 1e-6 |
| Wahre Pegel (Normal, Exponential, Vehikel B) | `tests/test_oracle_mixture.py::test_pegel_der_quellen_aus_f` | 5e-5 |
| Kopien (Zufallsgenerator, Tabelle, Konstanten der `verweilzeit-demo`) | `tests/test_copies.py::test_rng_ist_die_unveraenderte_vorlage`, `tests/test_copies.py::test_vehikel_b_tabelle_ist_unveraenderte_kopie_der_gemeinsamen_tabelle`, `tests/test_copies.py::test_konstanten_und_medianformel_gleich_verweilzeit_demo` | SHA-256 / exakt |
| Preset-Aussagen (Hilfetexte) und Verdikte | `tests/test_verdicts.py::test_preset_aussagen_der_hilfetexte_gegen_die_tabelle`, `tests/test_verdicts.py::test_jedes_preset_hat_eine_erwartete_verdikt_zeile` | exakt / Band |
| Rund 9 bis 20 ms je Lauf, 100 Läufe bei m = 500 unter 20 s | `tests/test_tabellen_und_baender.py::test_zeit_je_lauf_fuer_die_obergrenzen_der_regler` | grobe Obergrenze, kein Befund |
| Alle übrigen Zahlen der Tabelle „Befunde und Korrekturen“ (Werte der Vormessung) | `tests/test_vormessung_baender.py::test_tabelle_a_ml_und_l_momente_im_band_der_vormessung_v11` | Zitat der Vormessung, im Test als Band geprüft |

Messung dieser README, **nicht im Test**: die Laufzeiten in Sekunden, die Zahl der Tests und die Dauer der Suite, der Anteil unmöglicher L-Moment-Anpassungen bei m = 500 (rund 42 % für ξ = −0,9; 300 Läufe der Korrektur) sowie die Zahlen „zweiter Lauf“ (Läufe der unabhängigen Aussagenprüfung mit anderem Zufalls-Seed; sie zeigen die Streuung der Tabellenwerte und sind nicht nachgerechnet).

## Ehrliche Grenzen

### Wo die Annahmen enden

| Annahme | Was passiert, wenn sie verletzt ist | Hier gemessen? |
|---|---|---|
| **Unabhängige, gleich verteilte Beobachtungen** | Bei Abhängigkeit (Cluster) sind die Block-Maxima nicht mehr unabhängig. | Nein |
| **Stationär, ohne Trend** | Ein Trend verschiebt die Blöcke gegeneinander. | Nein |
| **Ausgedachte Quellen mit exakter Verteilung** | Es gibt keine Messreihe aus Logistik, Lager oder Hafen; die Zahlen sagen nichts über echten Betrieb. | – |
| **Alle Blöcke gleich lang** (b teilt N = 18 000) | Gleitende Blöcke und Blockgrößen unter 6 sind nicht gemessen. | Nein |
| **Nebenbedingung ξ ≥ −1 bei ML** | Eine Wahl dieser Demo: für ξ < −1 ist die Likelihood unbeschränkt. Andere Software kann bei kleinem m andere Ränder haben. | Ja (Experiment 3) |
| **Klammer ξ̂ ∈ [−0,999; 0,99] bei L-Momenten** | Eine Wahl dieser Demo; ohne Klammer gibt es mehr oder gleich viele unmögliche Anpassungen (die Klammer greift nur bei ξ nahe −1 oder kleinem m). | Ja (Schalter, Experiment 3) |
| **Nur Punktschätzung** | Keine Intervalle, keine Überdeckung, kein Streuband gegen Wiederkehrperiode und m. | Nein |

### Weitere Grenzen

* **Nur Punktschätzung.** Keine Intervalle, keine Überdeckung, kein Streuband gegen Wiederkehrperiode und m: Die Streuung von V ist die Streuung über Läufe bei dieser Blockgröße.
* **Die SD von ξ̂ bei m = 10** und die SD des Pegel-Fehlers der ML-GEV bei m = 20 sind Ausreißer-Maße und hängen an der fehlenden oberen Grenze des Optimierers; andere Software (R `ismev`, `evd`) kann bei kleinem m andere Ränder und andere Anteile haben.
* **ML und L-Momente sind Wahlen dieser Demo** (Nebenbedingung ξ ≥ −1, Klammer ξ̂ ∈ [−0,999; 0,99], eigener Nelder-Mead). Gumbel mit festem ξ = 0 gewinnt nur, wenn ξ = 0 vorab richtig bekannt ist; bei wahrem ξ = 0,2 liegt es (m = 50, 100er-Wert mit p = 0,01 je Block) bei −2,36 σ, in 99 % der Läufe zu niedrig.
* **Aussagen über eine Methode gelten für die gemessene Regel**, nicht für die Methodenklasse: „L-Momente“ heißt hier die exakt zurückgerechnete Anpassung mit der genannten Klammer, „Kleinste Quadrate“ die ungewichtete Anpassung auf Positionen nach Weibull.
* **Erklärungen.** Dass sich für die Normalverteilung kein Grenzwert des Bias ablesen lässt, ist als Abhängigkeit vom Abschneiden gemessen (ein Grenzwert bei sehr großem Abschneiden ist nicht ausgeschlossen); die Erklärung (Tail-Masse über dem Endpunkt) ist eine Deutung, kein Beweis.
* **Die Streamlit-Laufzeit** der Cloud ist nicht gemessen; die Obergrenzen der Regler (Läufe bis 100, m bis 500) sind nach den Zeiten dieses Rechners gesetzt.

## 🚧 Bewusst nicht enthalten / nicht gemessen

* **Gleitende Blöcke** (Blockmaxima-ML mit Bias zweiter Ordnung: Dombry und Ferreira, arXiv 1705.00465; gleitende statt disjunkter Blöcke: Bücher und Segers (2018), Electronic Journal of Statistics 12 (1), arXiv 1706.01968; von beiden nur das Abstract gelesen, nicht nachgerechnet, die Zahl „über 18 % kleinere Varianz“ wird hier nicht benutzt).
* **Blockgrößen unter b = 6** und über 360; **Blockgröße bei Abhängigkeit**.
* **Intervalle** (Delta, Profil, Bootstrap) und Überdeckungstabellen.
* **Newton-Verfahren** für ML (der Nelder-Mead reicht für die Laufzeiten, siehe oben).
* **Eine echte Datenquelle**: es gibt keine.
* **Quellen im Original:** Smith 1985, Hosking/Wallis/Wood 1985, Hosking 1990, Cunnane 1978 und Fisher/Tippett 1928 sind nicht gelesen; Befunde daraus stehen „nach Zusammenfassung, ungeprüft“ und es gibt keinen Test daraus.

## Quellen

Stufen: **W** = Wortlaut gelesen, **A** = Abstract oder Katalogzeile, **S** = nur Suchtreffer oder Zusammenfassung, **R** = eigene Rechnung. Aus geschützten Werken steht hier nichts wörtlich.

* **Coles (2001)**, *An Introduction to Statistical Modeling of Extreme Values*, Springer, Kap. 3 (GEV, Pegel, Blockgröße, Regularität nach Smith, S. 55): **W**, gelesen in einer Kopie mit unklaren Weitergaberechten, deshalb nur Gleichungen und Befunde mit Quelle.
* **Embrechts, Klüppelberg, Mikosch (1997)**, *Modelling Extremal Events for Insurance and Finance*, Springer, 6.3.1 und 6.3.2 (ML numerisch, PWM): **W**, gleiche Einschränkung.
* **Smith (1985)**, *Maximum likelihood estimation in a class of nonregular cases*, Biometrika 72 (1), 67–90: Volltext nicht gelesen, Befund über ξ nach Coles S. 55 (**W** über Coles), sonst **A**.
* **Hosking, Wallis, Wood (1985)**, Technometrics 27 (3), 251–261, und **Hosking (1990)**, J. R. Stat. Soc. B 52 (1), 105–124: nicht gelesen (**A**/**S**); die Rückrechnung von ξ̂ aus τ₃ ist hier exakt (Bisektion), die Näherung nur als Startwert.
* **Cunnane (1978)**, *Unbiased plotting positions – a review*, J. Hydrology 37, 205–222, und **Fisher/Tippett (1928)**, Proc. Cambridge Philos. Soc. 24, 180–190: nicht gelesen (**S**/**A**).
* **Dombry/Ferreira**, arXiv 1705.00465, und **Bücher/Segers (2018)**, Electronic Journal of Statistics 12 (1), arXiv 1706.01968: nur Abstract (**A**), nur für den Hinweis auf gleitende Blöcke oben; keine Zahl und kein Test daraus.
* **Eigene Rechnung (R):** die Näherung von Hosking für ξ̂ aus τ₃ hat im Bereich ξ ∈ [−0,5; 0,5] einen Fehler unter 1e-3 (Test), die Angabe der Originalarbeit ist nicht gelesen.

## Tests

Die Suite hat **176 Tests** (`pytest --collect-only`). Laufzeit der ganzen Suite seriell rund 100 s (zwei Läufe beim Bau: 92 s und 108 s; bei der Korrektur 239 s auf einem durch andere Arbeit stark ausgelasteten Rechner) (Windows, Python 3.14.6, numpy 2.5.3, scipy 1.18.1, streamlit 1.65.0; eine Messung dieser README, nicht im Test; die Demo-CI auf GitHub ist nicht gemessen). Zusammensetzung:

* **Handbeispiele** `tests/test_handbeispiele.py` (12 Tests): H1 bis H7 des Plans als exakte Tests (Brüche, Ganzzahlen, Tafelwerte), nur Standardbibliothek und numpy.
* **Orakel** mit anderem Rechenweg (scipy nur hier): `tests/test_oracle_gev.py` (16 Tests: Likelihood, Quantilfunktion und Pegel gegen `genextreme` mit c = −ξ und Bisektion, eigene ML gegen `genextreme.fit`, Gumbel-ML gegen `gumbel_r.fit`, L-Momente gegen Quadratur und gegen Paar- und Tripel-Aufzählung), `tests/test_oracle_population.py` (12 Tests: Quadratur gegen Simulation, Fréchet, Normal ohne ablesbaren Grenzwert, einzelne große Normalstichproben über Zufalls-Seeds, ξ_pop), `tests/test_oracle_mixture.py` (6 Tests: eigene Aufzählung von Vehikel B gegen die kopierte Tabelle, Überlebensfunktion gegen scipy, Pegel, Umkehrtabelle).
* **Literatur** `tests/test_literatur_gev.py` (2 Tests): Coles (2001) Gl. 3.4 und die Regularität nach Smith; nichts aus nicht gelesenen Quellen.
* **Eigenschaften und Randfälle** `tests/test_eigenschaften_und_randfaelle.py` (31 Tests): Verschiebung und Skalierung, Reihenfolge, Monotonie, Endpunkt bei ML, lnRMSE² = Bias² + SD², gleicher Zufalls-Seed gleiche Ausgabe; m = 0 bis 3, alle Maxima gleich, ξ = −1 und ξ < −1, ξ → 0, b = N, b teilt N nicht, Pareto-Verteilung mit α ≤ 1, p = 0 und 1, Läufe = 1.
* **Zufall und Reihen** `tests/test_rng_und_szenario.py` (9 Tests), **Bänder gegen die Vormessung** `tests/test_vormessung_baender.py` (4 Tests) und **Neu-Rechnen als Band** `tests/test_tabellen_und_baender.py` (7 Tests).
* **Kopien** `tests/test_copies.py` (4 Tests), **Claims** `tests/test_claims.py` (6 Tests), **Verdikte** `tests/test_verdicts.py` (18 Tests) und **Presets** `tests/test_presets.py` (10 Tests).
* **Oberfläche:** `tests/test_app.py` (3 Tests, jeder Knopf und jeder Zahlenregler an den Grenzen), `tests/test_app_regler.py` (12 Tests, jede Option jedes Auswahlreglers, jedes Preset, Permalink), `tests/test_layout.py` (21 Tests), `tests/test_footer.py` (2 Tests), `tests/test_runtime_imports.py` (1 Test).

Zufallsergebnisse werden nie auf das letzte Bit verglichen (Linux-CI, `log` und `pow` aus der C-Bibliothek), sondern als Band mit Standardfehler; Brüche und Ganzzahlen sind exakt. Eine Prüfung unter Linux mit Docker (`demo_linux_check.py`, Python 3.12) lief vor dieser Korrektur durch: 172 bestanden, 2 übersprungen (die zwei Tests, die Dateien außerhalb der Demo lesen), 63 s; nach dieser Korrektur ist sie nicht wiederholt.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Kopf, Schnellstart, fünf Experimente, Grenzen, Formeln |
| `gvb_gev.py` | GEV-Kern: Verteilung, Pegel, negative Log-Likelihood, L-Momente und exakte Rückrechnung |
| `gvb_fit.py` | ML (eigener Nelder-Mead), L-Momente, Gumbel; nur numpy |
| `gvb_sources.py` | Quellen mit exakter Verteilung, Vehikel B aus der kopierten Tabelle, wahre Pegel |
| `gvb_scenario.py` | Reihen und Block-Maxima aus dem SplitMix64, Zufalls-Seed-Mischung |
| `gvb_population.py` | Populations-Bias per Quadratur (nur Grenzwert für ξ_pop ≥ 0), Fréchet-Kontrolle, große Normal-Stichprobe |
| `gvb_evaluation.py` | Einstellungen, Läufe, Kennzahlen, Verdikte als reine Funktionen |
| `gvb_reference.py` | Lädt die eingefrorenen Messungen |
| `gvb_rng.py` | Kopie der Vorlage `rng_splitmix64_exact.py` (unverändert, Kopier-Test) |
| `gvb_mixture.json`, `gvb_mixture_quelle.json`, `gvb_mixture_inv.json` | Tabelle von Vehikel B (unveränderte Kopie, Quelle und Stand je Feld, eingefrorene Umkehrtabelle) |
| `gvb_tables.json` | eingefrorene Messungen (Tabellen A, B, C) mit Quelle und Stand |
| `gvb_constants.py`, `gvb_presets.py`, `gvb_layout.py`, `gvb_texts.py`, `gvb_visualization.py` | Regler-Grenzen und Presets, Permalink und Schnellstart, fester Kopf, Texte, Plotly-Abbildungen |
| `tools/gvb_make_tables.py` | erzeugt und prüft die eingefrorenen Daten (`--check`, `--lang`) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, pandas, plotly) und Test (plus pytest, scipy nur als Gegenprobe) |
| `tests/` | Handbeispiele, Orakel (scipy nur hier), Eigenschaften, Randfälle, Bänder, Claims, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

* **scipy, scikit-learn, statsmodels in der App.** Die App rechnet mit numpy und der Standardbibliothek; scipy steht nur in `requirements-dev.txt` als Gegenprobe (`genextreme` mit c = −ξ).
* **Rückverweise und Imports aus anderen Demos.** Die Vehikel-B-Tabelle ist eine Kopie; nur der Kopier-Test liest `verweilzeit-demo`, falls vorhanden.
* **PDF-Export** und Intervalle: nicht Teil dieser Demo.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py --server.port 10084
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Eingefrorene Daten prüfen (rechnet einen Ausschnitt neu): `python tools/gvb_make_tables.py fit --check` (ebenso `block`, `positions`, `mixture-inv`).

Gebaut mit Streamlit, Plotly, NumPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html).
