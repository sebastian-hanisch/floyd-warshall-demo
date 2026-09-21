"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 100.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.25                      # Lageabweichung der Kreuzungen in Blocklängen

NETS = ("small", "small_cycle", "city", "ev", "random")
NET_LABELS = {
    "small": "🔀 Kleines Netz (eine negative Kante)",
    "small_cycle": "🔁 Kleines Netz mit negativem Zyklus",
    "city": "🏙️ Stadtnetz (erzeugt)",
    "ev": "🔋 E-Lieferwagen mit Rekuperation (erzeugt)",
    "random": "🕸️ Zufallsnetz (erzeugt)",
}
SMALL_NETS = ("small", "small_cycle")                      # eigene kleine Graphen mit Namen: Matrix mit Zahlen, feste Aufgabe
SIZED_NETS = ("city", "ev")                                # Netze mit Größenregler (Raster)

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 4, 20, 8                # n = Seite², höchstens 400 Knoten (die Matrix hat n² Zellen)
HILL_MIN, HILL_MAX, DEFAULT_HILL = 0, 40, 30
ETA_MIN, ETA_MAX, DEFAULT_ETA = 0, 90, 60
REACH_MIN, REACH_MAX, DEFAULT_REACH = 1.0, 3.2, 1.5
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0.0, 3.0, 1.0
NODES_MIN, NODES_MAX, DEFAULT_NODES = 20, 400, 100
DEGREE_MIN, DEGREE_MAX, DEFAULT_DEGREE = 1.5, 20.0, 3.0    # mittlerer Ausgangsgrad: dünn bis dicht
POT_MIN, POT_MAX, DEFAULT_POT = 0, 20, 0                   # Potenzialspanne (Zufallsnetz): 0 = keine negativen Kanten
DISTANCE_MIN, DISTANCE_MAX, DEFAULT_DISTANCE = 10, 100, 60     # Prozent: Rang der Entfernung des Ziels vom Start (Routen-Ansicht)
DEFAULT_SEED = 7
DEFAULT_NET = "small"

DEFAULT_VARIANT = "numpy"
VARIANT_LABELS = {"numpy": "Matrixoperation je k (schnell, alle Größen)", "classic": "dreifache Schleife in Python (n³ Vergleiche)", "skip": "Schleife, überspringt unerreichbare Paare"}
PYTHON_VARIANT_MAX_NODES = 120                             # darüber ist die reine Python-Schleife zu langsam (n³ > 1.7 Mio. Schritte)

SWEEP_SEEDS = tuple(range(100000, 100005))

COLORS = {"fw": "#d62728", "dijkstra": "#1f77b4", "negative": "#2ca02c", "cycle": "#d62728", "start": "#111111", "goal": "#ff7f0e", "changed": "#ffd54f"}

_BASE = dict(side=DEFAULT_SIDE, hill=DEFAULT_HILL, eta=DEFAULT_ETA, reach=DEFAULT_REACH, spread=DEFAULT_SPREAD, nodes=DEFAULT_NODES, degree=DEFAULT_DEGREE, pot=DEFAULT_POT,
             variant=DEFAULT_VARIANT, distance=DEFAULT_DISTANCE, seed=DEFAULT_SEED)
PRESETS = {
    "🔀 Kleines Netz": {**_BASE, "net": "small"},
    "🔁 Negativer Zyklus": {**_BASE, "net": "small_cycle"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city"},
    "🔋 E-Lieferwagen": {**_BASE, "net": "ev", "distance": 20},
    "🕸️ Zufallsnetz": {**_BASE, "net": "random", "degree": 6.0},
}
PRESET_HELP = {
    "🔀 Kleines Netz": "Kleines eigenes Liefernetz mit einer Rückvergütung (Ost → Nord, −3 Euro): Floyd-Warshall braucht 216 Vergleiche (6³), 26 davon verbessern einen Wert. Für Depot → Hafen findet es 6 Euro, n-mal Dijkstra nur 8; n-mal Dijkstra liegt an 7 von 30 Paaren falsch.",
    "🔁 Negativer Zyklus": "Dasselbe Netz, aber mit einer Rückvergütung auf Süd → West: der Hin- und Rückweg zwischen Süd und West kostet −2 Euro. Zwei Werte der Diagonale werden negativ (Süd und West), für 13 von 30 Paaren gibt es keine kürzeste Route, und nur 16 von 30 Paaren sind überhaupt erreichbar.",
    "🏙️ Stadtnetz": "Erzeugtes Stadtnetz (8 × 8 = 64 Kreuzungen, ohne negative Kanten): Floyd-Warshall braucht 262 144 Vergleiche (64³), n-mal Dijkstra nur 21 504 Kantenprüfungen und n-mal Bellman-Ford 142 464 - in einem so dünnen Netz lohnt sich n³ nach Zählern nicht. Die längste kürzeste Route hat 9 Kanten.",
    "🔋 E-Lieferwagen": "Hügeliges Stadtnetz (8 × 8, 30 m Hügel, 60 % Rückgewinnung): 42 von 336 Kanten sind negativ, n-mal Dijkstra liegt an 524 von 4 032 Paaren falsch. Für das gezeigte Paar (Abstand 20 %) findet Floyd-Warshall 12 Wh, n-mal Dijkstra 15.",
    "🕸️ Zufallsnetz": "Zufallsnetz (100 Knoten, mittlerer Grad 6, ohne negative Kanten): Floyd-Warshall braucht 1 000 000 Vergleiche, n-mal Dijkstra 60 000 Kantenprüfungen (16.7-fach weniger) und n-mal Bellman-Ford 319 800. \"Überspringen\" spart 41 % der Vergleiche (591 649 bleiben).",
}
