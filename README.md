# Floyd-Warshall – alle Paare mit drei Schleifen – Streamlit-Demo

Sechstes Stück der **Kürzeste-Wege-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Fortsetzung der [Bellman-Ford-Demo](../bellman-ford-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Floyd-Warshall** – an einem wachsenden Beispiel.
Bellman-Ford beantwortet die Entfernungen von **einem** Start. Floyd-Warshall lässt die Knoten nacheinander als **Zwischenknoten** zu: nach Schritt *k* ist jede Entfernung die beste Route, die nur über die Knoten 0 bis *k − 1* läuft.
Die Formel ist eine Zeile, der Code drei verschachtelte Schleifen; das Verfahren verträgt negative Kanten, erkennt negative Zyklen an der Diagonale und packt jede Route aus einer Nachfolger-Matrix aus.
Der Preis: ***n*³** Vergleiche und *n*² Speicher – unabhängig davon, wie dünn das Netz ist.

**Einordnung in die Reihe (die Kanten des Graphen):** Floyd-Warshall ist der zweite Vorläufer von **Johnson** (Konvergenz: Bellman-Ford + Dijkstra); die Demo misst, warum man dort nicht bei *n*³ bleibt.
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)                                       [gebaut]
  └─ dijkstra-demo (Kosten korrekt, blind in alle Richtungen)                        [gebaut]
       ├─ bidirectional-demo → contraction-hierarchies-demo                          [gebaut]
       ├─ bellman-ford-demo ─┐                                                       [gebaut]
       │   floyd-warshall-demo ─┴→ Johnson (Konvergenz: Umgewichtung mit Potenzialen)  [dieses Stück → Johnson nicht gebaut]
       └─ Mehrkriterien-Routing (Zeit gegen CO₂, Pareto)                             [nicht gebaut]
```

## Quellen

| Bestandteil | Quelle |
|---|---|
| Verfahren (Rekursion über Zwischenknoten, Nachfolger-Matrix, Diagonale) | Floyd (1962), Warshall (1962), Roy (1959); die Bücher (*Grokking Algorithms*, *Optimization Algorithms*) behandeln Floyd-Warshall nicht, ein Buchbeispiel gibt es nicht zu spiegeln |
| Drei Varianten (Matrixoperation, Python-Schleife, "überspringt"), Schleifenreihenfolgen, Zwischenknoten-Reihenfolgen, n-mal Dijkstra / Bellman-Ford als Zähler | eigene Umsetzung (Dijkstra und Bellman-Ford aus der Bellman-Ford-Demo) |
| Alle Netze | **eigene Graphen und Erzeuger**: kleines Liefernetz (mit einer negativen Kante, und die Variante mit negativem Zyklus), Stadtnetz, E-Lieferwagen mit Rekuperation im hügeligen Stadtnetz, Zufallsnetz mit einstellbarer Dichte |
| Zahlen | **eigene Messungen** an diesen Netzen |

Aus den Büchern stammt keine Zahl, kein Graph und kein Text. **Kein OpenStreetMap-Auszug**, also keine ODbL-Pflichten.

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Kleines Netz (6 Orte, eine Rückvergütung) | ✅ **216** Vergleiche (6³), 26 davon verbessern einen Wert; für Depot → Hafen **6 Euro** gegen **8** bei n-mal Dijkstra, das an **7 von 30** Paaren falsch liegt |
| Negativer Zyklus (Süd ↔ West: 1 − 3 = −2 Euro) | 🔁 die Diagonale wird bei **Süd und West** negativ; für **13 von 30** Paaren gibt es keine kürzeste Route, nur **16 von 30** sind erreichbar |
| E-Lieferwagen (8 × 8, 30 m Hügel, 60 % Rückgewinnung) | ✅ **42** von 336 Kanten negativ; n-mal Dijkstra liegt an **524 von 4 032** Paaren falsch (13 %), Floyd-Warshall und n-mal Bellman-Ford stimmen überein |
| Aufwand im Stadtnetz (64 Knoten) | ⚠️ **262 144** Vergleiche gegen **21 504** Kantenprüfungen (n-mal Dijkstra) und **142 464** (n-mal Bellman-Ford) |
| Aufwand gegen Dichte und Größe (Zufallsnetze ohne negative Kanten) | ⚠️ **nach Zählern gewinnt Floyd-Warshall nirgends**: bei 200 Knoten und Grad 2 das **100-Fache** von n-mal Dijkstra (11-Fache von n-mal Bellman-Ford), bei 400 Knoten und Grad 3 das **133-Fache** bzw. gut das **16-Fache**; erst im **vollständigen Netz** (60 Knoten) gleichen sich die Zahlen: 216 000 Vergleiche gegen 212 400 Kantenprüfungen |
| Variante "überspringt" | ⚠️ spart in dünnen Netzen viel (Anteil an *n*³ bei mittlerem Grad 2 / 3 / 6 / 12: **20 / 34 / 60 / 80 %**), im dichten Netz fast nichts; das Ergebnis bleibt dasselbe |
| Reihenfolge der drei Schleifen | ❌ nur mit ***k* außen** (kij, kji) ist die Matrix richtig; die anderen vier Reihenfolgen liegen bei **9 bis 14 %** der Zellen daneben (kleine Zufallsnetze mit 10 Knoten) – bei gleichem Aufwand |
| Negative Zyklen im Zufallsnetz (Kosten −3 bis 8, Grad 4) | ❌ die Werte **wachsen exponentiell**: kleinster Wert bei 10 / 80 / 400 Knoten etwa **−28 / −1.6 · 10¹⁰ / −3.6 · 10⁵⁰**; rund 96 % der Paare haben keine kürzeste Route |
| Reihenfolge der Zwischenknoten (nur für "überspringt") | ⚠️ "wenige Nachbarn zuerst" (wie beim Zusammenziehen in Contraction Hierarchies) senkt die Vergleiche im **Zufallsnetz von 34 % auf 28 %**, im Stadtnetz bringt es nichts (41 % gegen 42 %); "viele Nachbarn zuerst" kostet in beiden deutlich mehr (44 % bzw. 60 %) |
| Korrektheit | ✅ jede Variante liefert auf jedem geprüften Netz exakt die Matrix von networkx (`floyd_warshall_numpy`), dieselbe Nachfolger-Matrix, und meldet genau dann einen negativen Zyklus, wenn networkx einen findet; jede ausgepackte Route existiert im Netz und hat genau die berichteten Kosten; nach Schritt *k* benutzen alle Routen nur Zwischenknoten unter *k* |

Die Zähler sind Schritte des Verfahrens und plattformfest, aber **kein gemeinsames Maß** (ein Vergleich in einer Matrixoperation ist billiger als eine Kantenprüfung mit Warteschlange). Laufzeiten stehen in der App nur als Messwerte und werden nirgends behauptet oder getestet:
Floyd-Warshall läuft als vektorisierte Matrixoperation (in C), n-mal Dijkstra und n-mal Bellman-Ford sind reines Python – die Laufzeiten sind deshalb nicht als Vergleich der Verfahren zu lesen. Was Floyd-Warshall bietet, ist der einfache Code, die Nachfolger-Matrix und Matrixoperationen.

## Was die Demo zeigt

1. **Floyd-Warshall in Aktion** (Schritt-Regler + Abspielen): die Entfernungsmatrix als **Heatmap** (bei kleinen Netzen mit den Werten in den Zellen), der gerade zugelassene Zwischenknoten als helles Band, die in diesem Schritt verbesserten Zellen orange umrandet, negative Diagonale rot, und – im letzten Schritt – die Zellen, in denen n-mal Dijkstra danebenliegt, mit weißem Kreuz. Daneben das Netz mit den bisher erlaubten Zwischenknoten und der besten Route des gewählten Paares.
2. **Alle Paare auf einmal – und was das kostet:** Vergleiche (auch die der Variante "überspringt"), n-mal Dijkstra, n-mal Bellman-Ford, Matrixgröße; Urteil (✅ / ℹ️ / 🔁); Erreichbarkeit und längste kürzeste Route.
3. **Vergleich** (Expander) mit Laufzeiten als Messwerte; **Experimente auf Knopfdruck**: Reihenfolge der drei Schleifen, Aufwand gegen Dichte und Größe, Wachstum der Werte bei negativen Zyklen, Reihenfolge der Zwischenknoten.
4. **Wo die Annahmen enden** (Tabelle mit den Ansatzpunkten der nächsten Stücke) und **Mathematische Formulierung** (Rekursion, Korrektheit, warum *k* außen steht, Nachfolger, negative Zyklen, Aufwand, Bezug zu Johnson und zur transitiven Hülle).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz, Variante und – bei erzeugten Netzen – das gezeigte Paar wählen; die Adresszeile spiegelt die Konfiguration (Permalink). Regler, die zum gewählten Netz nicht gehören, sind ausgeblendet. Die Python-Schleifen gibt es nur bis 120 Knoten; höchstens 400 Knoten, weil die Matrix *n*² Zellen hat.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `fw_graph.py`, `fw_queues.py`, `fw_sp.py` | Graph in CSR-Form, Warteschlange, Dijkstra und Bellman-Ford (aus der Bellman-Ford-Demo) für den n-mal-Vergleich |
| `fw_algorithm.py` | Floyd-Warshall (drei Varianten, Nachfolger-Matrix, Zwischenstände), Schleifenreihenfolgen, betroffene Paare bei negativen Zyklen, Route und Kantenzahl je Paar |
| `fw_scenario.py` | Netze: kleines Netz (mit und ohne negativen Zyklus), Stadtnetz, E-Lieferwagen, Zufallsnetz |
| `fw_evaluation.py` | Kennzahlen, Paarwahl, Experimente |
| `fw_visualization.py`, `fw_presets.py`, `fw_constants.py` | Abbildungen, Presets und Permalink, Konstanten |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe läuft gegen networkx (`floyd_warshall_numpy`, `negative_edge_cycle`) für alle Varianten auf Netzen mit und ohne negative Kanten und Zyklen, mit unerreichbaren Paaren, Nullkanten und einem einzelnen Knoten.
