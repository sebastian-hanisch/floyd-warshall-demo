"""Floyd-Warshall - alle Paare mit drei Schleifen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Floyd-Warshall - und lässt stattdessen das Beispiel wachsen.
Sechstes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe, zweiter Vorläufer von Johnson: Bellman-Ford beantwortet einen Start, Floyd-Warshall alle Paare auf einmal.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import fw_constants as C
import fw_evaluation as ev
from fw_algorithm import route_from
from fw_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from fw_scenario import make_network
from fw_visualization import (
    build_effort_degree,
    build_effort_size,
    build_growth,
    build_intermediate_order,
    build_loop_order,
    build_matrix,
    build_network,
    state_at,
    table_rows,
)

st.set_page_config(page_title="Floyd-Warshall – Sebastian Hanisch", layout="wide")


def _num(x):
    return f"{x:,.0f}".replace(",", ".")


def _cost(net, x):
    return f"{x:,.0f} {net.unit}".replace(",", ".") if float(x).is_integer() else f"{x:g} {net.unit}"


def _pct(x, digits=0):
    return "–" if x is None or np.isnan(x) else f"{x:.{digits}%}"


@st.cache_resource(show_spinner=False, max_entries=8)
def _network(params):
    return make_network(*params)


@st.cache_resource(show_spinner=False, max_entries=8)
def _analysis(params, variant, distance):
    return ev.analyse(_network(params), variant, distance, params[-1])


@st.cache_data(show_spinner=False)
def _loop_orders():
    return ev.loop_order_table()


@st.cache_data(show_spinner=False)
def _effort_degree():
    return ev.effort_vs_degree()


@st.cache_data(show_spinner=False)
def _effort_size():
    return ev.effort_vs_size()


@st.cache_data(show_spinner=False)
def _complete():
    return ev.complete_graph_effort()


@st.cache_data(show_spinner=False)
def _growth():
    return ev.cycle_growth()


@st.cache_data(show_spinner=False)
def _intermediate():
    return ev.intermediate_order_table()


st.title("🧮 Floyd-Warshall – alle Paare mit drei Schleifen")
st.markdown(
    """
Bellman-Ford beantwortet die Entfernungen von **einem** Start. Wer sie für **alle Paare** braucht - eine Distanzmatrix für die Tourenplanung, zum Beispiel -, könnte n-mal Dijkstra oder n-mal Bellman-Ford rechnen.
**Floyd-Warshall** geht anders vor: es lässt die Knoten nacheinander als **Zwischenknoten** zu. Nach Schritt $k$ ist jede Entfernung die beste Route, die nur über die Knoten $0,\\dots,k-1$ als Zwischenstationen läuft. Die Formel dahinter ist eine Zeile, der Code besteht aus drei verschachtelten Schleifen.
Das Verfahren verträgt negative Kanten, erkennt negative Zyklen an der Diagonale und packt jede Route aus einer Nachfolger-Matrix aus. Der Preis: **$n^3$** Schritte und $n^2$ Speicher - unabhängig davon, wie dünn das Netz ist.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - sechstes Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe, Fortsetzung der Bellman-Ford-Demo - **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Schwächen von Floyd-Warshall sind der Aufwand $n^3$ (auch bei dünnen Netzen) und der Speicher $n^2$. Das Verfahren geht auf Floyd (1962) und Warshall (1962) zurück; "
    "alle Netze und Zahlen dieser Demo sind eigene Graphen und Messungen."
)

with st.expander("So funktioniert Floyd-Warshall", expanded=True):
    st.markdown(
        """
1. **Start:** die Matrix $D$ enthält 0 auf der Diagonale, die Kantenkosten für vorhandene Kanten und $\\infty$ sonst.
2. **Zwischenknoten zulassen:** für $k = 0, 1, \\dots, n-1$: für jedes Paar $(i,j)$ prüfen, ob der Umweg über $k$ billiger ist: $D_{ij} \\leftarrow \\min(D_{ij},\\; D_{ik} + D_{kj})$. Nach dem Durchgang für $k$ sind alle Routen berücksichtigt, die nur über die Knoten $0,\\dots,k$ führen.
3. **Nachfolger merken:** wird $D_{ij}$ verbessert, merkt man sich, dass der erste Schritt von $i$ nach $j$ jetzt derselbe ist wie von $i$ nach $k$. So lässt sich jede Route später in ihrer Länge auspacken.
4. **Reihenfolge der Schleifen:** $k$ muss die **äußerste** Schleife sein - sonst fehlen Zwischenknoten, wenn sie gebraucht werden (das zeigt ein Experiment unten).
5. **Negativer Zyklus:** wird ein Wert der Diagonale negativ, liegt der Knoten auf einem negativen Zyklus oder auf einem Weg, der zu einem solchen hin- und wieder zurückführt. Umgekehrt wird jeder Knoten auf einem negativen Zyklus negativ, Knoten in dessen Umgebung können aber bei 0 oder darüber bleiben. Für Paare, deren Routen über so einen Knoten führen können, gibt es keine kürzeste Route.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

# Knotenzahl des gewählten Netzes: die Python-Schleifen (n³ Schritte) gibt es nur für kleine Netze
_net_now = st.session_state["net_select"]
_n_now = {"small": 6, "small_cycle": 6}.get(_net_now, int(st.session_state.get("nodes_slider", st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))) if _net_now == "random" else int(st.session_state.get("side_slider", st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))) ** 2)
variant_options = [v for v in C.VARIANT_LABELS if v == "numpy" or _n_now <= C.PYTHON_VARIANT_MAX_NODES]
if st.session_state["variant_select"] not in variant_options:
    st.session_state["variant_select"] = C.DEFAULT_VARIANT

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Klein und fest (ein Liefernetz mit einer negativen Kante, dasselbe mit negativem Zyklus) oder erzeugt (Stadtnetz, E-Lieferwagen mit Rekuperation, Zufallsnetz). Alle Kosten sind ganze Zahlen; "
             "höchstens 400 Knoten, weil die Matrix n² Zellen hat.",
    )
    if net_key in C.SIZED_NETS:
        seed_widget("side_slider")
        side = st.slider("Kreuzungen je Seite", *bounds("side_slider"), key="side_slider",
                         help="Größe des Rasters: n = Seite² Knoten. Floyd-Warshall braucht n³ Vergleiche: bei 8 / 14 / 20 Kreuzungen je Seite 262 144 / 7 529 536 / 64 000 000.")
        st.session_state[KEPT["side_slider"]] = side
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
    if net_key == "ev":
        seed_widget("hill_slider")
        hill = st.slider("Hügel [m Höhenunterschied]", *bounds("hill_slider"), key="hill_slider", help="Höhenunterschied zwischen Tal und Spitze: je höher, desto mehr negative Kanten (bei 0 m gibt es keine).")
        st.session_state[KEPT["hill_slider"]] = hill
        seed_widget("eta_slider")
        eta = st.slider("Rückgewinnung bergab [%]", *bounds("eta_slider"), key="eta_slider", help="Wie viel Lageenergie der Wagen beim Bergabfahren zurückgewinnt. Unter 100 % entsteht nie ein negativer Zyklus.")
        st.session_state[KEPT["eta_slider"]] = eta
    else:
        hill = int(st.session_state.get(KEPT["hill_slider"], C.DEFAULT_HILL))
        eta = int(st.session_state.get(KEPT["eta_slider"], C.DEFAULT_ETA))
    if net_key == "city":
        seed_widget("reach_slider")
        reach = st.slider("Reichweite der Straßen [Blocklängen]", *bounds("reach_slider"), key="reach_slider", step=0.1, help="Wie weit eine Straße zwischen zwei Kreuzungen reichen darf (1 = nur Nachbarn im Raster).")
        st.session_state[KEPT["reach_slider"]] = reach
        seed_widget("spread_slider")
        spread = st.slider("Streuung der Kosten", *bounds("spread_slider"), key="spread_slider", step=0.25, help="Kosten einer Straße = Länge × (1 + Streuung × Zufall), gerundet auf ganze Meter.")
        st.session_state[KEPT["spread_slider"]] = spread
    else:
        reach = float(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        spread = float(st.session_state.get(KEPT["spread_slider"], C.DEFAULT_SPREAD))
    if net_key == "random":
        seed_widget("nodes_slider")
        nodes = st.slider("Knoten", *bounds("nodes_slider"), key="nodes_slider", step=10,
                          help="Anzahl der Knoten n. Vergleiche n³ bei 100 / 200 / 400 Knoten: 1 000 000 / 8 000 000 / 64 000 000.")
        st.session_state[KEPT["nodes_slider"]] = nodes
        seed_widget("degree_slider")
        degree = st.slider("Mittlerer Grad", *bounds("degree_slider"), key="degree_slider", step=0.5,
                           help="Kanten je Knoten (ausgehend). Anteil der Vergleiche, die die Variante \"überspringt\" braucht (200 Knoten), bei Grad 2 / 3 / 6 / 12: 20 % / 34 % / 60 % / 80 % von n³ - je dichter, desto weniger Unerreichbares gibt es zu überspringen.")
        st.session_state[KEPT["degree_slider"]] = degree
        seed_widget("pot_slider")
        pot = st.slider("Potenzialspanne", *bounds("pot_slider"), key="pot_slider",
                        help="Kosten = 1 bis 9 plus Potenzial(Start) − Potenzial(Ziel): je größer die Spanne, desto mehr negative Kanten - nie ein negativer Zyklus. 0 = keine negativen Kanten.")
        st.session_state[KEPT["pot_slider"]] = pot
    else:
        nodes = int(st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))
        degree = float(st.session_state.get(KEPT["degree_slider"], C.DEFAULT_DEGREE))
        pot = int(st.session_state.get(KEPT["pot_slider"], C.DEFAULT_POT))
    variant = st.selectbox("Variante", variant_options, key="variant_select", format_func=lambda k: C.VARIANT_LABELS[k],
                           help="Alle Varianten liefern dieselbe Matrix und dieselbe Nachfolger-Matrix. Matrixoperation: je k eine Operation auf Spalte und Zeile (schnell, für alle Größen). Dreifache Schleife: der Lehrbuch-Code in reinem Python, "
                                f"nur bis {C.PYTHON_VARIANT_MAX_NODES} Knoten. \"Überspringt\": lässt Paare aus, deren Routen über k noch nicht existieren - spart Vergleiche, nicht Ergebnis.")
    if net_key not in C.SMALL_NETS:
        seed_widget("distance_slider")
        distance = st.slider("Entfernung Start–Ziel [%]", *bounds("distance_slider"), key="distance_slider",
                             help="Welches Paar die Routen-Ansicht zeigt: das Ziel ist der Knoten, dessen Entfernung vom Start in der Rangfolge aller erreichbaren Knoten bei diesem Prozentwert liegt (100 = der am weitesten entfernte). Die Matrix enthält alle Paare.")
        st.session_state[KEPT["distance_slider"]] = distance
    else:
        distance = int(st.session_state.get(KEPT["distance_slider"], C.DEFAULT_DISTANCE))
    if net_key in ("city", "ev", "random"):
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen.")

sync_query_params({"net_select": net_key, "side_slider": int(side), "hill_slider": int(hill), "eta_slider": int(eta), "reach_slider": round(float(reach), 1), "spread_slider": round(float(spread), 2),
                   "nodes_slider": int(nodes), "degree_slider": round(float(degree), 1), "pot_slider": int(pot), "distance_slider": int(distance), "variant_select": variant, "seed_input": int(seed)})

# nicht zum Netz gehörende Regler ändern das Netz nicht: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
d = dict(side=C.DEFAULT_SIDE, hill=C.DEFAULT_HILL, eta=C.DEFAULT_ETA, reach=C.DEFAULT_REACH, spread=C.DEFAULT_SPREAD, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, pot=C.DEFAULT_POT, seed=C.DEFAULT_SEED)
if net_key == "ev":
    d.update(side=int(side), hill=int(hill), eta=int(eta), seed=int(seed))
elif net_key == "city":
    d.update(side=int(side), reach=round(float(reach), 1), spread=round(float(spread), 2), seed=int(seed))
elif net_key == "random":
    d.update(nodes=int(nodes), degree=round(float(degree), 1), pot=int(pot), seed=int(seed))
params = (net_key, d["side"], d["hill"], d["eta"], d["reach"], d["spread"], d["nodes"], d["degree"], d["pot"], d["seed"])
distance_used = C.DEFAULT_DISTANCE if net_key in C.SMALL_NETS else int(distance)
with st.spinner("Rechne ..."):
    a = _analysis(params, variant, distance_used)
net, m, fw, g = a.net, a.metrics, a.fw, a.net.graph
small = bool(g.names)
n = g.n
view_key = (params, variant, distance_used)
last_step = len(fw.history)
stride = 1 if n <= 100 else max(1, n // 40)

# --- Floyd-Warshall in Aktion --------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Floyd-Warshall in Aktion")
if st.session_state.get("fw_step_owner") != view_key:
    st.session_state["fw_step"] = last_step
    st.session_state["fw_step_owner"] = view_key
step_col, play_col = st.columns([5, 2])
with step_col:
    if last_step > 1:
        step = st.slider("Zwischenknoten k (erlaubt sind die Knoten 0 bis k − 1)" if stride == 1 else f"Zwischenstände (alle {stride} Knoten)", 0, last_step, key="fw_step",
                         help="Wie viele Knoten schon als Zwischenknoten erlaubt sind: 0 = nur die Kanten des Netzes, ganz rechts = alle Knoten erlaubt, die Matrix ist fertig. " + ("" if stride == 1 else "Bei großen Netzen zeigt die Ansicht nur etwa 40 Zwischenstände."))
    else:
        step = last_step
        st.caption("Es gibt nur einen Schritt.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
mark_dj = False
if m["neg_edges"] > 0 and not m["cycle"]:
    mark_dj = st.checkbox("Zellen markieren, in denen n-mal Dijkstra danebenliegt (im letzten Schritt)", value=True, key="mark_dj",
                          help="Weiße Kreuze in den Zellen der fertigen Matrix, in denen n-mal Dijkstra eine andere (falsche) Entfernung liefert als Floyd-Warshall.")
view_slot = st.empty()


def _render(current):
    with view_slot.container():
        k_done = state_at(a, current)[0]
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_matrix(a, current, mark_dj), width="stretch", key=f"matrix_chart_{current}")
        c2.plotly_chart(build_network(a, current, height=360 if small else 520), width="stretch", key=f"net_chart_{current}")
        if k_done == 0:
            st.markdown("**Vor dem ersten Schritt:** die Matrix enthält nur die Kanten des Netzes (und 0 auf der Diagonale).")
        else:
            who = f" (Knoten **{g.names[k_done - 1]}** ist jetzt als Zwischenknoten erlaubt)" if small and stride == 1 else ""
            st.markdown(f"**Nach Schritt {k_done} von {n}:** erlaubte Zwischenknoten sind die Knoten 0 bis {k_done - 1}{who}.")
        if small:
            rows = table_rows(a, current)
            st.markdown((f"**{len(rows)} Verbesserung(en) in diesem Schritt:** " + "; ".join(f"{i} → {j}: {'∞' if not np.isfinite(o) else f'{o:g}'} → {nw:g}" for i, j, o, nw in rows[:14]) + (" ..." if len(rows) > 14 else "")) if rows else "**Keine Verbesserung in diesem Schritt.**")
            kk, dd, nn, _ = state_at(a, current)
            route = route_from(nn, a.s, a.t)
            if route and np.isfinite(dd[a.s, a.t]):
                st.markdown(f"**Beste Route {g.names[a.s]} → {g.names[a.t]} bisher:** {' → '.join(g.names[i] for i in route)} ({_cost(net, dd[a.s, a.t])})")
            elif m["pair_affected"] and current >= last_step:
                st.markdown(f"**Route {g.names[a.s]} → {g.names[a.t]}:** es gibt keine kürzeste Route - der Weg führt über einen negativen Zyklus, die Kosten lassen sich beliebig senken.")
            else:
                st.markdown(f"**Route {g.names[a.s]} → {g.names[a.t]}:** noch keine über die erlaubten Zwischenknoten.")


if auto_play:
    n_frames = min(max(last_step, 1), 40)
    for k in sorted({int(round(x)) for x in np.linspace(0, last_step, n_frames + 1)}):
        _render(k)
        time.sleep(min(0.7, 6.0 / n_frames))
    step = last_step
else:
    _render(step)
st.caption(net.note)

st.markdown("---")

# --- Alle Paare auf einmal ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Alle Paare auf einmal – und was das kostet")
st.caption("**Zähler** = Schritte, die ein Verfahren ausführt: Floyd-Warshall zählt Vergleiche in der Matrix, n-mal Dijkstra und n-mal Bellman-Ford zählen Kantenprüfungen. Die Zähler sind plattformfest, aber kein gemeinsames Maß "
           "(ein Vergleich in einer Matrixoperation ist billiger als eine Kantenprüfung mit Warteschlange). Laufzeiten stehen nur als Messwerte im Vergleich unten.")
code = ev.verdict(a)
p1, p2, p3, p4 = st.columns(4)
p1.metric("Floyd-Warshall", _num(m["comparisons"]), delta=f"überspringt: {_num(m['skip_comparisons'])} ({m['skip_comparisons'] / m['comparisons']:.0%})", delta_color="off", help=f"n³ Vergleiche bei {n} Knoten; {_num(m['improvements'])} davon verbessern einen Wert.")
p2.metric("n-mal Dijkstra", _num(m["dijkstra_checks"]), delta=f"{m['comparisons'] / max(m['dijkstra_checks'], 1):.1f}× weniger als Floyd-Warshall" if m["dijkstra_checks"] < m["comparisons"] else "so viele wie Floyd-Warshall", delta_color="off",
          help="Kantenprüfungen aller n Läufe (nur Kanten erreichbarer Knoten). Richtig nur ohne negative Kanten.")
p3.metric("n-mal Bellman-Ford", _num(m["bf_checks"]), delta=f"{m['comparisons'] / max(m['bf_checks'], 1):.1f}× weniger als Floyd-Warshall" if m["bf_checks"] < m["comparisons"] else "mehr als Floyd-Warshall", delta_color="off",
          help="Kantenprüfungen aller n Läufe mit frühem Abbruch; verträgt negative Kanten, meldet negative Zyklen je Start.")
p4.metric("Matrix", _num(m["cells"]) + " Zellen", delta=f"{m['cells'] * 8 / 1e6:.2f} MB als Zahlen", delta_color="off", help="n × n Entfernungen; dazu eine gleich große Nachfolger-Matrix für die Routen.")
if code == "cycle":
    names = ", ".join(g.names[i] for i in fw.diagonal_negative()) if small else f"{m['diag_negative']} Knoten"
    st.warning(f"🔁 **Negativer Zyklus:** die Diagonale ist bei {names} negativ. Für {m['affected']} von {m['pairs']} Paaren gibt es keine kürzeste Route (jede Route über einen dieser Knoten lässt sich beliebig verbilligen); "
               f"erreichbar sind {m['reachable_share'] * m['pairs']:.0f} von {m['pairs']} Paaren. n-mal Bellman-Ford meldet den Zyklus bei {m['bf_cycle_sources']} von {n} Startknoten, n-mal Dijkstra merkt nichts.")
elif code == "dijkstra_wrong":
    st.success(f"✅ Floyd-Warshall liefert die richtige Matrix (n-mal Bellman-Ford bestätigt sie: {'ja' if m['bf_exact'] else 'nein'}); n-mal Dijkstra liegt an {m['dijkstra_wrong']} von {m['pairs']} Paaren ({m['dijkstra_wrong'] / m['pairs']:.0%}) falsch. "
               f"Der Preis: {_num(m['comparisons'])} Vergleiche gegen {_num(m['dijkstra_checks'])} Kantenprüfungen bei Dijkstra.")
elif code == "all_right_negative":
    st.info(f"ℹ️ Das Netz hat {m['neg_edges']} negative Kanten, n-mal Dijkstra liegt hier aber an keinem Paar falsch: keine Garantie, nur kein Fehler in diesem Netz. Floyd-Warshall braucht {_num(m['comparisons'])} Vergleiche, n-mal Dijkstra {_num(m['dijkstra_checks'])} Kantenprüfungen.")
else:
    st.info(f"ℹ️ Keine negativen Kanten: n-mal Dijkstra liefert dieselbe Matrix mit {_num(m['dijkstra_checks'])} Kantenprüfungen statt {_num(m['comparisons'])} Vergleichen ({m['comparisons'] / max(m['dijkstra_checks'], 1):.1f}-fach weniger), n-mal Bellman-Ford mit {_num(m['bf_checks'])}. "
            "Nach Zählern lohnt sich n³ hier nicht.")
if code != "cycle" and not (a.s == a.t):
    pair_txt = f"**Gezeigtes Paar** ({g.names[a.s] if small else a.s} → {g.names[a.t] if small else a.t}): Floyd-Warshall {_cost(net, m['cost_fw'])}, n-mal Dijkstra {_cost(net, m['cost_dj_pair'])}; die längste kürzeste Route im Netz hat {m['hops_max']} Kanten; "
    pair_txt += f"{m['reachable_share']:.0%} aller Paare sind erreichbar."
    st.markdown(pair_txt)

st.markdown("---")

# --- Vergleich ---------------------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Floyd-Warshall, n-mal Dijkstra und n-mal Bellman-Ford im Vergleich"):
    st.table({"Verfahren": [f"Floyd-Warshall ({C.VARIANT_LABELS[a.variant].split(' (')[0]})", "n-mal Dijkstra", "n-mal Bellman-Ford"], "Zähler": [_num(m["comparisons"]), _num(m["dijkstra_checks"]), _num(m["bf_checks"])],
              "Laufzeit [ms]": [f"{a.seconds['fw'] * 1000:.1f}", f"{a.seconds['dj'] * 1000:.1f}", f"{a.seconds['bf'] * 1000:.1f}"]})
    st.caption("Die Laufzeiten sind Messwerte dieses Laufs (ein Lauf, schwankend). Floyd-Warshall läuft je nach Variante als Matrixoperation (vektorisiert, in C) oder als Python-Schleife; n-mal Dijkstra und n-mal Bellman-Ford sind reines Python: "
               "Laufzeiten dieser beiden Umsetzungen sind nicht als Vergleich der Verfahren zu lesen. Die Zähler oben sind es.")

st.markdown("---")

# --- Experimente -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die Reihenfolge der drei Schleifen")
if st.button("Alle sechs Reihenfolgen vergleichen (dauert einen Moment)", key="loop_start"):
    st.session_state["loop_on"] = True
if st.session_state.get("loop_on"):
    with st.spinner("Rechne sechs Reihenfolgen × 5 Netze ..."):
        lrows = _loop_orders()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_loop_order(lrows), width="stretch", key="loop_chart")
    c2.table({"Reihenfolge": [r["order"] for r in lrows], "falsche Zellen": [_pct(r["wrong_share"], 1) for r in lrows], "Netze mit Fehler": [f"{r['nets_wrong']} von {r['nets']}" for r in lrows]})
    wrong = [r for r in lrows if r["order"] not in ("kij", "kji")]
    st.caption(f"Zufallsnetze mit 10 Knoten ohne negative Kanten, Mittel über 5 Netze. Nur wenn **k außen** steht (kij, kji), ist die Matrix richtig. Bei den anderen vier Reihenfolgen liegen zwischen {min(r['wrong_share'] for r in wrong):.1%} und {max(r['wrong_share'] for r in wrong):.1%} "
               "der Zellen daneben: die Schleife über i und j sieht einen Zwischenknoten, bevor die Routen zu ihm fertig sind. Alle sechs Varianten haben denselben Aufwand n³ - falsch ist nur das Ergebnis.")

st.markdown("---")

st.subheader("📐 Wann lohnt sich n³?")
if st.button("Aufwand gegen Dichte und Größe messen (dauert einen Moment)", key="effort_start"):
    st.session_state["effort_on"] = True
if st.session_state.get("effort_on"):
    with st.spinner("Rechne Zähler für 4 Dichten und 4 Größen × 5 Netze ..."):
        drows, srows, comp = _effort_degree(), _effort_size(), _complete()
    c1, c2 = st.columns(2)
    c1.markdown("**Gegen die Dichte** (200 Knoten)")
    c1.plotly_chart(build_effort_degree(drows), width="stretch", key="effort_degree_chart")
    c2.markdown("**Gegen die Größe** (mittlerer Grad 3)")
    c2.plotly_chart(build_effort_size(srows), width="stretch", key="effort_size_chart")
    d2, b4 = drows[0], srows[-1]
    st.caption(f"Zufallsnetze ohne negative Kanten, Mittel über 5 Netze. Floyd-Warshall braucht immer n³ Vergleiche, egal wie viele Kanten das Netz hat. n-mal Dijkstra braucht nur n · m Kantenprüfungen: bei 200 Knoten und Grad 2 sind das {d2['fw'] / d2['dijkstra']:.0f}-mal weniger, "
               f"n-mal Bellman-Ford ist {d2['fw'] / d2['bf']:.0f}-mal billiger. Bei 400 Knoten und Grad 3 sind es {b4['fw'] / b4['dijkstra']:.0f}-mal bzw. {b4['fw'] / b4['bf']:.0f}-mal. Erst im **vollständigen Netz** (60 Knoten, jeder mit jedem verbunden) gleichen sich die Zahlen: "
               f"{_num(comp['fw'])} Vergleiche gegen {_num(comp['dijkstra'])} Kantenprüfungen bei Dijkstra. Die Variante \"überspringt\" spart in dünnen Netzen viel ({d2['skip'] / d2['fw']:.0%} von n³ bleiben bei Grad 2), im dichten Netz fast nichts. "
               "**Nach Zählern gewinnt Floyd-Warshall also nirgends** - was es bietet, ist der einfache Code, die Nachfolger-Matrix und Matrixoperationen, die in C laufen.")

st.markdown("---")

st.subheader("🔬 Negative Zyklen: die Werte wachsen")
if st.button("Wachstum der Werte messen (dauert einen Moment)", key="growth_start"):
    st.session_state["growth_on"] = True
if st.session_state.get("growth_on"):
    with st.spinner("Rechne 6 Größen × 5 Netze mit vielen negativen Zyklen ..."):
        grows = _growth()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_growth(grows), width="stretch", key="growth_chart")
    c2.table({"Knoten": [str(r["n"]) for r in grows], "kleinster Wert": [f"{r['low_median']:.1e}" for r in grows], "Paare ohne Route": [_pct(r["affected_share"]) for r in grows]})
    st.caption(f"Zufallsnetze mit Kosten von −3 bis 8 und mittlerem Grad 4 (Median über 5 Netze). Sobald es negative Zyklen gibt, bleibt die Matrix nicht nur unbrauchbar, ihre Werte **wachsen exponentiell**: bei 10 Knoten liegt der kleinste Wert um {grows[0]['low_median']:.0f}, "
               f"bei 80 schon bei {grows[3]['low_median']:.1e}, bei 400 bei {grows[5]['low_median']:.1e} (float64 reicht bis etwa 10³⁰⁸; das erreicht hier keines der Netze). In diesen Netzen haben fast alle Knoten eine negative Diagonale und für rund {grows[-1]['affected_share']:.0%} "
               "der Paare gibt es keine kürzeste Route. Deshalb prüft man die Diagonale, statt den Zahlen zu vertrauen.")

st.markdown("---")

st.subheader("🔬 Die Reihenfolge der Zwischenknoten")
if st.button("Vier Reihenfolgen der Zwischenknoten vergleichen (dauert einen Moment)", key="korder_start"):
    st.session_state["korder_on"] = True
if st.session_state.get("korder_on"):
    with st.spinner("Rechne 4 Reihenfolgen × 2 Netztypen × 5 Netze ..."):
        krows = _intermediate()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_intermediate_order(krows), width="stretch", key="korder_chart")
    kb = {r["net"]: r for r in krows}
    c2.table({"Reihenfolge": ["wie im Netz", "wenige Nachbarn zuerst", "viele Nachbarn zuerst", "zufällig"],
              "Zufallsnetz": [f"{kb['random'][k]:.0%}" for k in ("index", "degree_asc", "degree_desc", "random")], "Stadtnetz": [f"{kb['city'][k]:.0%}" for k in ("index", "degree_asc", "degree_desc", "random")]})
    st.caption(f"Anteil der Vergleiche von \"überspringt\" an n³ (Zufallsnetz mit 200 Knoten, Stadtnetz 14 × 14, Mittel über 5 Netze); die Matrix bleibt in jeder Reihenfolge dieselbe. Werden Knoten mit **wenigen Nachbarn zuerst** zugelassen (wie beim Zusammenziehen in Contraction Hierarchies), "
               f"sinken die Vergleiche im Zufallsnetz von {kb['random']['index']:.0%} auf {kb['random']['degree_asc']:.0%}; im Stadtnetz bringt das nichts ({kb['city']['index']:.0%} gegen {kb['city']['degree_asc']:.0%}). **Viele Nachbarn zuerst** kostet in beiden Netzen deutlich mehr "
               f"({kb['random']['degree_desc']:.0%} und {kb['city']['degree_desc']:.0%}).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **n³ ist tragbar** | Floyd-Warshall braucht immer n³ Vergleiche: bei 400 Knoten und Grad 3 das 133-Fache der Kantenprüfungen von n-mal Dijkstra und gut das 16-Fache von n-mal Bellman-Ford. Dünne Netze bestrafen es am meisten. | **Johnson**: einmal Bellman-Ford, dann n-mal Dijkstra |
| **n² Speicher sind da** | Die Matrix hat n² Zellen: bei 400 Knoten 160 000 (1.3 MB als float64), bei 10 000 Knoten schon 100 Millionen (800 MB) - dazu die Nachfolger-Matrix. | Verfahren, die nur die gebrauchten Paare berechnen |
| **Es gibt keinen negativen Zyklus** | Die Diagonale wird negativ, und die Werte wachsen exponentiell (Median des kleinsten Werts bei 10 / 80 / 400 Knoten: −28 / −1.6 · 10¹⁰ / −3.6 · 10⁵⁰). Für viele Paare gibt es keine kürzeste Route. | (das Problem hat keine Lösung) |
| **Alle Paare werden gebraucht** | Für nur *k* Startknoten reichen *k* Läufe von Dijkstra: *k* · *m* Kantenprüfungen statt n³. | n-mal Dijkstra, **Johnson** |
| **Nur eine Kostenart** | Zeit gegen Energie gleichzeitig ist ein anderes Problem. | **Mehrkriterien-Routing** |
"""
)
st.caption("Die Nachbarn der Kürzeste-Wege-Linie: Johnson und Mehrkriterien-Routing (beide gebaut). Ebenfalls gebaut: Breitensuche, Dijkstra, bidirektionale Suche, Contraction Hierarchies und Bellman-Ford.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Gerichteter Graph $G=(V,E)$ mit Kosten $c_e \in \mathbb{R}$, Knoten $0,\dots,n-1$. Gesucht: $\delta(i,j)$, die kleinste Summe über alle Routen $i \leadsto j$ - für alle Paare.

**Rekursion.** $d_k(i,j)$ sei die billigste Route von $i$ nach $j$, deren Zwischenknoten alle in $\{0,\dots,k-1\}$ liegen. Dann gilt
$$d_0(i,j)=c_{ij}\ (\text{0 für } i=j,\ \infty \text{ ohne Kante}),\qquad d_{k+1}(i,j)=\min\bigl(d_k(i,j),\; d_k(i,k)+d_k(k,j)\bigr).$$
Eine beste Route über $\{0,\dots,k\}$ benutzt den Knoten $k$ entweder gar nicht (dann $d_k(i,j)$) oder genau einmal (ohne negativen Zyklus): dann besteht sie aus einer besten Route $i \leadsto k$ und einer besten Route $k \leadsto j$, beide nur über $\{0,\dots,k-1\}$.

**Korrektheit.** Induktion über $k$; $d_n = \delta$, wenn es keinen negativen Zyklus gibt. **Warum $k$ außen steht:** die Rekursion braucht für Schritt $k+1$ die **vollständige** Matrix $d_k$. Steht $k$ innen, benutzt die Schleife Werte, die zu einem späteren $k$ gehören, bevor die Zwischenrouten zu diesem $k$ fertig sind - Routen über mehrere Zwischenknoten fehlen (das Experiment zeigt 9 bis 14 % falsche Zellen).
Der Schritt darf **in place** erfolgen: Zeile $k$ und Spalte $k$ ändern sich in Schritt $k$ nicht (ohne negativen Zyklus), weil $d_k(k,k)=0$ ist.

**Nachfolger.** Wird $d(i,j)$ über $k$ verbessert, setzt man $\mathrm{nxt}(i,j)\leftarrow \mathrm{nxt}(i,k)$. Die Route $i \leadsto j$ ist dann $i, \mathrm{nxt}(i,j), \mathrm{nxt}(\mathrm{nxt}(i,j),j),\dots$ in $O(\text{Länge})$.

**Negative Zyklen.** Liegt $i$ auf einem negativen Zyklus, wird $d(i,i)<0$ (Verfolgt man den Zyklus, sinkt $d(i,i)$ unter 0). Umgekehrt zeigt ein negativer Diagonalwert eine Rundreise $i \leadsto i$ mit negativen Kosten, also einen negativen Zyklus, der von $i$ aus erreichbar ist und wieder zu $i$ zurückführt (nicht unbedingt durch $i$ selbst). Jeder Knoten **auf** einem negativen Zyklus bekommt eine negative Diagonale; Knoten in dessen Umgebung können dagegen eine Diagonale $\geq 0$ behalten, obwohl sich eine Rundreise durch sie beliebig verbilligen ließe. Für $(i,j)$ gibt es **keine kürzeste Route**, wenn ein Knoten $k$ mit $d(k,k)<0$ von $i$ aus erreichbar ist und $j$ von $k$ aus.
Die Erreichbarkeit (endlich oder $\infty$) stimmt auch dann. Die Werte selbst wachsen bei vielen negativen Zyklen exponentiell, weil jede Verbesserung Summen bereits verkleinerter Werte bildet.

**Aufwand.** $\Theta(n^3)$ Vergleiche für jedes Netz, $\Theta(n^2)$ Speicher. n-mal Dijkstra mit Fibonacci-Heap: $O(n\,m + n^2\log n)$; n-mal Bellman-Ford: $O(n^2 m)$ im Lehrbuch. Erst für $m = \Theta(n^2)$ (dichte Netze) sind $n^3$ und $n \cdot m$ von gleicher Größenordnung.
Bei **Johnson** ersetzt man die negativen Kosten mit Potenzialen aus einem einzigen Bellman-Ford-Lauf und rechnet dann n-mal Dijkstra ($c'_{uv}=c_{uv}+p(u)-p(v) \ge 0$).

**Verwandt.** Ersetzt man $(\min,+)$ durch $(\lor,\land)$ auf einer 0-1-Matrix, berechnet dieselbe Schleife die **transitive Hülle** (Warshall): welche Knoten erreichen einander? (hier nicht gebaut)

Implementiert in `fw_graph.py` (CSR-Graph), `fw_algorithm.py` (drei Varianten, Nachfolger-Matrix, Zyklen, Schleifenreihenfolgen), `fw_sp.py` (Dijkstra und Bellman-Ford als n-mal-Vergleich aus der Bellman-Ford-Demo), `fw_scenario.py` (Netze), `fw_evaluation.py` (Kennzahlen, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kürzeste Wege: von der Breitensuche bis RAPTOR](https://sebastianhanisch.net/konzepte-kuerzeste-wege.html)."
)
