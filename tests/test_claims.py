"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-21, Toleranzen fangen Rundung ab). Vergleiche, Kantenprüfungen und ganzzahlige Kosten sind plattformfest;
Laufzeiten stehen in der App nur als Messwerte und werden hier nie geprüft."""

import numpy as np
import pytest

import fw_algorithm as alg
import fw_constants as C
import fw_evaluation as ev
import fw_scenario as sc

PRESET = {"small": "🔀 Kleines Netz", "small_cycle": "🔁 Negativer Zyklus", "city": "🏙️ Stadtnetz", "ev": "🔋 E-Lieferwagen", "random": "🕸️ Zufallsnetz"}


def _preset(key):
    p = C.PRESETS[PRESET[key]]
    net = sc.make_network(p["net"], p["side"], p["hill"], p["eta"], p["reach"], p["spread"], p["nodes"], p["degree"], p["pot"], p["seed"])
    return p, net, ev.analyse(net, p["variant"], p["distance"], p["seed"])


def _has(key, *needles):
    help_ = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in help_, (key, n)


# --- Preset-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_small_preset_numbers():
    _, net, a = _preset("small")
    m = a.metrics
    assert (m["comparisons"], m["improvements"], m["cost_fw"], m["cost_dj_pair"], m["dijkstra_wrong"], m["pairs"]) == (216, 26, 6.0, 8.0, 7, 30) and ev.verdict(a) == "dijkstra_wrong"
    assert (a.s, a.t) == (0, 5) and [net.graph.names[i] for i in m["route_fw"]] == ["Depot", "Ost", "Nord", "Süd", "Hafen"]
    _has("small", "216 Vergleiche (6³)", "26 davon", "6 Euro", "nur 8", "7 von 30 Paaren")


def test_small_cycle_preset_numbers():
    _, net, a = _preset("small_cycle")
    m = a.metrics
    assert ev.verdict(a) == "cycle" and [net.graph.names[i] for i in a.fw.diagonal_negative()] == ["Süd", "West"] and (m["affected"], m["pairs"]) == (13, 30)
    assert round(m["reachable_share"] * m["pairs"]) == 16
    _has("small_cycle", "−2 Euro", "Süd und West", "13 von 30 Paaren", "16 von 30")


def test_city_preset_numbers():
    _, net, a = _preset("city")
    m = a.metrics
    assert (m["n"], m["comparisons"], m["dijkstra_checks"], m["bf_checks"], m["hops_max"]) == (64, 262144, 21504, 142464, 9) and ev.verdict(a) == "no_negative" and m["dijkstra_wrong"] == 0
    _has("city", "8 × 8 = 64", "262 144 Vergleiche (64³)", "21 504", "142 464", "9 Kanten")


def test_ev_preset_numbers():
    _, net, a = _preset("ev")
    m = a.metrics
    assert (m["neg_edges"], m["m"], m["dijkstra_wrong"], m["pairs"], m["cost_fw"], m["cost_dj_pair"]) == (42, 336, 524, 4032, 12.0, 15.0) and ev.verdict(a) == "dijkstra_wrong" and m["bf_exact"]
    _has("ev", "42 von 336", "524 von 4 032", "Abstand 20 %", "12 Wh", "15")


def test_random_preset_numbers():
    _, net, a = _preset("random")
    m = a.metrics
    assert (m["n"], m["m"], m["comparisons"], m["dijkstra_checks"], m["bf_checks"], m["skip_comparisons"]) == (100, 600, 1000000, 60000, 319800, 591649) and ev.verdict(a) == "no_negative"
    assert m["comparisons"] / m["dijkstra_checks"] == pytest.approx(16.7, abs=0.05) and 1 - m["skip_comparisons"] / m["comparisons"] == pytest.approx(0.41, abs=0.005)
    _has("random", "1 000 000 Vergleiche", "60 000 Kantenprüfungen (16.7-fach", "319 800", "41 %", "591 649")


# --- Sidebar-Hilfen ----------------------------------------------------------------------------------------------------------------------------

def test_size_and_node_help_numbers_are_the_cubes():
    assert [s * s for s in (8, 14, 20)] == [64, 196, 400] and [n ** 3 for n in (64, 196, 400)] == [262144, 7529536, 64000000]
    assert [n ** 3 for n in (100, 200, 400)] == [1000000, 8000000, 64000000]
    assert C.SIDE_MAX ** 2 == 400 and C.NODES_MAX == 400


def test_degree_help_numbers_skip_share():
    rows = ev.effort_vs_degree()
    assert [r["skip"] / r["fw"] * 100 for r in rows] == pytest.approx([20, 34, 60, 80], abs=0.7)              # Hilfe: 20 % / 34 % / 60 % / 80 %


# --- Experimente und die Tabelle "Wo die Annahmen enden" ----------------------------------------------------------------------------------------

def test_loop_order_claims():
    rows = {r["order"]: r for r in ev.loop_order_table()}
    assert rows["kij"]["wrong_share"] == 0 == rows["kji"]["wrong_share"]
    wrong = [rows[o]["wrong_share"] for o in ("ikj", "jki", "ijk", "jik")]
    assert 0.09 <= min(wrong) and max(wrong) <= 0.14                                                          # "9 bis 14 % falsche Zellen"
    assert [rows[o]["nets_wrong"] for o in ("ikj", "jki", "ijk", "jik")] == [5, 5, 4, 4]


def test_effort_claims():
    d = {r["degree"]: r for r in ev.effort_vs_degree()}
    assert d[2.0]["fw"] / d[2.0]["dijkstra"] == pytest.approx(100.0) and d[2.0]["fw"] / d[2.0]["bf"] == pytest.approx(11.2, abs=0.1)
    s = {int(r["size"]): r for r in ev.effort_vs_size()}
    assert s[400]["fw"] == 64000000 and s[400]["dijkstra"] == 480000 and s[400]["fw"] / s[400]["dijkstra"] == pytest.approx(133.3, abs=0.1)     # Tabelle: "das 133-Fache"
    assert 16.0 <= s[400]["fw"] / s[400]["bf"] < 17.0                                                          # Tabelle: "gut das 16-Fache"
    assert all(r["fw"] > r["bf"] > r["dijkstra"] for r in list(d.values()) + list(s.values()))                # Floyd-Warshall gewinnt nach Zählern nirgends
    comp = ev.complete_graph_effort()
    assert (comp["fw"], comp["dijkstra"]) == (216000, 212400) and comp["skip"] == comp["fw"]                  # vollständiges Netz: die Zahlen gleichen sich


def test_memory_arithmetic_in_the_table():
    assert 400 ** 2 == 160000 and 400 ** 2 * 8 / 1e6 == pytest.approx(1.28) and 10000 ** 2 == 10 ** 8 and 10000 ** 2 * 8 / 1e6 == 800.0


def test_cycle_growth_claims():
    rows = {r["n"]: r for r in ev.cycle_growth()}
    assert rows[10]["low_median"] == -28.0 and rows[80]["low_median"] == pytest.approx(-1.6e10, rel=0.02) and rows[400]["low_median"] == pytest.approx(-3.6e50, rel=0.02)      # Tabelle: −28 / −1.6·10¹⁰ / −3.6·10⁵⁰
    assert all(r["nets_with_cycle"] == r["nets"] for r in rows.values()) and rows[400]["affected_share"] == pytest.approx(0.96, abs=0.02)
    assert rows[400]["diag_nodes"] > 0.9 * 400                                                                # "fast alle Knoten haben eine negative Diagonale"
    assert all(np.isfinite(r["low_median"]) for r in rows.values()) and abs(rows[400]["low_median"]) < 1e308     # "das erreicht hier keines der Netze"


def test_intermediate_order_claims():
    rows = {r["net"]: r for r in ev.intermediate_order_table()}
    assert rows["random"]["index"] == pytest.approx(0.34, abs=0.01) and rows["random"]["degree_asc"] == pytest.approx(0.28, abs=0.01) and rows["random"]["degree_desc"] == pytest.approx(0.44, abs=0.01)
    assert rows["random"]["degree_asc"] < 0.85 * rows["random"]["index"]                                       # wenige Nachbarn zuerst: deutlich weniger im Zufallsnetz
    assert rows["city"]["degree_asc"] == pytest.approx(rows["city"]["index"], abs=0.02)                        # im Stadtnetz bringt es nichts
    assert rows["city"]["degree_desc"] > 1.3 * rows["city"]["index"] and rows["random"]["degree_desc"] > 1.2 * rows["random"]["index"]


def test_floyd_warshall_and_n_times_bellman_ford_agree_on_every_net_without_cycle():
    for key in ("small", "city", "ev", "random"):
        _, net, a = _preset(key)
        assert a.metrics["bf_exact"], key
