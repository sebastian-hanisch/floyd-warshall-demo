"""Netze (kleines Netz, Zyklus-Variante, Stadtnetz, E-Lieferwagen, Zufallsnetz), Paarwahl, Kennzahlen, Experimente."""

import networkx as nx
import numpy as np
import pytest

import fw_algorithm as alg
import fw_constants as C
import fw_evaluation as ev
import fw_scenario as sc


def _nx(g):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.out(u), g.out_weights(u)):
            G.add_edge(u, int(v), weight=float(w))
    return G


def _strong(g):
    return nx.number_strongly_connected_components(_nx(g))


# --- Netze -----------------------------------------------------------------------------------------------------------------------------------

def test_small_network_has_one_negative_edge_and_everything_is_reachable():
    net = sc.small_network()
    g = net.graph
    assert g.n == 6 and g.m == len(sc.SMALL_LEGS) == 9 and g.directed and net.unit == "Euro" and _strong(g) == 1
    assert [(g.names[u], g.names[int(v)], float(w)) for u in range(g.n) for v, w in zip(g.out(u), g.out_weights(u)) if w < 0] == [("Ost", "Nord", -3.0)]
    assert not alg.floyd_warshall(g).negative_cycle


def test_cycle_variant_differs_in_two_legs_and_has_exactly_one_negative_cycle():
    plain, cyc = sc.small_network(False), sc.small_network(True)
    assert set(sc.SMALL_LEGS) ^ set(sc.CYCLE_LEGS) == {("Hafen", "Depot", 3), ("Süd", "West", -3)}
    r = alg.floyd_warshall(cyc.graph)
    assert r.negative_cycle and [cyc.graph.names[i] for i in r.diagonal_negative()] == ["Süd", "West"]
    assert not alg.floyd_warshall(plain.graph).negative_cycle


@pytest.mark.parametrize("eta", (0, 30, 60, 90))
@pytest.mark.parametrize("seed", range(3))
def test_ev_network_never_has_a_negative_cycle_and_has_integer_costs(eta, seed):
    net = sc.ev_network(7, 40, eta, seed)
    g = net.graph
    assert (g.weight == np.rint(g.weight)).all() and _strong(g) == 1 and not alg.floyd_warshall(g).negative_cycle


def test_ev_network_needs_hills_and_recuperation_for_negative_edges():
    assert (sc.ev_network(8, 0, 90, 1).graph.weight >= 0).all() and (sc.ev_network(8, 30, 0, 1).graph.weight >= 0).all() and (sc.ev_network(8, 30, 90, 1).graph.weight < 0).any()


def test_random_network_is_strongly_connected_and_its_structure_does_not_depend_on_the_potential_span():
    graphs = [sc.build_random(120, 3.0, p, 4) for p in (0, 4, 20)]
    assert all(_strong(g) == 1 for g in graphs) and all(np.array_equal(graphs[0].indices, g.indices) for g in graphs)
    assert (graphs[0].weight >= 1).all() and (graphs[2].weight < 0).any() and abs(graphs[0].m / graphs[0].n - 3.0) < 0.05
    assert not alg.floyd_warshall(graphs[2]).negative_cycle


def test_city_network_is_connected_with_positive_whole_number_costs():
    g = sc.city_network(8, 1.5, 1.0, 2).graph
    assert _strong(g) == 1 and (g.weight >= 1).all() and (g.weight == np.rint(g.weight)).all() and g.n == 64
    assert np.array_equal(sc.make_network("city", 8, seed=2).graph.indices, g.indices)


def test_make_network_rejects_unknown_nets():
    with pytest.raises(ValueError):
        sc.make_network("ring")


# --- Kennzahlen -------------------------------------------------------------------------------------------------------------------------------

def test_default_variant_only_allows_python_loops_for_small_nets():
    assert ev.default_variant(6, "classic") == "classic" and ev.default_variant(C.PYTHON_VARIANT_MAX_NODES, "skip") == "skip"
    assert ev.default_variant(C.PYTHON_VARIANT_MAX_NODES + 1, "classic") == "numpy" and ev.default_variant(400, "numpy") == "numpy"


@pytest.mark.parametrize("key", C.NETS)
@pytest.mark.parametrize("variant", alg.VARIANTS)
def test_analysis_invariants_and_exactness_for_every_net_and_variant(key, variant):
    net = sc.make_network(key, side=5, nodes=40)
    a = ev.analyse(net, variant)
    m, g = a.metrics, net.graph
    G = _nx(g)
    assert m["cycle"] == nx.negative_edge_cycle(G) == (key == "small_cycle")
    assert m["comparisons"] == (g.n ** 3 if variant != "skip" else m["skip_comparisons"]) and m["skip_comparisons"] <= g.n ** 3 and m["cells"] == g.n ** 2
    if not m["cycle"]:
        ref = np.array(nx.floyd_warshall_numpy(G, weight="weight"))
        assert np.allclose(np.where(np.isfinite(ref), ref, 0), np.where(np.isfinite(a.fw.dist), a.fw.dist, 0)) and m["bf_exact"]
        assert m["dijkstra_wrong"] == int((~np.isclose(a.dj_dist, a.fw.dist) & np.isfinite(a.fw.dist)).sum())
        assert m["cost_fw"] == pytest.approx(a.fw.dist[a.s, a.t]) and len(m["route_fw"]) - 1 <= m["hops_max"]
    else:
        assert m["diag_negative"] > 0 and 0 < m["affected"] < m["pairs"] and m["hops_max"] == -1 and m["bf_cycle_sources"] > 0
    assert ev.verdict(a) in ("cycle", "dijkstra_wrong", "all_right_negative", "no_negative")
    if key == "city":
        assert m["neg_edges"] == 0 and ev.verdict(a) == "no_negative" and m["dijkstra_wrong"] == 0


def test_pick_pair_uses_the_fixed_task_for_small_nets_and_the_distance_rank_otherwise():
    small = sc.small_network()
    fw = alg.floyd_warshall(small.graph)
    assert ev.pick_pair(small, fw, 10, 1) == (0, 5) == ev.pick_pair(small, fw, 99, 5)
    net = sc.make_network("city", side=8)
    fwc = alg.floyd_warshall(net.graph)
    pairs = {p: ev.pick_pair(net, fwc, p, 7) for p in (10, 50, 100)}
    assert len({s for s, _ in pairs.values()}) == 1 and pairs == {p: ev.pick_pair(net, fwc, p, 7) for p in (10, 50, 100)}
    s = pairs[10][0]
    assert fwc.dist[s, pairs[10][1]] < fwc.dist[s, pairs[50][1]] <= fwc.dist[s, pairs[100][1]] and fwc.dist[s, pairs[100][1]] == np.max(fwc.dist[s])


def test_analysis_carries_the_dijkstra_matrix_and_trace_snapshots():
    a = ev.analyse(sc.small_network())
    assert a.dj_dist.shape == (6, 6) and len(a.fw.history) == 6 and a.fw.history[-1][0] == 6
    big = ev.analyse(sc.make_network("city", side=12))                       # n = 144 > 100: nur etwa 40 Zwischenstände
    assert 30 <= len(big.fw.history) <= 60 and big.fw.history[-1][0] == 144


# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

def test_loop_order_table_rows():
    rows = {r["order"]: r for r in ev.loop_order_table(n=8, seeds=C.SWEEP_SEEDS[:3])}
    assert list(rows) == list(alg.LOOP_ORDERS) and rows["kij"]["wrong_share"] == 0 == rows["kji"]["wrong_share"]
    assert all(rows[o]["wrong_share"] > 0 for o in ("ikj", "jki", "ijk", "jik"))


def test_effort_rows_order_the_counters_as_expected():
    rows = ev.effort_vs_degree(degrees=(2.0, 6.0), n=60, seeds=C.SWEEP_SEEDS[:2])
    for r in rows:
        assert r["fw"] == r["n"] ** 3 and r["fw"] > r["skip"] > 0 and r["fw"] > r["bf"] > r["dijkstra"]
    assert rows[0]["skip"] / rows[0]["fw"] < rows[1]["skip"] / rows[1]["fw"]                       # dünn: mehr Unerreichbares zu überspringen
    sizes = ev.effort_vs_size(sizes=(20, 40), seeds=C.SWEEP_SEEDS[:2])
    assert sizes[1]["fw"] == 8 * sizes[0]["fw"] and sizes[1]["fw"] / sizes[1]["dijkstra"] > sizes[0]["fw"] / sizes[0]["dijkstra"]


def test_complete_graph_makes_the_counters_meet():
    r = ev.complete_graph_effort(n=20, seeds=C.SWEEP_SEEDS[:2])
    assert r["fw"] == 20 ** 3 and r["dijkstra"] == 20 * 20 * 19 and r["skip"] == r["fw"]


def test_cycle_growth_rows_and_free_graph():
    rows = ev.cycle_growth(ns=(10, 40), seeds=C.SWEEP_SEEDS[:3])
    assert rows[1]["low_median"] < rows[0]["low_median"] < 0 and all(r["nets_with_cycle"] == 3 and r["affected_share"] > 0.5 for r in rows)
    g = ev.free_graph(20, 4, 1)
    assert g.m == 80 and (g.weight < 0).any() and np.array_equal(g.indices, ev.free_graph(20, 4, 1).indices)


def test_intermediate_order_table_rows_keep_the_result_and_change_the_counter():
    rows = ev.intermediate_order_table(seeds=C.SWEEP_SEEDS[:2], n=60, side=8)
    assert [r["net"] for r in rows] == ["random", "city"] and all(0 < r[k] < 1 for r in rows for k in ("index", "degree_asc", "degree_desc", "random"))
    g = sc.build_random(50, 3.0, 0, 1)
    perm = np.random.default_rng(0).permutation(50)
    assert np.allclose(np.sort(alg.floyd_warshall(ev._permuted(g, perm)).dist.ravel()), np.sort(alg.floyd_warshall(g).dist.ravel()))
