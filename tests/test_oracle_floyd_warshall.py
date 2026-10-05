"""Unabhängiges Orakel: Matrix je Quelle per networkx-Bellman-Ford (inkl. Nullkanten und Nullzyklen), betroffene Paare bei negativen Zyklen per Bellman-Ford mit Zusatzrunden und
Erreichbarkeits-Hülle (anderer Rechenweg als die Diagonale), Zähler von "überspringt" und Verbesserungen aus einer eigenen Schleife, Zyklen der Bellman-Ford-Varianten."""

import networkx as nx
import numpy as np
import pytest

import fw_algorithm as alg
import fw_sp
from fw_graph import from_arcs, route_cost


def _arcs(rng, n, m, kind):
    pot = rng.integers(0, 12, n)
    arcs = {}
    for _ in range(4 * m + 4):
        if len(arcs) >= m:
            break
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u == v or (u, v) in arcs:
            continue
        c = int(rng.integers(0, 3)) if kind == "zero" else int(rng.integers(-3, 8)) if kind == "free" else int(rng.integers(0, 2)) + int(pot[u]) - int(pot[v])
        arcs[(u, v)] = float(c)
    return [(u, v, c) for (u, v), c in arcs.items()]


def _closure(n, arcs):
    reach = np.eye(n, dtype=bool)
    for u, v, _ in arcs:
        reach[u, v] = True
    for k in range(n):
        reach = reach | (reach[:, [k]] & reach[[k], :])
    return reach


def _no_shortest_route(n, arcs):
    """Paare ohne kürzeste Route: j ist von einem Knoten erreichbar, der in Runde n noch verbessert wird (Bellman-Ford je Start)."""
    reach, out = _closure(n, arcs), np.zeros((n, n), bool)
    for s in range(n):
        d = [np.inf] * n
        d[s] = 0.0
        for _ in range(n - 1):
            for u, v, c in arcs:
                if d[u] + c < d[v]:
                    d[v] = d[u] + c
        seeds = [v for u, v, c in arcs if d[u] + c < d[v]]
        out[s] = [any(reach[a, j] for a in seeds) for j in range(n)]
    return out


@pytest.mark.parametrize("kind", ["zero", "free", "potential"])
def test_matrix_cycles_routes_counters_and_affected_pairs(kind):
    rng = np.random.default_rng({"zero": 1, "free": 2, "potential": 3}[kind])
    cycles = 0
    for _ in range(40):
        n = int(rng.integers(2, 10))
        arcs = _arcs(rng, n, int(rng.integers(1, 3 * n)), kind)
        g = from_arcs(n, arcs, np.zeros((n, 2)), directed=True)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_weighted_edges_from(arcs)
        has_cycle = nx.negative_edge_cycle(G)
        reach = _closure(n, arcs)
        for variant in alg.VARIANTS:
            r = alg.floyd_warshall(g, variant)
            assert r.negative_cycle == has_cycle
            assert np.array_equal(np.isfinite(r.dist), reach)              # Erreichbarkeit stimmt auch mit negativen Zyklen
        if has_cycle:
            cycles += 1
            off = ~np.eye(n, dtype=bool)
            assert np.array_equal(alg.affected_pairs(r) & off, _no_shortest_route(n, arcs) & off)
            continue
        for variant in alg.VARIANTS:
            r = alg.floyd_warshall(g, variant)
            for s in range(n):
                ref = nx.single_source_bellman_ford_path_length(G, s)
                for t in range(n):
                    route = r.route(s, t)
                    if t in ref:
                        assert r.dist[s, t] == pytest.approx(ref[t]) and route[0] == s and route[-1] == t and route_cost(g, route) == pytest.approx(ref[t])
                    else:
                        assert not np.isfinite(r.dist[s, t]) and route == []
        # Zähler aus einer eigenen Schleife: Paare mit endlichem d(i,k) und d(k,j) vor Schritt k; Verbesserungen je Schritt
        d, skip, improved = alg.initial_matrix(g)[0], 0, 0
        for k in range(n):
            skip += int(np.isfinite(d[:, k]).sum()) * int(np.isfinite(d[k, :]).sum())
            cand = d[:, [k]] + d[[k], :]
            improved += int((cand < d).sum())
            d = np.minimum(d, cand)
        assert alg.floyd_warshall(g, "skip").counters["comparisons"] == skip == alg.floyd_warshall(g, "numpy").counters["skip_comparisons"]
        assert all(alg.floyd_warshall(g, v).counters["improvements"] == improved for v in alg.VARIANTS)
    if kind == "free":
        assert cycles > 0


@pytest.mark.parametrize("variant", fw_sp.VARIANTS)
def test_bellman_ford_variants_detect_cycles_like_networkx_and_report_a_real_negative_cycle(variant):
    rng = np.random.default_rng(9)
    for it in range(80):
        n = int(rng.integers(2, 9))
        arcs = _arcs(rng, n, int(rng.integers(1, 3 * n)), "free" if it % 2 else "potential")
        g = from_arcs(n, arcs, np.zeros((n, 2)), directed=True)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_weighted_edges_from(arcs)
        try:
            ref, cyc = nx.single_source_bellman_ford_path_length(G, 0), False
        except nx.NetworkXUnbounded:
            ref, cyc = None, True
        for cycle_check in ((False, True) if variant != "queue" else (False,)):
            r = fw_sp.bellman_ford(g, 0, variant, cycle_check=cycle_check)
            assert r.negative_cycle == cyc
            if cyc:
                assert r.cycle[0] == r.cycle[-1] and route_cost(g, r.cycle) < 0
            else:
                assert all((r.dist[v] == pytest.approx(ref[v])) if v in ref else not np.isfinite(r.dist[v]) for v in range(n))
