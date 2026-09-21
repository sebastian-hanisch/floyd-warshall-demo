"""Floyd-Warshall (alle Varianten) und die Vergleichsverfahren gegen networkx und unabhängige Rechnungen."""

import networkx as nx
import numpy as np
import pytest

import fw_algorithm as alg
import fw_sp
from fw_graph import from_arcs, route_cost

INF = float("inf")


def _random_arcs(n, m, seed, kind):
    """kind: 'positive', 'potential' (negative Kanten, nie ein Zyklus: Kosten = c + p(u) - p(v)), 'free' (beliebige Kosten, meist mit negativen Zyklen)."""
    rng = np.random.default_rng(seed)
    pot = rng.integers(0, 15, n)
    arcs = {}
    while len(arcs) < m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u == v or (u, v) in arcs:
            continue
        c = int(rng.integers(1, 10))
        if kind == "potential":
            c = c + int(pot[u]) - int(pot[v])
        elif kind == "free":
            c = int(rng.integers(-3, 10))
        arcs[(u, v)] = c
    return [(u, v, float(c)) for (u, v), c in arcs.items()]


def _graph(n, arcs):
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=True)


def _nx(n, arcs):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v, c in arcs:
        G.add_edge(u, v, weight=c)
    return G


CASES = [(n, m, seed, kind) for kind in ("positive", "potential", "free") for n, m, seed in ((6, 12, 1), (12, 30, 2), (20, 40, 3), (25, 30, 4), (10, 9, 5))]


@pytest.mark.parametrize("variant", alg.VARIANTS)
@pytest.mark.parametrize("n,m,seed,kind", CASES)
def test_matches_networkx_for_every_variant(variant, n, m, seed, kind):
    arcs = _random_arcs(n, m, seed, kind)
    g = _graph(n, arcs)
    G = _nx(n, arcs)
    r = alg.floyd_warshall(g, variant)
    assert r.negative_cycle == nx.negative_edge_cycle(G), (variant, kind)
    if not r.negative_cycle:
        ref = np.array(nx.floyd_warshall_numpy(G, weight="weight"))
        assert np.array_equal(np.isfinite(ref), np.isfinite(r.dist)) and np.allclose(ref[np.isfinite(ref)], r.dist[np.isfinite(ref)])
        for i in range(n):
            for j in range(n):
                route = r.route(i, j)
                if np.isfinite(ref[i, j]):
                    assert route[0] == i and route[-1] == j and len(set(route)) == len(route) and route_cost(g, route) == pytest.approx(ref[i, j])
                else:
                    assert route == []


@pytest.mark.parametrize("kind", ("positive", "potential"))
@pytest.mark.parametrize("seed", range(4))
def test_all_three_variants_give_identical_matrices_and_routes(kind, seed):
    arcs = _random_arcs(15, 40, seed, kind)
    g = _graph(15, arcs)
    a, b, c = (alg.floyd_warshall(g, v) for v in alg.VARIANTS)
    assert np.array_equal(a.dist, b.dist) and np.array_equal(a.dist, c.dist)
    assert np.array_equal(a.nxt, b.nxt) and np.array_equal(a.nxt, c.nxt)
    assert a.counters["improvements"] == b.counters["improvements"] == c.counters["improvements"]


@pytest.mark.parametrize("variant", alg.VARIANTS)
def test_counters_classic_is_n_cubed_and_skip_counts_agree(variant):
    arcs = _random_arcs(14, 22, 3, "positive")
    g = _graph(14, arcs)
    r = alg.floyd_warshall(g, variant)
    assert r.counters["comparisons"] == (14 ** 3 if variant != "skip" else r.counters["skip_comparisons"])
    skip = alg.floyd_warshall(g, "skip")
    numpy_ = alg.floyd_warshall(g, "numpy")
    assert skip.counters["comparisons"] == numpy_.counters["skip_comparisons"] < 14 ** 3                          # skip zählt genau die Paare mit endlichem d(i,k), d(k,j)


def test_skip_saves_nothing_on_a_complete_graph_and_much_on_a_sparse_one():
    n = 10
    complete = _graph(n, [(u, v, 1.0) for u in range(n) for v in range(n) if u != v])
    assert alg.floyd_warshall(complete, "skip").counters["comparisons"] == n ** 3
    path = _graph(n, [(i, i + 1, 1.0) for i in range(n - 1)])
    assert alg.floyd_warshall(path, "skip").counters["comparisons"] < n ** 3 // 3


def test_negative_cycle_shows_on_the_diagonal_and_blocks_routes():
    g = _graph(4, [(0, 1, 1.0), (1, 2, -3.0), (2, 1, 1.0), (2, 3, 2.0)])
    for v in alg.VARIANTS:
        r = alg.floyd_warshall(g, v)
        assert r.negative_cycle and set(r.diagonal_negative().tolist()) >= {1, 2} and r.route(0, 3) == []
        aff = alg.affected_pairs(r)
        assert aff[0, 3] and aff[0, 1] and aff[1, 3] and not aff[3, 0] and not aff[3, 3]
        assert aff.sum() == sum(1 for i in range(4) for j in range(4) if i in (0, 1, 2) and j in (1, 2, 3))


def test_affected_pairs_matches_bellman_ford_from_every_source():
    arcs = _random_arcs(14, 30, 6, "free")
    g = _graph(14, arcs)
    r = alg.floyd_warshall(g)
    aff = alg.affected_pairs(r)
    for s in range(14):
        bf = fw_sp.bellman_ford(g, s)
        assert bf.negative_cycle == bool(aff[s].any() and any(np.diag(r.dist)[j] < 0 and np.isfinite(r.dist[s, j]) for j in range(14)))


def test_unreachable_zero_edges_single_node_and_parallel_edges():
    g = _graph(4, [(0, 1, 0.0), (1, 0, 0.0), (1, 2, 4.0)])
    r = alg.floyd_warshall(g)
    assert list(r.dist[0]) == [0.0, 0.0, 4.0, INF] and r.route(0, 3) == [] and r.route(0, 2) == [0, 1, 2] and r.route(2, 2) == [2]
    solo = alg.floyd_warshall(_graph(1, []))
    assert solo.dist.tolist() == [[0.0]] and not solo.negative_cycle


@pytest.mark.parametrize("variant", alg.VARIANTS)
def test_invariant_after_step_k_only_the_first_k_nodes_are_intermediates(variant):
    arcs = _random_arcs(9, 22, 2, "potential")
    g = _graph(9, arcs)
    r = alg.floyd_warshall(g, variant, trace=True)
    assert [h[0] for h in r.history] == list(range(1, 10))
    for k, dist, nxt, changed in r.history:
        assert np.allclose(dist, alg.restricted_reference(g, k), equal_nan=False)
        assert np.isfinite(dist[changed]).all()
        for i in range(9):
            for j in range(9):                                                                               # jede Route des Snapshots benutzt nur Zwischenknoten 0 .. k-1 und hat die Snapshot-Kosten
                route = alg.route_from(nxt, i, j)
                if np.isfinite(dist[i, j]):
                    assert route and all(v < k for v in route[1:-1]) and route_cost(g, route) == pytest.approx(dist[i, j])
                else:
                    assert route == []


@pytest.mark.parametrize("variant", alg.VARIANTS)
def test_large_traces_keep_about_forty_snapshots_and_collect_the_changes_in_between(variant):
    n = 120 if variant != "numpy" else 130
    g = _graph(n, _random_arcs(n, 3 * n, 4, "positive"))
    r = alg.floyd_warshall(g, variant, trace=True)
    ks = [h[0] for h in r.history]
    assert ks[-1] == n and len(ks) <= 45 and ks == sorted(set(ks))
    assert sum(int(h[3].sum()) for h in r.history) >= r.counters["improvements"] // (n)                    # gesammelte Zellen (jede Zelle einmal je Eintrag)


def test_hop_matrix_counts_edges_of_the_shortest_routes():
    g = _graph(6, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 9.0), (4, 5, 1.0)])
    r = alg.floyd_warshall(g)
    h = alg.hop_matrix(r)
    assert h[0, 3] == 3 and h[0, 2] == 2 and h[3, 0] == -1 and h[4, 5] == 1 and h[2, 2] == 0
    assert all(h[i, j] == len(r.route(i, j)) - 1 for i in range(6) for j in range(6) if r.route(i, j))


def test_only_k_outermost_loop_orders_are_right():
    right = {"kij", "kji"}
    wrong_seen = {o: 0 for o in alg.LOOP_ORDERS}
    for seed in range(12):
        arcs = _random_arcs(8, 14, seed, "positive")
        g = _graph(8, arcs)
        ref = alg.floyd_warshall(g).dist
        for o in alg.LOOP_ORDERS:
            same = np.array_equal(alg.floyd_warshall_order(g, o), ref)
            if o in right:
                assert same, o
            elif not same:
                wrong_seen[o] += 1
    assert all(wrong_seen[o] > 0 for o in alg.LOOP_ORDERS if o not in right), wrong_seen              # jede falsche Reihenfolge liegt bei einigen Netzen daneben


def test_loop_order_rejects_unknown_orders():
    with pytest.raises(ValueError):
        alg.floyd_warshall_order(_graph(2, []), "kkk")
    with pytest.raises(ValueError):
        alg.floyd_warshall(_graph(2, []), "gpu")


# --- Vergleichsverfahren -----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
def test_n_times_dijkstra_and_bellman_ford_agree_with_floyd_warshall(seed):
    pos = _graph(20, _random_arcs(20, 60, seed, "positive"))
    d, checks = alg.all_pairs_by_dijkstra(pos)
    assert np.allclose(d, alg.floyd_warshall(pos).dist) and 0 < checks <= 20 * pos.m                      # nur Kanten erreichbarer Knoten werden geprüft
    ring = _graph(20, [(i, (i + 1) % 20, 1.0) for i in range(20)] + [(i, (i + 5) % 20, 2.0) for i in range(20)])
    assert alg.all_pairs_by_dijkstra(ring)[1] == 20 * ring.m                                              # stark zusammenhängend: genau n * m
    neg = _graph(20, _random_arcs(20, 60, seed, "potential"))
    b, bchecks, cycles = alg.all_pairs_by_bellman_ford(neg)
    assert cycles == 0 and np.allclose(b, alg.floyd_warshall(neg).dist) and bchecks > checks * 0


def test_n_times_dijkstra_is_wrong_with_negative_edges_where_floyd_warshall_is_right():
    neg = _graph(20, _random_arcs(20, 60, 1, "potential"))
    d, _ = alg.all_pairs_by_dijkstra(neg)
    assert not np.allclose(d, alg.floyd_warshall(neg).dist)
