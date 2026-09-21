"""Floyd-Warshall gegen n-mal Dijkstra und n-mal Bellman-Ford: Kennzahlen für ein Netz, Paarwahl für die Routen-Ansicht, Experimente (Schleifenreihenfolge, Aufwand gegen Dichte und Größe, negative Zyklen,
Reihenfolge der Zwischenknoten), Urteil für die App.

Der Aufwand wird in Zählern gemessen (Vergleiche, Kantenprüfungen; plattformfest). Die Zähler sind kein gemeinsames Maß (ein Vergleich in einer Matrixoperation ist billiger als eine Kantenprüfung samt Warteschlange);
Laufzeiten stehen nur als Messwerte in der App."""

import time
from dataclasses import dataclass

import numpy as np

import fw_algorithm as alg
import fw_constants as C
from fw_graph import from_arcs
from fw_scenario import build_random, city_network, make_network


@dataclass(frozen=True)
class Analysis:
    net: object
    fw: alg.FloydWarshall
    variant: str
    s: int
    t: int
    metrics: dict
    seconds: dict
    dj_dist: object = None                 # n x n Entfernungen von n-mal Dijkstra (mit negativen Kanten nicht mehr richtig)


def default_variant(n, variant=C.DEFAULT_VARIANT):
    """Die Python-Schleifen gibt es nur bis PYTHON_VARIANT_MAX_NODES Knoten (n³ Schritte in reinem Python)."""
    return variant if variant == "numpy" or n <= C.PYTHON_VARIANT_MAX_NODES else "numpy"


def pick_pair(net, fw, distance_pct=60, seed=C.DEFAULT_SEED):
    """Start und Ziel der Routen-Ansicht: bei den kleinen Netzen die feste Aufgabe (Depot -> Hafen); sonst ein Start (Netze mit Karte: nahe bei 30 % Breite und 50 % Höhe, sonst zufällig) und als Ziel der Knoten,
    dessen Entfernung vom Start in der Rangfolge aller erreichbaren Knoten bei `distance_pct` Prozent liegt."""
    g = net.graph
    if g.names:
        return 0, g.n - 1
    if net.geometric:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        s = int(np.argmin(np.hypot(*(g.xy - (lo + (hi - lo) * np.array([0.3, 0.5]))).T)))
    else:
        s = int(np.random.default_rng([int(seed), 808]).integers(0, g.n))
    d = fw.dist[s]
    reachable = np.where(np.isfinite(d))[0]
    reachable = reachable[reachable != s]
    if not len(reachable):
        return s, s
    order = reachable[np.argsort(d[reachable], kind="stable")]
    t = int(order[min(len(order) - 1, int(round(distance_pct / 100.0 * (len(order) - 1))))])
    return s, t


def analyse(net, variant=C.DEFAULT_VARIANT, distance=60, seed=C.DEFAULT_SEED):
    g = net.graph
    n = g.n
    variant = default_variant(n, variant)
    t0 = time.perf_counter()
    fw = alg.floyd_warshall(g, variant, trace=True)
    t_fw = time.perf_counter() - t0
    counts = fw if variant == "numpy" else alg.floyd_warshall(g, "numpy")               # der Zähler "skip" aus der Matrixoperation (billig)
    t0 = time.perf_counter()
    dj_d, dj_checks = alg.all_pairs_by_dijkstra(g)
    t_dj = time.perf_counter() - t0
    t0 = time.perf_counter()
    bf_d, bf_checks, bf_cycles = alg.all_pairs_by_bellman_ford(g)
    t_bf = time.perf_counter() - t0
    cyc = fw.negative_cycle
    off = ~np.eye(n, dtype=bool)
    fin = np.isfinite(fw.dist)
    aff = alg.affected_pairs(fw)
    neg_edges = int((g.weight < 0).sum())
    s, t = pick_pair(net, fw, distance, seed)
    m = {"n": n, "m": g.m, "comparisons": fw.counters["comparisons"], "skip_comparisons": counts.counters["skip_comparisons"], "improvements": fw.counters["improvements"],
         "dijkstra_checks": dj_checks, "bf_checks": bf_checks, "bf_cycle_sources": bf_cycles, "cycle": cyc, "diag_negative": int(len(fw.diagonal_negative())),
         "affected": int((aff & off).sum()) if cyc else 0, "pairs": n * (n - 1), "reachable_share": float(fin[off].mean()) if n > 1 else 1.0,
         "neg_edges": neg_edges, "neg_share": neg_edges / max(g.m, 1), "cells": n * n,
         "dijkstra_wrong": int((~np.isclose(dj_d, fw.dist) & ~(np.isinf(dj_d) & np.isinf(fw.dist))).sum()) if not cyc else -1,
         "bf_exact": bool(not cyc and bf_cycles == 0 and np.allclose(bf_d[fin], fw.dist[fin]) and np.array_equal(np.isfinite(bf_d), fin)),
         "hops_max": int(alg.hop_matrix(fw).max()) if not cyc else -1, "s": s, "t": t}
    m["cost_fw"] = float(fw.dist[s, t]) if not cyc else float("nan")
    m["route_fw"] = fw.route(s, t)
    m["pair_affected"] = bool(aff[s, t]) if cyc else False
    m["cost_dj_pair"] = float(dj_d[s, t])
    return Analysis(net, fw, variant, s, t, m, {"fw": t_fw, "dj": t_dj, "bf": t_bf}, dj_d)


def verdict(a):
    """Code für die App: cycle / dijkstra_wrong / all_right_negative / no_negative."""
    m = a.metrics
    if m["cycle"]:
        return "cycle"
    if m["neg_edges"] == 0:
        return "no_negative"
    return "dijkstra_wrong" if m["dijkstra_wrong"] > 0 else "all_right_negative"


# --- Experimente -------------------------------------------------------------------------------------------------------------------------------

def loop_order_table(n=10, degree=2.5, seeds=C.SWEEP_SEEDS):
    """Die Lehr-Falle: alle sechs Schleifenreihenfolgen auf kleinen Zufallsnetzen (ohne negative Kanten): Anteil falscher Zellen der Matrix und Zahl der Netze mit mindestens einer falschen Zelle."""
    rows = []
    for order in alg.LOOP_ORDERS:
        wrong, nets = [], 0
        for sd in seeds:
            g = build_random(n, degree, 0, sd)
            ref = alg.floyd_warshall(g).dist
            bad = float((alg.floyd_warshall_order(g, order) != ref).sum()) / (n * n)
            wrong.append(bad)
            nets += bad > 0
        rows.append({"order": order, "wrong_share": float(np.mean(wrong)), "nets_wrong": nets, "nets": len(seeds)})
    return rows


def _effort_row(g):
    fw = alg.floyd_warshall(g, "numpy")
    d, dj = alg.all_pairs_by_dijkstra(g)
    b, bf, cyc = alg.all_pairs_by_bellman_ford(g)
    assert cyc == 0 and np.allclose(d, fw.dist) and np.allclose(b, fw.dist)
    return {"n": g.n, "m": g.m, "fw": fw.counters["comparisons"], "skip": fw.counters["skip_comparisons"], "dijkstra": dj, "bf": bf}


def effort_vs_degree(degrees=(2.0, 3.0, 6.0, 12.0), n=200, seeds=C.SWEEP_SEEDS):
    """Zähler gegen die Dichte (Zufallsnetz ohne negative Kanten, n Knoten): Floyd-Warshall n³, "überspringen", n-mal Dijkstra, n-mal Bellman-Ford."""
    rows = []
    for deg in degrees:
        acc = {}
        for sd in seeds:
            for k, v in _effort_row(build_random(n, deg, 0, sd)).items():
                acc.setdefault(k, []).append(v)
        rows.append({"degree": deg, **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


def effort_vs_size(sizes=(50, 100, 200, 400), degree=3.0, seeds=C.SWEEP_SEEDS):
    rows = []
    for n in sizes:
        acc = {}
        for sd in seeds:
            for k, v in _effort_row(build_random(n, degree, 0, sd)).items():
                acc.setdefault(k, []).append(v)
        rows.append({"size": n, **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


def free_graph(n, degree, seed, lo=-3, hi=9):
    """Zufallsnetz mit beliebigen ganzzahligen Kosten von lo bis hi - 1 (meist viele negative Zyklen)."""
    rng = np.random.default_rng([int(seed), 313])
    arcs = {}
    while len(arcs) < n * degree:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (u, v) not in arcs:
            arcs[(u, v)] = float(rng.integers(lo, hi))
    return from_arcs(n, [(u, v, w) for (u, v), w in arcs.items()], np.zeros((n, 2)), directed=True)


def cycle_growth(ns=(10, 20, 40, 80, 200, 400), degree=4, seeds=C.SWEEP_SEEDS):
    """Negative Zyklen: Zufallsnetze mit Kosten -3 bis 8. Kleinster Wert der Matrix (Median über die Netze), Anteil der Paare ohne kürzeste Route und Zahl der Knoten mit negativer Diagonale (Mittel)."""
    rows = []
    for n in ns:
        lows, aff, diag, cyc = [], [], [], 0
        for sd in seeds:
            fw = alg.floyd_warshall(free_graph(n, degree, sd), "numpy")
            cyc += fw.negative_cycle
            lows.append(float(fw.dist[np.isfinite(fw.dist)].min()))
            aff.append(float((alg.affected_pairs(fw) & ~np.eye(n, dtype=bool)).sum()) / (n * (n - 1)))
            diag.append(len(fw.diagonal_negative()))
        rows.append({"n": n, "low_median": float(np.median(lows)), "affected_share": float(np.mean(aff)), "diag_nodes": float(np.mean(diag)), "nets_with_cycle": cyc, "nets": len(seeds)})
    return rows


def _permuted(g, perm):
    inv = np.empty(len(perm), dtype=int)
    inv[perm] = np.arange(len(perm))
    arcs = [(int(inv[u]), int(inv[int(v)]), float(w)) for u in range(g.n) for v, w in zip(g.out(u), g.out_weights(u))]
    return from_arcs(g.n, arcs, g.xy[perm], directed=True)


def intermediate_order_table(seeds=C.SWEEP_SEEDS, n=200, degree=3.0, side=14):
    """Reihenfolge der Zwischenknoten k (die Knoten werden umnummeriert): Anteil der Vergleiche, die "überspringen" braucht (gegen n³), für Nummer wie im Netz, wenig Nachbarn zuerst (wie beim Zusammenziehen in Contraction
    Hierarchies), viele Nachbarn zuerst und zufällig. Das Ergebnis der Matrix bleibt dasselbe."""
    rows = []
    for name, make in (("random", lambda sd: build_random(n, degree, 0, sd)), ("city", lambda sd: city_network(side, C.DEFAULT_REACH, C.DEFAULT_SPREAD, sd).graph)):
        acc = {k: [] for k in ("index", "degree_asc", "degree_desc", "random")}
        for sd in seeds:
            g = make(sd)
            deg = g.degree()
            perms = {"index": np.arange(g.n), "degree_asc": np.argsort(deg, kind="stable"), "degree_desc": np.argsort(-deg, kind="stable"), "random": np.random.default_rng(sd).permutation(g.n)}
            for k, p in perms.items():
                acc[k].append(alg.floyd_warshall(_permuted(g, p), "numpy").counters["skip_comparisons"] / g.n ** 3)
        rows.append({"net": name, **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


def complete_graph_effort(n=60, seeds=C.SWEEP_SEEDS):
    """Der Grenzfall dichtes Netz: vollständiger gerichteter Graph mit zufälligen Kosten 1 bis 9. Hier prüft n-mal Dijkstra genau n² (n - 1) Kanten - fast so viele Schritte wie Floyd-Warshall mit n³."""
    rows = []
    for sd in seeds:
        rng = np.random.default_rng([int(sd), 909])
        arcs = [(u, v, float(rng.integers(1, 10))) for u in range(n) for v in range(n) if u != v]
        rows.append(_effort_row(from_arcs(n, arcs, np.zeros((n, 2)), directed=True)))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
