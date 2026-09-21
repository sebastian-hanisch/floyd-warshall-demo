"""Plotly-Abbildungen: Entfernungsmatrix als Heatmap (Zwischenknoten als Band, geänderte Zellen umrandet, Fehler von n-mal Dijkstra markiert), Netz mit Route und erlaubten Zwischenknoten, Experimente.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import fw_algorithm as alg
import fw_constants as C

ORDER_NAMES = {"index": "wie im Netz", "degree_asc": "wenige Nachbarn zuerst", "degree_desc": "viele Nachbarn zuerst", "random": "zufällig"}


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.12), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def state_at(a, step):
    """Zustand nach Eintrag `step` des Protokolls (0 = vor dem ersten k): k erledigt, Entfernungen, Nachfolger, geänderte Zellen (bool-Matrix)."""
    fw = a.fw
    n = fw.n
    if step <= 0 or not fw.history:
        d = np.full((n, n), np.inf)
        from_arcs_d, nxt = alg.initial_matrix(a.net.graph)
        return 0, from_arcs_d, nxt.astype(np.int16), np.zeros((n, n), dtype=bool)
    k, d, nxt, changed = fw.history[min(step, len(fw.history)) - 1]
    return k, d, nxt, changed


def _fmt(x):
    return "∞" if not np.isfinite(x) else f"{x:g}"


def build_matrix(a, step, mark_dijkstra=False, height=None):
    """Die Entfernungsmatrix nach `step` Einträgen des Protokolls. Zeile = von, Spalte = nach; grau = noch keine Route; orange Umrandung = in diesem Schritt verbessert; halbtransparentes Kreuz aus Zeile und Spalte =
    der Zwischenknoten k, der gerade erlaubt wurde; rot umrandete Diagonale = negativer Zyklus. Bei kleinen Netzen stehen die Werte in den Zellen."""
    g = a.net.graph
    n = g.n
    k, d, nxt, changed = state_at(a, step)
    small = n <= 12
    z = np.where(np.isfinite(d), d, np.nan)
    finite = z[np.isfinite(z)]
    lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1
    labels = list(g.names) if g.names else [str(i) for i in range(n)]
    fig = go.Figure(go.Heatmap(z=z, x=labels if small else list(range(n)), y=labels if small else list(range(n)), colorscale="Viridis", zmin=lo, zmax=hi, xgap=1 if small else 0, ygap=1 if small else 0,
                               hovertemplate="von %{y} nach %{x}: %{z:g} " + a.net.unit + "<extra></extra>", colorbar=dict(title=f"[{a.net.unit}]", thickness=12, len=0.6), hoverongaps=False))
    if small:
        fig.update_traces(text=[[_fmt(v) for v in row] for row in d], texttemplate="%{text}", textfont=dict(size=13))
    ci, cj = np.nonzero(changed)
    if len(ci):
        xs = [labels[j] if small else j for j in cj]
        ys = [labels[i] if small else i for i in ci]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="in diesem Schritt verbessert", hoverinfo="skip",
                                 marker=dict(symbol="square-open", size=26 if small else (9 if n <= 100 else 5), color=C.COLORS["goal"], line=dict(width=3 if small else 1.5))))
    if k > 0:
        pos = k - 1
        band = dict(type="rect", xref="x", yref="y", fillcolor="rgba(255,255,255,0.28)", line=dict(width=0), layer="above")
        fig.add_shape(**band, x0=pos - 0.5, x1=pos + 0.5, y0=-0.5, y1=n - 0.5)
        fig.add_shape(**band, x0=-0.5, x1=n - 0.5, y0=pos - 0.5, y1=pos + 0.5)
    diag = np.where(np.diag(d) < 0)[0]
    if len(diag):
        fig.add_trace(go.Scatter(x=[labels[i] if small else i for i in diag], y=[labels[i] if small else i for i in diag], mode="markers", name="negative Diagonale: negativer Zyklus", hoverinfo="skip",
                                 marker=dict(symbol="square-open", size=26 if small else 8, color=C.COLORS["cycle"], line=dict(width=4 if small else 2))))
    if mark_dijkstra and a.dj_dist is not None and step >= len(a.fw.history):
        bad = np.where(~np.isclose(a.dj_dist, d) & ~(np.isinf(a.dj_dist) & np.isinf(d)))
        if len(bad[0]):
            fig.add_trace(go.Scatter(x=[labels[j] if small else j for j in bad[1]], y=[labels[i] if small else i for i in bad[0]], mode="markers", name="n-mal Dijkstra liegt daneben", hoverinfo="skip",
                                     marker=dict(symbol="x", size=12 if small else (7 if n <= 100 else 4), color="white", line=dict(width=2, color=C.COLORS["dijkstra"]))))
    fig.update_yaxes(autorange="reversed", showticklabels=small or n <= 40, title="von")
    fig.update_xaxes(side="top", showticklabels=small or n <= 40, title="nach")
    if small:
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category")
    h = height or (360 if small else 460)
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40 if small else 30, b=10), legend=dict(orientation="h", y=-0.04))
    if not small:
        fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")          # quadratische Zellen: eine n x n Matrix soll wie eine Matrix aussehen
        fig.update_xaxes(constrain="domain")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _segments(g):
    src = np.repeat(np.arange(g.n), g.degree())
    dst = g.indices
    keep = np.ones(len(src), dtype=bool)
    if not g.directed:
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        first = np.zeros(len(src), dtype=bool)
        _, idx = np.unique(lo * g.n + hi, return_index=True)
        first[idx] = True
        keep &= first
    u, v = src[keep], dst[keep]
    x = np.full(3 * len(u), None, dtype=object)
    y = np.full(3 * len(u), None, dtype=object)
    x[0::3], x[1::3] = g.xy[u, 0], g.xy[v, 0]
    y[0::3], y[1::3] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def _shifted(g, u, v, amount=0.11):
    p, q = g.xy[u], g.xy[v]
    d = q - p
    norm = float(np.hypot(*d)) or 1.0
    return p + np.array([d[1], -d[0]]) / norm * amount, q + np.array([d[1], -d[0]]) / norm * amount


def build_network(a, step, height=520):
    """Das Netz nach `step`: die bisher erlaubten Zwischenknoten (Nummern 0 .. k-1) dunkel, die anderen hell, dazu die beste Route des gewählten Paares, die nur über diese Zwischenknoten läuft (rot).
    Negative Kanten grün."""
    net, g = a.net, a.net.graph
    small = bool(g.names)
    k, d, nxt, _ = state_at(a, step)
    s, t = a.s, a.t
    route = alg.route_from(nxt, s, t) if not (a.fw.negative_cycle and step >= len(a.fw.history)) else []
    fig = go.Figure()
    if net.geometric:
        ex, ey = _segments(g)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1), hoverinfo="skip", showlegend=False))
    neg = g.weight < 0
    if small:
        src = np.repeat(np.arange(g.n), g.degree())
        both = {(int(u), int(v)) for u, v in zip(src, g.indices)}
        for u, v, w in zip(src.tolist(), g.indices.tolist(), g.weight.tolist()):
            p, q = _shifted(g, u, v) if (v, u) in both else (g.xy[u], g.xy[v])
            col = "rgba(44,160,44,0.9)" if w < 0 else "rgba(120,120,120,0.55)"
            fig.add_annotation(x=q[0], y=q[1], ax=p[0], ay=p[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.3 if w >= 0 else 2.2,
                               arrowcolor=col, standoff=13, startstandoff=13)
            mid = (p + q) / 2
            fig.add_annotation(x=mid[0], y=mid[1], text=_fmt(w), showarrow=False, font=dict(size=12, color="#1b7f1b" if w < 0 else "#555"), bgcolor="rgba(255,255,255,0.8)")
    elif neg.any() and net.geometric:
        src = np.repeat(np.arange(g.n), g.degree())
        u, v = src[neg], g.indices[neg]
        x = np.full(3 * len(u), None, dtype=object)
        y = np.full(3 * len(u), None, dtype=object)
        x[0::3], x[1::3], y[0::3], y[1::3] = g.xy[u, 0], g.xy[v, 0], g.xy[u, 1], g.xy[v, 1]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="rgba(44,160,44,0.9)", width=2), name="negative Kanten", hoverinfo="skip"))
    if route:
        pts = g.xy[route]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=C.COLORS["fw"], width=5), name=f"beste Route über die Zwischenknoten < {k}" if k else "direkte Kante", hoverinfo="skip"))
    allowed = np.arange(g.n) < k
    size = 15 if small else (3 if g.n > 1500 else (5 if g.n > 300 else 8))
    text = list(g.names) if small else None
    for mask, name, color in ((~allowed, "noch nicht als Zwischenknoten erlaubt", "rgba(200,200,200,0.95)"), (allowed, "als Zwischenknoten erlaubt", "#3b4a8a")):
        idx = np.where(mask)[0]
        if len(idx):
            fig.add_trace(go.Scatter(x=g.xy[idx, 0], y=g.xy[idx, 1], mode="markers+text" if small else "markers", name=name, text=[text[i] for i in idx] if small else None, textposition="top center",
                                     hoverinfo="skip", marker=dict(size=size, color=color, line=dict(color="gray", width=1))))
    for node, name, color, symbol in ((s, "Start", C.COLORS["start"], "diamond"), (t, "Ziel", C.COLORS["goal"], "star")):
        label = g.names[node] if small else f"{node}"
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", name=f"{name}: {label}", hoverinfo="skip", marker=dict(size=15, color=color, symbol=symbol, line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False)
    if not small:
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        pad = 0.16 * (hi - lo)
        fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + pad[0]])
        fig.update_yaxes(range=[lo[1] - pad[1], hi[1] + pad[1] + 0.05 * (hi[1] - lo[1])])
    return _base(fig, height)


def table_rows(a, step):
    """Kleine Netze: je Zeile ein Zwischenknoten-Schritt ist zu lang für eine Tabelle; hier nur die Verbesserungen des Schritts als Text (von, nach, alt, neu)."""
    g = a.net.graph
    k, d, nxt, changed = state_at(a, step)
    prev = state_at(a, step - 1)[1] if step > 1 else alg.initial_matrix(g)[0]
    out = []
    for i, j in zip(*np.nonzero(changed)):
        out.append((g.names[i] if g.names else int(i), g.names[j] if g.names else int(j), float(prev[i, j]), float(d[i, j])))
    return out


def build_effort_degree(rows, height=340):
    x = [r["degree"] for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("fw", "Floyd-Warshall (n³ Vergleiche)", "#d62728", "solid"), ("skip", "Floyd-Warshall, überspringt Unerreichbares", "#ff7f0e", "solid"),
                                   ("bf", "n-mal Bellman-Ford (Kantenprüfungen)", "#2ca02c", "solid"), ("dijkstra", "n-mal Dijkstra (Kantenprüfungen)", "#1f77b4", "dash")):
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.update_layout(xaxis=dict(title="mittlerer Ausgangsgrad (Kanten je Knoten)", type="log"), yaxis=dict(title="Zähler", type="log"))
    return _base(fig, height)


def build_effort_size(rows, height=340):
    x = [r["size"] for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("fw", "Floyd-Warshall (n³)", "#d62728", "solid"), ("skip", "überspringt Unerreichbares", "#ff7f0e", "solid"), ("bf", "n-mal Bellman-Ford", "#2ca02c", "solid"), ("dijkstra", "n-mal Dijkstra", "#1f77b4", "dash")):
        fig.add_trace(go.Scatter(x=x, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.update_layout(xaxis=dict(title="Knoten im Netz", type="log"), yaxis=dict(title="Zähler", type="log"))
    return _base(fig, height)


def build_loop_order(rows, height=300):
    fig = go.Figure(go.Bar(x=[r["order"] for r in rows], y=[r["wrong_share"] * 100 for r in rows], marker_color=["#2ca02c" if r["order"] in ("kij", "kji") else "#d62728" for r in rows],
                           hovertemplate="Reihenfolge %{x}: %{y:.1f} % der Zellen falsch<extra></extra>"))
    fig.update_layout(xaxis_title="Reihenfolge der drei Schleifen (k, i = Zeile, j = Spalte)", yaxis_title="falsche Zellen der Matrix [%]")
    return _base(fig, height)


def build_growth(rows, height=320):
    fig = go.Figure(go.Scatter(x=[r["n"] for r in rows], y=[-r["low_median"] for r in rows], mode="lines+markers", line=dict(color="#d62728"),
                               hovertemplate="%{x} Knoten: kleinster Wert −%{y:.3g}<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Knoten im Netz", type="log"), yaxis=dict(title="Betrag des kleinsten Werts der Matrix (Median)", type="log"))
    return _base(fig, height)


def build_intermediate_order(rows, height=300):
    fig = go.Figure()
    for key, color in (("index", "#7f7f7f"), ("degree_asc", "#2ca02c"), ("degree_desc", "#d62728"), ("random", "#1f77b4")):
        fig.add_trace(go.Bar(x=[{"random": "Zufallsnetz", "city": "Stadtnetz"}[r["net"]] for r in rows], y=[r[key] * 100 for r in rows], name=ORDER_NAMES[key], marker_color=color))
    fig.update_layout(barmode="group", yaxis_title="Vergleiche von \"überspringen\" [% von n³]")
    return _base(fig, height)
