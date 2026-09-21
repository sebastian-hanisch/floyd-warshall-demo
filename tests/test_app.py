"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Varianten, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Abspielen, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import fw_constants as C
from fw_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
# Urteil je Preset: success = n-mal Dijkstra liegt daneben, Floyd-Warshall nicht; warning = negativer Zyklus; info = kein Unterschied (gemessen, siehe test_claims)
EXPECTED_KIND = {"🔀 Kleines Netz": "success", "🔁 Negativer Zyklus": "warning", "🏙️ Stadtnetz": "info", "🔋 E-Lieferwagen": "success", "🕸️ Zufallsnetz": "info"}


def _run(setup=None, timeout=600, net=None):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    if net:
        at.query_params["net"] = net
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input)}


def _kinds(at):
    return {"success": len(at.success), "warning": len(at.warning), "info": len(at.info)}


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def test_default_renders_without_exception():
    at = _run()
    assert any("Floyd-Warshall in Aktion" in m.value for m in at.markdown)
    assert _kinds(at) == {"success": 1, "warning": 0, "info": 0}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    want = {"success": 0, "warning": 0, "info": 0}
    want[EXPECTED_KIND[name]] = 1
    assert _kinds(at) == want


@pytest.mark.parametrize("variant", list(C.VARIANT_LABELS))
@pytest.mark.parametrize("net", ("small", "small_cycle", "city", "ev"))
def test_every_variant_renders_on_small_and_generated_nets(variant, net):
    def setup(at):
        at.session_state["net_select"] = net
        at.session_state["variant_select"] = variant
        at.session_state["side_slider"] = 6
    at = _run(setup)
    assert len(at.success) + len(at.warning) + len(at.info) == 1


def test_python_loops_are_only_offered_for_small_nets_and_a_chosen_one_falls_back():
    small = _run()
    assert list(small.sidebar.selectbox(key="variant_select").options) == list(C.VARIANT_LABELS.values())
    big = _run(net="random", setup=lambda a: a.session_state.__setitem__("nodes_slider", 200))
    assert list(big.sidebar.selectbox(key="variant_select").options) == [C.VARIANT_LABELS["numpy"]]
    at = _run(lambda a: a.session_state.__setitem__("variant_select", "classic"))
    at.session_state["net_select"] = "random"
    at.session_state["nodes_slider"] = 300
    at.run()
    assert not at.exception and at.selectbox(key="variant_select").value == C.DEFAULT_VARIANT


def test_extreme_settings_render():
    def small(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MIN

    def big_ev(at):
        at.session_state["net_select"] = "ev"
        at.session_state["side_slider"] = C.SIDE_MAX
        at.session_state["hill_slider"] = C.HILL_MAX
        at.session_state["eta_slider"] = C.ETA_MAX

    def dense(at):
        at.session_state["net_select"] = "random"
        at.session_state["nodes_slider"] = 200
        at.session_state["degree_slider"] = C.DEGREE_MAX
        at.session_state["pot_slider"] = C.POT_MAX
    for setup in (small, big_ev, dense):
        at = _run(setup)
        assert at.slider(key="fw_step").value == at.slider(key="fw_step").max


def test_hidden_controls_follow_the_net():
    small, small_c, city, ev, rnd = (_labels(_run(net=n)) for n in ("small", "small_cycle", "city", "ev", "random"))
    assert small == small_c == {"Netz", "Variante"}                                                    # feste Aufgabe: kein Paar, kein Seed
    common = {"Netz", "Variante", "Entfernung Start–Ziel [%]", "Zufalls-Seed"}
    assert city == common | {"Kreuzungen je Seite", "Reichweite der Straßen [Blocklängen]", "Streuung der Kosten"}
    assert ev == common | {"Kreuzungen je Seite", "Hügel [m Höhenunterschied]", "Rückgewinnung bergab [%]"}
    assert rnd == common | {"Knoten", "Mittlerer Grad", "Potenzialspanne"}


def test_hidden_slider_values_come_back_when_the_net_is_shown_again():
    # Die erste Sicht muss das Netz mit dem Regler sein: AppTest verliert den Wert, wenn der Regler zuerst ausgeblendet war (im echten Browser bleibt er erhalten).
    at = _run(net="ev")
    at.session_state["hill_slider"] = 12
    at.run()
    at.session_state["net_select"] = "small"
    at.run()
    at.session_state["net_select"] = "ev"
    at.run()
    assert not at.exception and at.slider(key="hill_slider").value == 12


def test_step_slider_returns_to_the_last_step_when_anything_changes():
    at = _run(net="city")
    at.slider(key="fw_step").set_value(5)
    at.run()
    assert at.slider(key="fw_step").value == 5
    at.session_state["variant_select"] = "skip"
    at.run()
    assert not at.exception and at.slider(key="fw_step").value == at.slider(key="fw_step").max


def test_every_step_of_the_small_nets_renders():
    for net in ("small", "small_cycle"):
        at = _run(net=net)
        for k in range(0, int(at.slider(key="fw_step").max) + 1):
            at.slider(key="fw_step").set_value(k)
            at.run()
            assert not at.exception, (net, k)


def test_play_renders_several_frames_without_duplicate_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    for setup in (lambda a: None, lambda a: a.session_state.__setitem__("net_select", "city")):
        at = _run(setup)
        _play(at)
        assert not at.exception, [e.value for e in at.exception]
    at = _run(net="city")
    at.session_state["side_slider"] = 12                     # 144 Knoten: Schrittweite 3
    at.run()
    _play(at)
    assert not at.exception


def test_dijkstra_mark_checkbox_is_only_shown_with_negative_edges_and_no_cycle():
    assert _run().checkbox(key="mark_dj").value
    assert not _run(net="small_cycle").checkbox and not _run(net="city").checkbox
    at = _run(net="ev")
    at.checkbox(key="mark_dj").set_value(False)
    at.run()
    assert not at.exception


def test_permalink_parameters_select_the_net_and_are_clamped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "random"
    at.query_params["nodes"] = "999999"
    at.query_params["var"] = "gpu"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == "random"
    assert at.slider(key="nodes_slider").value == C.NODES_MAX and at.selectbox(key="variant_select").value == C.DEFAULT_VARIANT


def test_unknown_net_in_the_permalink_falls_back_to_the_default():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET


def test_experiments_run_on_demand():
    at = _run()
    assert not any("Mittel über 5 Netze" in c.value for c in at.caption)
    for key in ("loop_start", "effort_start", "growth_start", "korder_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
    text = " ".join(c.value for c in at.caption)
    for needle in ("Nur wenn **k außen** steht", "Floyd-Warshall braucht immer n³ Vergleiche", "wachsen exponentiell", "Werden Knoten mit **wenigen Nachbarn zuerst** zugelassen"):
        assert needle in text, needle


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    # Matrix und Netz stehen in der Play-Schleife: ihre Schlüssel tragen den Schritt (f"..._{current}")
    assert sorted(keys) == sorted(["matrix_chart", "net_chart", "loop_chart", "effort_degree_chart", "effort_size_chart", "growth_chart", "korder_chart"]), keys
    assert all('_{current}"' in c for c in calls if 'key=f"matrix_chart' in c or 'key=f"net_chart' in c) and sum('key=f"' in c for c in calls) == 2
    viz = (ROOT / "fw_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("_base(fig") >= 5


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))
