"""Tests derived from chapter5.ipynb for counterfactuals and ctf-calculus.

Chapter 5 covers layer-3 queries, counterfactual ancestors, twin / multi-world
networks, exclusion, consistency, unnesting, and the three rules of
ctf-calculus.
"""

from __future__ import annotations

import pytest
import sympy as sp

from src import CausalGraph, Pr, Summation, SymbolicSCM, variables
from src.__tests__.helpers import as_names, assert_contains_all, intervention_mains


def _scm_5_1():
    x, y, u = variables("X Y U")
    scm = SymbolicSCM(
        f={
            x: sp.Piecewise((0, u < 4), (1, True)),
            y: sp.Piecewise(
                (0, sp.Eq(u, 0) | sp.Eq(u, 4)),
                (x, sp.Eq(u, 1) | sp.Eq(u, 5)),
                (1 - x, sp.Eq(u, 2) | sp.Eq(u, 6)),
                (1, True),
            ),
        },
        pu={u: [1 / 40, 1 / 40, 1 / 80, 3 / 16, 1 / 160, 1 / 160, 11 / 160, 107 / 160]},
    )
    return scm, x, y, u


def test_example_5_1_probability_table_and_conditionals():
    scm, x, y, u = _scm_5_1()
    table = scm.get_probability_table(include_u=True)
    expected_u_rows = {
        0: (0, 0, 0.02500),
        1: (0, 0, 0.02500),
        2: (0, 1, 0.01250),
        3: (0, 1, 0.18750),
        4: (1, 0, 0.00625),
        5: (1, 1, 0.00625),
        6: (1, 0, 0.06875),
        7: (1, 1, 0.66875),
    }
    for u_val, (xv, yv, p) in expected_u_rows.items():
        row = table[table[u] == u_val].iloc[0]
        assert int(row[x]) == xv
        assert int(row[y]) == yv
        assert row["probability"] == pytest.approx(p, rel=1e-5)

    pgx1 = Pr({y: 1}, given={x: 1})
    pgx0 = Pr({y: 1}, given={x: 0})
    assert scm.query_exp(pgx1) == pytest.approx(0.9, rel=1e-8)
    assert scm.query_exp(Pr({y: 1, x: 1}) / Pr({x: 1})) == pytest.approx(0.9, rel=1e-8)
    assert scm.query_exp(pgx0) == pytest.approx(0.8, rel=1e-8)


def test_example_5_1_interventional_and_counterfactual_queries():
    scm, x, y, u = _scm_5_1()
    py1_dx1 = Pr({y: 1}, do={x: 1})
    py1_dx0 = Pr({y: 1}, do={x: 0})
    assert scm.query_exp(py1_dx1) == pytest.approx(0.8875, rel=1e-8)
    assert scm.query_exp(py1_dx0) == pytest.approx(0.9375, rel=1e-8)

    table_do_x1 = scm.do({x: 1}).get_probability_table(include_u=True)
    assert table_do_x1["probability"][table_do_x1[u].isin([1, 3, 5, 7])].sum() == pytest.approx(
        0.8875, rel=1e-8
    )

    y_x0 = scm.do({x: 0}).v["Y"]
    y_x1 = scm.do({x: 1}).v.Y
    assert str(y_x0) == "Y_{X=0}"
    assert str(y_x1) == "Y_{X=1}"

    assert scm.query_exp(Pr({y_x1: 1}, given={x: 0}) > Pr({y_x0: 1}, given={x: 0})) is True
    assert scm.query_exp(Pr({y_x1: 1}, given={x: 1}) < Pr({y_x0: 1}, given={x: 1})) is True
    assert str(Pr({y_x1: 1}, given={x: 0, y: 0})) == (
        r"P\left(Y_{X=1} = 1 \mid X = 0, Y = 0\right)"
    )


def test_example_5_3_effect_of_treatment_on_the_treated():
    x, y, u = variables("x y u")
    scm = SymbolicSCM(
        f={
            x: sp.Piecewise((0, u < 2), (1, True)),
            y: sp.Piecewise(
                (x, sp.Eq(u, 0) | sp.Eq(u, 2)),
                (1 - x, sp.Eq(u, 1) | sp.Eq(u, 3)),
                (1, True),
            ),
        },
        pu={u: [9 / 40, 7 / 40, 1 / 5, 1 / 5, 1 / 5]},
    )
    y_x1 = scm.do({x: 1}).v["y"]
    y_x0 = scm.do({x: 0}).v.y
    assert scm.query_exp(Pr({y_x1: 1}) > Pr({y_x0: 1})) is True
    assert scm.query_exp(Pr({y: 1}, given={x: 1}) > Pr({y_x1: 1})) is True
    ett = scm.query_exp(Pr({y_x1: 1}, given={x: 1}) - Pr({y_x0: 1}, given={x: 1}))
    assert isinstance(ett, float)


def test_example_5_5_tv_ate_and_counterfactuals():
    x, y, z, ur, ux, uy, uz = variables("x y z ur ux uy uz")
    scm = SymbolicSCM(
        f={
            z: ur & uz,
            x: ~(z ^ ux),
            y: (x & ur) | (~x & ur & uy) | (~x & ~ur & ~uy),
        },
        pu={ur: 0.25, uz: 0.95, ux: 0.9, uy: 0.7},
    )
    y_x1 = scm.do({x: 1}).v.y
    y_x0 = scm.do({x: 0}).v.y
    assert scm.query_exp(Pr({y: 1}, given={x: 1}) - Pr({y: 1}, given={x: 0}) > 0) is True
    assert scm.query_exp(Pr({y_x1: 1}) - Pr({y_x0: 1}) > 0) is False
    p_pn = scm.query({y_x1: 1}, given={x: 0, y: 0})
    assert 0.0 <= p_pn <= 1.0
    assert 0.0 <= scm.query_exp(Pr({y_x1: 1}, given={x: 0, y: 1})) <= 1.0
    assert scm.query({y_x1: 1, y_x0: 0}) >= 0.0


def test_example_5_7_nested_counterfactual_and_summation():
    x, w, y, uwx, ux, uw, uy = variables("x w y uwx ux uw uy")
    scm = SymbolicSCM(
        f={
            x: ux | uwx,
            w: (x & uwx) | (~x & uw),
            y: (~x & w) | (w & uy) | ~w,
        },
        pu={ux: 0.6, uwx: 0.3, uw: 0.5, uy: 0.1},
    )
    w_x0 = scm.do({x: 0}).v.w
    w_x1 = scm.do({x: 1}).v.w
    y_x1_wx0 = scm.do({x: 1, w: w_x0}).v.y
    y_x0 = scm.do({x: 0}).v.y
    y_x0_wx1 = scm.do({x: 0, w: w_x1}).v.y

    nested = scm.query_exp(Pr({y_x1_wx0: 1}) - Pr({y_x0: 1}))
    summed = scm.query_exp(
        Summation((Pr({w_x1: w}) - Pr({w_x0: w})) * Pr({y: 1}, {x: 0, w: w}), w)
    )
    nested2 = scm.query_exp(Pr({y_x0_wx1: 1}) - Pr({y_x0: 1}))
    assert isinstance(nested, float)
    assert isinstance(summed, float)
    assert isinstance(nested2, float)


def test_example_5_8_do_with_counterfactual_assignment():
    x, w1, w2, y, ux, uw1, uw2, uy = variables("x w1 w2 y ux uw1 uw2 uy")
    scm = SymbolicSCM(
        f={
            x: ux,
            w1: uw1,
            w2: uw2,
            y: (x & ~w1 & ~w2) | (~x & w1 & ~w2) | (~x & ~w1 & w2) | ~x,
        },
        pu={ux: 0.5, uw1: 0.3, uw2: 0.7, uy: 0.1},
    )
    w_x1 = scm.do({x: 1}).v.w1
    value = scm.query_exp(Pr({y: 1}, do={w1: w_x1}))
    assert 0.0 <= value <= 1.0


def test_example_5_9_and_5_10_counterfactual_ancestors():
    x, w, z, y = variables("x w z y")
    ex_5_9 = CausalGraph(
        v=[x, w, z, y],
        directed_edges=[(x, w), (w, y), (z, x), (z, y)],
    )
    y_xw = ex_5_9.do({x, w}).v.y
    assert as_names(ex_5_9.get_ctf_ancestors(y_xw)) == {"y_{w}", "z"}

    y_z = ex_5_9.do(z).v.y
    assert as_names(ex_5_9.get_ctf_ancestors([y_z])) == {"y_{z}", "w_{z}", "x_{z}"}

    ex_5_10 = CausalGraph(
        v=[x, w, z, y],
        directed_edges=[(x, w), (w, z), (z, y)],
        bidirected_edges=[(x, y)],
    )
    y_w = ex_5_10.do({w}).v.y
    assert as_names(ex_5_10.get_ctf_ancestors(y_w)) == {"y_{w}", "z_{w}"}
    w_xz = ex_5_10.do({x, z}).v.w
    assert as_names(ex_5_10.get_ctf_ancestors(w_xz)) == {"w_{x}"}


def test_ett_bayes_expansion_and_summation_expression():
    x, w, z, y = variables("x w z y")
    ex_5_9 = CausalGraph(
        v=[x, w, z, y],
        directed_edges=[(x, w), (w, y), (z, x), (z, y)],
    )
    y_x = ex_5_9.do({x: 1}).v.y
    ett = Pr({y_x}, given={x: 0})
    assert str(ett) == r"P\left(y_{x=1} \mid x = 0\right)"
    assert str(ett.apply_bayes()) == r"P\left(y_{x=1}, x = 0\right)/P\left(x = 0\right)"
    assert as_names(ex_5_9.get_ctf_ancestors({y_x, x})) == {
        "x",
        "w_{x=1}",
        "z",
        "y_{x=1}",
    }

    expr = Summation(
        Pr({y: 1}, given={w: w, z: z})
        * Pr({w: w}, {z: z})
        * Pr({x: 0}, {z: z})
        * Pr({z}),
        w,
        z,
    )
    assert "sum" in str(expr).lower() or str(expr).startswith(r"\sum_")


def test_figure_5_8_ctf_unnest():
    x, w, y = variables("x w y")
    fig = CausalGraph(v=[x, w, y], directed_edges=[(x, w), (w, y), (x, y)])
    w_x0 = fig.do({x: 0}).v.w
    y_x1_wx0 = fig.do({x: 1, w: w_x0}).v.y
    exp = fig.apply_ctf_unnest(Pr({y_x1_wx0}))
    assert str(exp) == r"\sum_{w}{P\left(y_{x=1,w=w}, w_{x=0} = w\right)}"
    event_vars = list(exp.atoms(Pr))[0].get_event().keys()
    assert as_names(event_vars) == {"y_{x=1,w=w}", "w_{x=0}"}
    anc = fig.get_ctf_ancestors(list(event_vars))
    assert as_names(anc) - as_names(event_vars) == set()

    y_x0 = fig.do({x: 0}).v.y
    assert as_names(fig.get_ctf_ancestors({y_x0})) == {"y_{x=0}", "w_{x=0}"}
    assert as_names(fig.get_ctf_ancestors({y_x0}) - [y_x0]) == {"w_{x=0}"}


def test_example_5_13_consistency_roundtrip():
    w, x, z, y, uw, ux, uz, uy = variables("w x z y uw ux uz uy")
    scm = SymbolicSCM(
        f={
            w: uw,
            x: ux ^ w,
            z: uz,
            y: z ^ x ^ uy,
        },
        pu={uw: 0.7, uz: 0.4, ux: 0.5, uy: 0.3},
    )
    y_x1 = scm.do({x: 1}).v.y
    y_x0 = scm.do({x: 0}).v.y
    assert scm.query_exp(Pr({y_x1: 1, x: 1})) == pytest.approx(
        scm.query_exp(Pr({y: 1, x: 1})), rel=1e-8
    )
    assert scm.query_exp(Pr({y_x0: 1, x: 0})) == pytest.approx(
        scm.query_exp(Pr({y: 1, x: 0})), rel=1e-8
    )

    y_x, w_x, z_x = scm.graph.do({x}).v[["y", "w", "z"]]
    exp = Pr({y_x, w_x, z_x, x})
    consistent = scm.graph.apply_consistency(
        exp, target_var={y_x, w_x, z_x}, intervention_var=x, method="remove"
    )
    assert "x" in str(consistent)

    y_w, x_w = scm.graph.do({w}).v[["y", "x"]]
    added = scm.graph.apply_consistency(
        Pr({y_w, x_w}), target_var={y_w}, intervention_var=x, method="add"
    )
    assert as_names(added.get_event()) == {"y_{w,x}", "x_{w}"}


def test_example_5_13_nested_consistency():
    z, x, y, ux, uz, uy = variables("z x y ux uz uy")
    scm = SymbolicSCM(
        f={z: uz, x: ux ^ z, y: uy ^ x ^ z},
        pu={ux: 0.5, uz: 0.3, uy: 0.7},
    )
    x_z1 = scm.do({z: 1}).v.x
    y_x0 = scm.do({x: 0}).v.y
    y_x_z1 = scm.do({x: x_z1}).v.y
    delta = scm.query_exp(-Pr({y_x_z1: 1, x_z1: 0}) + Pr({y_x0: 1, x_z1: 0}))
    assert delta == pytest.approx(0.0, abs=1e-8)

    exp = Pr({y_x_z1: 1, x_z1: 0})
    res = scm.graph.apply_consistency(
        exp, target_var={y_x_z1}, intervention_var=x_z1, method="remove"
    )
    assert as_names(res.get_event()) == {"y_{x=0}", "x_{z=1}"}
    restored = scm.graph.apply_consistency(
        res, target_var={y_x0}, intervention_var=x_z1, method="add"
    )
    assert any(str(k).startswith("y_{x=") for k in restored.get_event())
    assert "x_{z=1}" in as_names(restored.get_event())


def test_example_5_14_and_5_15_unnest_do_expressions():
    w, x, z, y = variables("w x z y")
    ex_5_14 = CausalGraph(
        v=[w, x, z, y],
        directed_edges=[(w, x), (x, y), (z, y)],
    )
    x_w = ex_5_14.do({w}).v.x
    unnested = ex_5_14.apply_ctf_unnest(Pr({y}, do={x_w}))
    assert str(unnested) == r"\sum_{x}{P\left(y_{x=x}, x_{w} = x\right)}"

    x, w1, w2, y, ux, uw1, uw2, uy = variables("x w1 w2 y ux uw1 uw2 uy")
    scm = SymbolicSCM(
        f={
            x: ux,
            w1: uw1,
            w2: uw2,
            y: (x & ~w1 & ~w2) | (~x & w1 & ~w2) | (~x & ~w1 & w2) | ~x,
        },
        pu={ux: 0.5, uw1: 0.3, uw2: 0.7, uy: 0.1},
    )
    graph = scm.graph
    w1_x = graph.do({x}).v.w1
    w2_w1_x = graph.do({w1_x}).v.w2
    exp = Pr({y}, do={w2_w1_x, x})
    assert_contains_all(str(exp), r"do(", "x", r"w2_{w1=w1_{x}}")
    unnested = str(graph.apply_ctf_unnest(exp))
    assert_contains_all(unnested, r"y_{x,w2=w2}", r"w2_{w1=w1} = w2", r"w1_{x} = w1")


def test_example_5_16_unnest_joint_counterfactuals():
    x, w1, w2, y = variables("x w1 w2 y")
    graph = CausalGraph(
        v=[x, w1, w2, y],
        directed_edges=[(x, w1), (x, w2), (w1, y), (w2, y)],
    )
    w1_x, w2_x = graph.do({x}).v[["w1", "w2"]]
    y_w1_x = graph.do({w1_x}).v.y
    y_w2_x = graph.do({w2_x}).v.y
    exp = Pr({y_w1_x: 1, y_w2_x: 0})
    assert as_names(exp.get_event()) == {"y_{w1=w1_{x}}", "y_{w2=w2_{x}}"}
    unnested = str(graph.apply_ctf_unnest(exp))
    assert_contains_all(
        unnested,
        r"y_{w1=w1} = 1",
        r"y_{w2=w2} = 0",
        r"w1_{x} = w1",
        r"w2_{x} = w2",
    )


def test_example_5_17_exclusion():
    z, x, y, w = variables("z x y w")
    graph = CausalGraph(
        v=[z, x, y, w],
        directed_edges=[(z, x), (x, y), (x, w), (y, w), (z, y)],
    )
    x_zyw = graph.do({z, y, w}).v.x
    y_xz = graph.do({x, z}).v.y
    w_zxy = graph.do({z, x, y}).v.w
    w_zy = graph.do({z, y}).v.w
    excluded_x = graph.apply_exclusion_var(x_zyw)
    excluded_y = graph.apply_exclusion_var(y_xz)
    excluded_w_zxy = graph.apply_exclusion_var(w_zxy)
    excluded_w_zy = graph.apply_exclusion_var(w_zy)
    assert excluded_x.main == "x" and intervention_mains(excluded_x) == {"z"}
    assert excluded_y.main == "y" and intervention_mains(excluded_y) == {"x", "z"}
    assert excluded_w_zxy.main == "w" and intervention_mains(excluded_w_zxy) == {"x", "y"}
    assert excluded_w_zy.main == "w" and intervention_mains(excluded_w_zy) == {"y", "z"}


def test_twin_network_d_separation():
    x, y, z = variables("x y z")
    causal_chain = CausalGraph(v=[x, y, z], directed_edges=[(x, z), (z, y)])
    twn = causal_chain.build_TWN(interventions=x)
    y_x = twn.v.y[1] if isinstance(twn.v.y, list) else twn.v.y
    # Counterfactual and factual copies are stored under the same main name.
    y_copies = twn.v["y"]
    x_copies = twn.v["x"]
    if isinstance(y_copies, list):
        y_fact, y_ctf = y_copies
    else:
        y_fact, y_ctf = y_copies, y_copies
    if isinstance(x_copies, list):
        x_fact, _ = x_copies
    else:
        x_fact = x_copies
    assert twn.is_d_separator(y_ctf, x_fact, given=None) is True

    ex_twn = CausalGraph(
        v=[x, y, z],
        directed_edges=[(x, y), (z, y), (z, x)],
        bidirected_edges=[(x, z)],
    )
    twn_net = ex_twn.build_TWN(interventions={x})
    y_copies = twn_net.v["y"]
    x_copies = twn_net.v["x"]
    y_ctf = y_copies[1] if isinstance(y_copies, list) else y_copies
    x_fact = x_copies[0] if isinstance(x_copies, list) else x_copies
    # In the well-formed TWN, y_x is independent of x given nothing after exclusion.
    # The notebook's "bad" TWN (missing cross-world bidirected structure) is not independent.
    v1 = ex_twn.v
    v2 = ex_twn.do({x}).v
    bad_twn = CausalGraph(
        v=[v1.x, v1.y, v1.z, v2.x, v2.y, v2.z],
        directed_edges=[
            (v1.x, v1.y),
            (v2.x, v2.y),
            (v1.z, v1.y),
            (v2.z, v2.y),
            (v1.z, v1.x),
        ],
        bidirected_edges=[
            (v1.z, v2.z),
            (v1.x, v2.z),
            (v1.x, v1.z),
            (v1.y, v2.y),
        ],
    )
    y_x = v2.y
    assert bad_twn.is_d_separator(y_x, v1.x, given={v1.z}) is False


def test_example_5_18_mwn_and_amwn():
    x, y, w, z = variables("x y w z")
    graph = CausalGraph(
        v=[x, y, w, z],
        directed_edges=[(x, w), (w, y), (z, y), (z, x)],
    )
    mwn = graph.build_MWN({x}, {w, x})
    y_copies = mwn.v["y"]
    w_copies = mwn.v["w"]
    assert len(y_copies) >= 2
    assert len(w_copies) >= 2

    y_xw = graph.do({x, w}).v.y
    w_x = graph.do({x}).v.w
    z_x = graph.do({x}).v.z
    assert str(graph.apply_exclusion_var(y_xw)) == "y_{w}"
    assert intervention_mains(graph.apply_exclusion_var(w_x)) == {"x"}
    assert str(graph.apply_exclusion_var(z_x)) == "z"

    y_w = graph.apply_exclusion_var(y_xw)
    assert mwn.is_d_separator({y_w, w_x}, x, given={z}) is True
    assert as_names(graph.get_ctf_ancestors({y_xw, w_x, x})) == {"x", "y_{w}", "w_{x}", "z"}

    amwn = graph.build_AMWN({y_xw, w_x, x})
    assert as_names(amwn.v) == {"x", "y_{w}", "w_{x}", "z"}


def test_example_5_19_and_5_20_amwn_independence():
    x, w, t, y = variables("x w t y")
    graph = CausalGraph(v=[x, w, t, y], directed_edges=[(x, w), (w, y), (w, t)])
    y_x = graph.do({x}).v.y
    amwn = graph.build_AMWN({y_x, x})
    assert amwn.is_d_separator(x, y_x) is True

    t_w = graph.do({w}).v.t
    t_x = graph.do({x}).v.t
    amwn2 = graph.build_AMWN({t_w, t_x, y_x})
    assert amwn2.is_d_separator(t_w, y_x, given={t_x}) is False

    x, y, z = variables("x y z")
    ex_5_20 = CausalGraph(
        v=[x, y, z],
        directed_edges=[(x, z), (z, y)],
        bidirected_edges=[(x, y)],
    )
    z_x = ex_5_20.do({x}).v.z
    amwn = ex_5_20.build_AMWN({z_x, x, y})
    assert amwn.is_d_separator(x, z_x, given={y}) is False
    amwn2 = ex_5_20.build_AMWN({z_x, z, y})
    assert amwn2.is_d_separator(y, z_x, given={z}) is False
    y_x = ex_5_20.do({x}).v.y
    amwn3 = ex_5_20.build_AMWN({z, x, y_x})
    assert amwn3.is_d_separator(x, y_x, given={z}) is False


def test_example_5_21_ctf_d_separation():
    x, y, z, w = variables("x y z w")
    graph = CausalGraph(
        v=[x, y, z, w],
        directed_edges=[(w, z), (z, y), (z, x), (x, y)],
        bidirected_edges=[(x, z)],
    )
    y_xw = graph.do({x, w}).v.y
    assert graph.is_ctf_d_separator(x, y_xw, given={z, w}) is False


def test_example_5_22_ctf_calculus_rules():
    x, y, z = variables("x y z")
    graph = CausalGraph(v=[x, y, z], directed_edges=[(x, y), (z, y), (z, x)])
    y_x = graph.do({x: 1}).v.y
    exp = Pr({y_x}, {x: 0})
    assert str(exp) == r"P\left(y_{x=1} \mid x = 0\right)"

    exp = Summation(Pr({y_x}, {x: 0, z: z}) * Pr({z: z}, {x: 0}), z)
    tgt = [pr for pr in exp.atoms(Pr) if y_x in pr.get_event()][0]
    sub = tgt.apply_bayes()
    num = [pr for pr in sub.atoms(Pr) if y_x in pr.get_event()][0]
    denom = [pr for pr in sub.atoms(Pr) if y_x not in pr.get_event()][0]
    new_num = graph.apply_r3(num, target_var={z}, intervention_var={x: 1}, method="add")
    new_denom = graph.apply_r3(denom, target_var={z}, intervention_var={x: 1}, method="add")
    z_x = new_denom.vars.z
    sub = sub.subs({num: new_num, denom: new_denom})
    exp = exp.subs({tgt: sub})
    assert_contains_all(
        str(exp),
        r"y_{x=1}",
        r"x = 0",
        r"z_{x=1} = z",
        r"z = z \mid x = 0",
    )

    tgt = [pr for pr in exp.atoms(Pr) if y_x in pr.get_event()][0]
    r1 = graph.apply_r1(tgt, target_var={y_x}, intervention_var=z, method="add")
    y_xz = r1.vars.y
    exp = exp.subs({tgt: r1})
    assert "y_{x=1,z=z}" in str(exp)

    tgt1 = [pr for pr in exp.atoms(Pr) if y_xz in pr.get_event()][0]
    tgt2 = [
        pr
        for pr in exp.atoms(Pr)
        if (x in pr.get_event() and z_x in pr.get_event() and y_xz not in pr.get_event())
    ][0]
    tgt1_r3 = graph.apply_r3(tgt1, target_var={z_x}, intervention_var=x, method="remove")
    tgt2_r3 = graph.apply_r3(tgt2, target_var={z_x}, intervention_var=x, method="remove")
    exp = exp.subs({tgt1: tgt1_r3, tgt2: tgt2_r3})

    num = [pr for pr in exp.atoms(Pr) if y_xz in pr.get_event()][0]
    denom = [
        pr
        for pr in exp.atoms(Pr)
        if (z in pr.get_event() and x in pr.get_event() and y_xz not in pr.get_event())
    ][0]
    combined = Pr.apply_bayes_inverse(num / denom)
    exp = exp.subs({num: combined, denom: 1})

    tgt = [pr for pr in exp.atoms(Pr) if y_xz in pr.get_event()][0]
    tgt_r2 = graph.apply_r2(tgt, target_var={x}, method="remove")
    tgt_r2 = graph.apply_r2(tgt_r2, target_var={x: 1}, method="add")
    exp = exp.subs({tgt: tgt_r2})

    tgt = [pr for pr in exp.atoms(Pr) if y_xz in pr.get_event()][0]
    tgt_bayes = tgt.apply_bayes()
    num = [pr for pr in tgt_bayes.atoms(Pr) if y_xz in pr.get_event()][0]
    denom = [pr for pr in tgt_bayes.atoms(Pr) if y_xz not in pr.get_event()][0]
    num_r1 = graph.apply_r1(num, target_var={y_xz}, intervention_var={x: 1, z: z}, method="remove")
    tgt_rev = Pr.apply_bayes_inverse(num_r1 / denom)
    exp = exp.subs({tgt: tgt_rev})
    assert_contains_all(
        str(exp),
        r"z = z \mid x = 0",
        r"y \mid z = z, x = 1",
    )


def test_example_5_23_ctf_calculus_identifies_observational_expression():
    x, y, z = variables("x y z")
    graph = CausalGraph(
        v=[x, y, z],
        directed_edges=[(x, z), (z, y)],
        bidirected_edges=[(x, y)],
    )
    y_x = graph.do({x: 1}).v.y
    z_x = graph.do({x: 1}).v.z
    exp = Summation(Pr({y_x}, {x: 0, z_x: z}) * Pr({z_x: z}, {x: 0}), z)

    tgt = [pr for pr in exp.atoms(Pr) if y_x in pr.get_event()][0]
    tgt_bayes = tgt.apply_bayes()
    exp = exp.subs({tgt: tgt_bayes})
    num = [pr for pr in exp.atoms(Pr) if y_x in pr.get_event()][0]
    num_r1 = graph.apply_r1(num, target_var={y_x}, intervention_var={z}, method="add")
    y_xz_x = num_r1.vars.y
    exp = exp.subs({num: num_r1})

    tgt = [pr for pr in exp.atoms(Pr) if y_xz_x in pr.get_event()][0]
    tgt_r3 = graph.apply_r3(tgt, target_var={y_xz_x}, intervention_var={x}, method="remove")
    y_z = tgt_r3.vars.y
    exp = exp.subs({tgt: tgt_r3})

    num = [pr for pr in exp.atoms(Pr) if y_z in pr.get_event()][0]
    denom = [pr for pr in exp.atoms(Pr) if (x in pr.get_event() and y_z not in pr.get_event())][0]
    tgt_bayes = Pr.apply_bayes_inverse(num / denom)
    exp = exp.subs({num: tgt_bayes, denom: 1})
    tgt_r2 = graph.apply_r2(tgt_bayes, target_var={z_x}, method="remove")
    exp = exp.subs({tgt_bayes: tgt_r2})

    z_x0 = graph.do({x: 0}).v.z
    tgt = [pr for pr in exp.atoms(Pr) if y_z in pr.get_event()][0]
    tgt_r2 = graph.apply_r2(tgt, target_var={z_x0: z}, method="add")
    exp = exp.subs({tgt: tgt_r2})

    tgt = [pr for pr in exp.atoms(Pr) if y_z in pr.get_event()][0]
    tgt_bayes = tgt.apply_bayes()
    num = [pr for pr in tgt_bayes.atoms(Pr) if y_z in pr.get_event()][0]
    denom = [pr for pr in tgt_bayes.atoms(Pr) if y_z not in pr.get_event()][0]
    num_r1 = graph.apply_r1(num, target_var={z_x0}, intervention_var={x}, method="remove")
    denom_r1 = graph.apply_r1(denom, target_var={z_x0}, intervention_var={x}, method="remove")
    exp = exp.subs({tgt: tgt_bayes})
    exp = exp.subs({num: num_r1, denom: denom_r1})

    tgt = [pr for pr in exp.atoms(Pr) if y_z in pr.get_event()][0]
    r1 = graph.apply_r1(tgt, target_var={y_z}, intervention_var={z}, method="remove")
    exp = exp.subs({tgt: r1})

    tgt = [pr for pr in exp.atoms(Pr) if z_x in pr.get_event()][0]
    r2 = graph.apply_r2(tgt, target_var={x}, method="remove")
    r2 = graph.apply_r2(r2, target_var={x: 1}, method="add")
    exp = exp.subs({tgt: r2})

    tgt = [pr for pr in exp.atoms(Pr) if z_x in pr.get_event()][0]
    tgt_bayes = tgt.apply_bayes()
    num = [pr for pr in tgt_bayes.atoms(Pr) if z_x in pr.get_event()][0]
    denom = [pr for pr in tgt_bayes.atoms(Pr) if z_x not in pr.get_event()][0]
    r1 = graph.apply_r1(num, target_var={z_x}, intervention_var={x}, method="remove")
    exp = exp.subs({tgt: r1 / denom})

    tgt1_num = [pr for pr in exp.atoms(Pr) if y in pr.get_event()][0]
    tgt1_denom = [pr for pr in exp.atoms(Pr) if (pr.get_event()[x] == 0 and y not in pr.get_event())][0]
    tgt2_num = [pr for pr in exp.atoms(Pr) if z in pr.get_event() and pr.get_event()[x] == 1][0]
    tgt2_denom = [pr for pr in exp.atoms(Pr) if (pr.get_event()[x] == 1 and z not in pr.get_event())][0]
    tgt1_bayes = Pr.apply_bayes_inverse(tgt1_num / tgt1_denom)
    tgt2_bayes = Pr.apply_bayes_inverse(tgt2_num / tgt2_denom)
    exp = exp.subs({tgt1_num: tgt1_bayes, tgt1_denom: 1})
    exp = exp.subs({tgt2_num: tgt2_bayes, tgt2_denom: 1})
    assert_contains_all(
        str(exp),
        r"y \mid x = 0, z = z",
        r"z = z \mid x = 1",
    )
