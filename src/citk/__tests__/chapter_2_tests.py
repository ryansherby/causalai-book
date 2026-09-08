"""Tests derived from chapter2.ipynb for SymbolicSCM and CausalGraph tools.

Chapter 2 introduces structural causal models, observational / interventional /
counterfactual queries, causal diagrams, and d-separation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from citk import CausalGraph, SymbolicSCM, variables
from citk.__tests__.helpers import as_name_sets, as_names


@pytest.fixture
def dice_scm():
    """Example 2.1 — two dice combined into X and Y."""
    x, y, u1, u2 = variables("x y u1 u2")
    m1 = SymbolicSCM(
        f={
            x: u1 + u2,
            y: u1 - u2,
        },
        pu={
            u1: [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
            u2: [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
        },
    )
    return m1, x, y, u1, u2


@pytest.fixture
def treatment_scm():
    """Example 2.2 — treatment X, outcome Y, symptom Z."""
    z, x, y, ur, uz, ux, uy = variables("z x y ur uz ux uy")
    m2 = SymbolicSCM(
        f={
            z: ur & uz,
            x: z & ux | ~z & ~ux,
            y: x & ur | ~x & ur & uy | ~x & ~ur & ~uy,
        },
        pu={
            ur: 0.25,
            uz: 0.95,
            ux: 0.9,
            uy: 0.7,
        },
    )
    return m2, z, x, y


def test_example_2_1_scm_variables(dice_scm):
    m1, x, y, u1, u2 = dice_scm
    assert as_names(m1.v) == {"x", "y"}
    assert as_names(m1.u) == {"u1", "u2"}
    assert x in m1.v and y in m1.v
    assert m1.graph is not None


def test_example_2_1_probability_table_includes_exogenous(dice_scm):
    m1, x, y, u1, u2 = dice_scm
    table = m1.get_probability_table(include_u=True)
    assert list(table.columns[:4]) == [u1, u2, x, y] or set(
        table.columns[:4]
    ) >= {u1, u2, x, y}
    first = table.iloc[0]
    assert int(first[u1]) == 1
    assert int(first[u2]) == 1
    assert int(first[x]) == 2
    assert int(first[y]) == 0
    assert first["probability"] == pytest.approx(1 / 36, rel=1e-5)
    assert table["probability"].sum() == pytest.approx(1.0, rel=1e-6)


def test_example_2_1_observational_table(dice_scm):
    m1, x, y, u1, u2 = dice_scm
    table = m1.get_probability_table()
    assert set(table.columns) >= {x, y, "probability"}
    assert table["probability"].sum() == pytest.approx(1.0, rel=1e-6)
    row = table[(table[x] == 2) & (table[y] == 0)]
    assert len(row) == 1
    assert row.probability.iloc[0] == pytest.approx(1 / 36, rel=1e-5)


def test_example_2_1_sample_draws_from_support(dice_scm):
    m1, x, y, u1, u2 = dice_scm
    sample = m1.sample(n=25)
    assert isinstance(sample, pd.DataFrame)
    assert len(sample) == 25
    assert x in sample.columns and y in sample.columns


def test_example_2_2_joint_distribution(treatment_scm):
    m2, z, x, y = treatment_scm
    table = m2.get_probability_table()
    expected = {
        (0, 0, 0): 0.475875,
        (0, 0, 1): 0.210375,
        (0, 1, 0): 0.075000,
        (0, 1, 1): 0.001250,
        (1, 0, 0): 0.007125,
        (1, 0, 1): 0.016625,
        (1, 1, 1): 0.213750,
    }
    assert table["probability"].sum() == pytest.approx(1.0, rel=1e-6)
    for (zv, xv, yv), prob in expected.items():
        match = table[(table[z] == zv) & (table[x] == xv) & (table[y] == yv)]
        assert match.probability.sum() == pytest.approx(prob, rel=1e-5)


def test_example_2_3_conditional_query_matches_table(treatment_scm):
    m2, z, x, y = treatment_scm
    pt = m2.get_probability_table()
    from_table = (
        pt.query("x == 1 and y == 1").probability.sum()
        / pt.query("x == 1").probability.sum()
    )
    from_query = m2.query({y: 1}, given={x: 1})
    assert from_query == pytest.approx(0.7413793103448275, rel=1e-9)
    assert from_table == pytest.approx(from_query, rel=1e-9)
    assert m2.query({y: 1}, given={x: 0}) == pytest.approx(
        m2.query({y: 1, x: 0}) / m2.query({x: 0}), rel=1e-9
    )
    assert m2.query({z: 1}) == pytest.approx(0.007125 + 0.016625 + 0.213750, rel=1e-6)


def test_example_2_4_intervention_does_not_change_independent_outcome(dice_scm):
    m1, x, y, u1, u2 = dice_scm
    observational = m1.query({y: 0})
    intervened = m1.do({x: 2})
    assert as_names(intervened.v) == {"x_{x=2}", "y_{x=2}"}
    x2, y2 = intervened.v
    assert str(y2) == "y_{x=2}"
    assert intervened.query({y: 0}) == pytest.approx(observational, rel=1e-9)
    assert intervened.query({y2: 0}) == pytest.approx(observational, rel=1e-9)
    assert m1.query({y2: 0}) == pytest.approx(observational, rel=1e-9)


def test_example_2_5_tv_and_ate_disagree(treatment_scm):
    m2, z, x, y = treatment_scm
    p_y1_do_x1 = m2.do({x: 1}).query({y: 1})
    p_y1_do_x0 = m2.do({x: 0}).query({y: 1})
    tv = m2.query({y: 1}, given={x: 1}) - m2.query({y: 1}, given={x: 0})
    ate = p_y1_do_x1 - p_y1_do_x0
    assert tv > 0
    assert ate < 0
    # Effectiveness: intervening on X=x forces X=x with probability 1.
    assert m2.do({x: 1}).query({x: 1}) == pytest.approx(1.0, rel=1e-9)
    assert m2.do({x: 0}).query({x: 0}) == pytest.approx(1.0, rel=1e-9)


def test_example_2_6_counterfactual_query(treatment_scm):
    m2, z, x, y = treatment_scm
    z1, x1, y1 = m2.do({x: 1}).v
    assert str(y1) == "y_{x=1}"
    value = m2.query({y1: 1}, given={x: 0, y: 0})
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(
        m2.query({y1: 1, x: 0, y: 0}) / m2.query({x: 0, y: 0}), rel=1e-9
    )


def test_example_2_10_causal_chain_conditional_independence():
    x, y, z, ux, uz, uy = variables("x y z ux uz uy")
    ex2_10 = SymbolicSCM(
        f={
            x: ux,
            z: x | ~uz,
            y: z & uy,
        },
        pu={
            ux: 0.5,
            uz: 0.5,
            uy: 0.5,
        },
    )
    joint = ex2_10.query({y: 1, x: 1})
    marginals = ex2_10.query({y: 1}) * ex2_10.query({x: 1})
    assert joint != pytest.approx(marginals, rel=1e-8)

    cond_joint = ex2_10.query({y: 1, x: 1}, given={z: 1})
    cond_marginals = ex2_10.query({y: 1}, given={z: 1}) * ex2_10.query(
        {x: 1}, given={z: 1}
    )
    assert cond_joint == pytest.approx(cond_marginals, rel=1e-8)


def test_example_2_11_causal_fork_conditional_independence():
    x, y, z, ux, uz, uy = variables("x y z ux uz uy")
    ex2_11 = SymbolicSCM(
        f={
            z: uz,
            x: z | ~ux,
            y: z | uy,
        },
        pu={
            ux: 0.5,
            uz: 0.5,
            uy: 0.5,
        },
    )
    joint = ex2_11.query({y: 1, x: 1})
    marginals = ex2_11.query({y: 1}) * ex2_11.query({x: 1})
    assert joint != pytest.approx(marginals, rel=1e-8)

    cond_joint = ex2_11.query({y: 1, x: 1}, given={z: 1})
    cond_marginals = ex2_11.query({y: 1}, given={z: 1}) * ex2_11.query(
        {x: 1}, given={z: 1}
    )
    assert cond_joint == pytest.approx(cond_marginals, rel=1e-8)


def test_example_2_12_collider_dependence_after_conditioning():
    x, y, z, ux, uy, uz = variables("x y z ux uy uz")
    ex2_12 = SymbolicSCM(
        f={
            x: ux,
            y: uy,
            z: ~y & (~x | uz),
        },
        pu={
            ux: 0.5,
            uy: 0.5,
            uz: 0.5,
        },
    )
    joint = round(ex2_12.query({y: 1, x: 1}), 4)
    marginals = round(ex2_12.query({y: 1}), 4) * round(ex2_12.query({x: 1}), 4)
    assert joint == pytest.approx(marginals, rel=1e-8)

    cond_joint = ex2_12.query({y: 1, x: 1}, given={z: 0})
    cond_marginals = ex2_12.query({y: 1}, given={z: 0}) * ex2_12.query(
        {x: 1}, given={z: 0}
    )
    assert cond_joint != pytest.approx(cond_marginals, rel=1e-8)


def test_example_2_13_descendant_of_collider():
    x, y, z, w, ux, uy, uz, uw = variables("x y z w ux uy uz uw")
    ex2_13 = SymbolicSCM(
        f={
            x: ux,
            y: uy,
            z: ~y & (~x | uz),
            w: z ^ uw,
        },
        pu={
            ux: 0.5,
            uy: 0.5,
            uz: 0.5,
            uw: 0.25,
        },
    )
    joint = round(ex2_13.query({y: 1, x: 0}), 4)
    marginals = round(ex2_13.query({y: 1}), 4) * round(ex2_13.query({x: 0}), 4)
    assert joint == pytest.approx(marginals, rel=1e-8)

    cond_joint = ex2_13.query({y: 1, x: 0}, given={w: 0})
    cond_marginals = ex2_13.query({y: 1}, given={w: 0}) * ex2_13.query(
        {x: 0}, given={w: 0}
    )
    assert cond_joint != pytest.approx(cond_marginals, rel=1e-8)


def test_definition_2_4_3_d_separation_sprinkler_graph():
    C, S, R, W, L = variables("C S R W L")
    graph = CausalGraph(
        [C, S, R, W, L],
        directed_edges=[(C, S), (C, R), (R, W), (S, W), (W, L)],
    )
    assert graph.is_d_separator(x=S, y=W, given=None) is False
    assert graph.find_all_d_separators(x=S, y=W) == []

    assert graph.is_d_separator(x=R, y=L, given={W}) is True
    separators = as_name_sets(graph.find_all_d_separators(x=R, y=L))
    assert separators == {
        frozenset({"W"}),
        frozenset({"W", "S"}),
        frozenset({"W", "C"}),
        frozenset({"W", "S", "C"}),
    }
    minimal = graph.find_minimal_d_separator(x=R, y=L, restricted={W})
    assert as_names(minimal) == {"W"}


def test_d_separation_with_bidirected_confounding():
    t, x, w, z, r, y, s = variables("t x w z r y s")
    graph = CausalGraph(
        [t, x, w, z, r, y, s],
        directed_edges=[
            (t, x),
            (t, z),
            (x, w),
            (w, z),
            (z, y),
            (r, y),
            (y, s),
        ],
        bidirected_edges=[(w, y)],
    )
    assert graph.find_all_d_separators(x=x, y=y) == []
    assert graph.is_d_separator(x, y) is False
    assert as_names(graph.get_parents(w)) == {"x"}
    assert as_names(graph.get_children(z)) == {"y"}
    assert "s" in as_names(graph.get_descendants(w))
    assert "t" in as_names(graph.get_ancestors(z))
