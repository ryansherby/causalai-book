"""Tests derived from chapter4.ipynb for backdoor and frontdoor adjustment.

Chapter 4 covers identification of causal effects using adjustment criteria
on causal diagrams.
"""

from __future__ import annotations

from IPython.display import Latex
from sympy import Function

from src import CausalGraph, SymbolicSCM, variables
from src.__tests__.helpers import as_name_sets, as_name_triples, as_names


def test_example_4_1_sprinkler_backdoor_adjustment():
    v = se, sp, ra, we, sl = variables("Se Sp Ra We Sl")
    f_se, f_sp, f_ra, f_we, f_sl = [Function(f"f_{{{k}}}") for k in v]
    u_se, u_sp, u_ra, u_we, u_sl = variables("U_{se} U_{sp} U_{ra} U_{we} U_{sl}")

    m = SymbolicSCM(
        f={
            se: f_se(u_se),
            sp: f_sp(se, u_sp),
            ra: f_ra(se, u_ra),
            we: f_we(sp, ra, u_we),
            sl: f_sl(we, u_sl),
        },
        pu={
            u_se: 0.5,
            u_sp: 0.5,
            u_ra: 0.5,
            u_we: 0.5,
            u_sl: 0.5,
        },
    )

    assert m.graph.is_backdoor_adjustment(x=[sp], y=[we], z=[ra]) is True
    formula = m.graph.is_backdoor_adjustment(x=[sp], y=[we], z=[ra], latex=True)
    assert isinstance(formula, Latex)
    assert "do(" in formula.data

    found = m.graph.find_backdoor_adjustment(
        x=[sp],
        y=[we],
        included=[ra],
    )
    assert ra in found or "Ra" in as_names(found)

    latex_found = m.graph.find_backdoor_adjustment(
        x=[sp],
        y=[we],
        included=[ra],
        latex=True,
    )
    assert isinstance(latex_found, Latex)


def test_example_4_4_backdoor_set_and_formula():
    x, y, z1, z2 = variables("X Y Z1 Z2")
    ex4_4 = CausalGraph(
        v=[x, y, z1, z2],
        directed_edges=[
            (x, y),
            (z1, x),
            (z2, y),
            (z2, z1),
        ],
        bidirected_edges=[
            (x, z1),
            (z2, z1),
        ],
    )

    assert ex4_4.is_backdoor_adjustment(x=[x], y=[y], z=[z1, z2]) is True
    formula = ex4_4.get_backdoor_adjustment_formula(
        x=[x],
        y=[y],
        z=[z2],
        given=[z1],
    )
    assert isinstance(formula, Latex)
    assert "do(" in formula.data
    assert "Z1" in formula.data and "Z2" in formula.data


def test_figure_4_13_proper_causal_paths():
    x1, x2, w1, w2, w3, y = variables("X1 X2 W1 W2 W3 Y")
    fig4_13 = CausalGraph(
        v=[x1, x2, w1, w2, w3, y],
        directed_edges=[
            (x1, w1),
            (x2, w2),
            (w1, x2),
            (w2, y),
            (x1, w3),
            (w3, y),
        ],
        bidirected_edges=[
            (x1, y),
        ],
    )
    paths = fig4_13.find_all_proper_causal_paths(
        x=[x1, x2],
        y=[y],
        full_path=True,
    )
    path_names = [tuple(str(node) for node in path) for path in paths]
    assert ("X1", "W3", "Y") in path_names
    assert ("X2", "W2", "Y") in path_names
    assert len(path_names) == 2


def test_figure_4_14_all_backdoor_adjustments():
    x1, x2, y, z1, z2, z3 = variables("X1 X2 Y Z1 Z2 Z3")
    fig4_14 = CausalGraph(
        v=[x1, x2, y, z1, z2, z3],
        directed_edges=[
            (x1, z1),
            (z1, z2),
            (z2, z3),
            (z3, x2),
            (x2, y),
            (x1, y),
        ],
        bidirected_edges=[
            (z2, y),
            (z3, y),
        ],
    )
    adjustments = fig4_14.find_all_backdoor_adjustments(x={x1, x2}, y=y)
    assert as_name_sets(adjustments) == {
        frozenset({"Z1", "Z3"}),
        frozenset({"Z1", "Z2", "Z3"}),
    }
    formula = fig4_14.get_backdoor_adjustment_formula(
        x=[x1, x2],
        y=y,
        z=[z1, z3],
    )
    assert isinstance(formula, Latex)
    assert "do(" in formula.data


def test_figure_4_24_frontdoor_when_backdoor_fails():
    x, z, y, w1, w2, w3 = variables("X Z Y W1 W2 W3")
    fig4_24 = CausalGraph(
        v=[x, z, y, w1, w2, w3],
        directed_edges=[
            (x, z),
            (z, y),
            (w1, y),
            (w1, x),
            (w2, x),
            (w2, z),
            (w3, z),
            (w3, y),
        ],
        bidirected_edges=[
            (x, y),
        ],
    )
    assert fig4_24.find_all_backdoor_adjustments(x={x}, y=y) is None

    frontdoor = fig4_24.find_all_frontdoor_adjustments(x={x}, y=y)
    assert as_name_triples(frontdoor) == {
        (frozenset({"Z"}), frozenset({"W2"}), frozenset({"W3"})),
        (frozenset({"Z"}), frozenset({"W2"}), frozenset({"W3", "W1"})),
        (frozenset({"Z"}), frozenset({"W1", "W2"}), frozenset({"W3"})),
    }

    formula = fig4_24.get_frontdoor_adjustment_formula(
        x={x},
        y=y,
        z=[z],
        xz=[w2],
        zy=[w3],
    )
    assert isinstance(formula, Latex)
    assert "do(" in formula.data
    assert fig4_24.is_frontdoor_adjustment(
        x={x}, y=y, z=[z], xz=[w2], zy=[w3]
    ) is True
