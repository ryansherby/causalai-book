# Causal Inference Toolkit (CITK)

A Python-native library for building causally intelligent systems: structural
causal models, causal diagrams, adjustment criteria, counterfactual queries,
and ctf-calculus.

**Authors.** Ryan Sherby (`ryan.sherby@columbia.edu`) and the
[Columbia Causal AI Lab](https://causalai.net/) (director: Elias Bareinboim).
Copyright © 2025 Columbia Causal AI Lab. See [LICENSE](citk/LICENSE).

## Installation

Install from PyPI (when published):

```bash
pip install citk
```

Install the current repository (recommended while the package is in development):

```bash
pip install -e "./citk[test]"
```

To run the bundled chapter tests after installing:

```bash
citk-test
```

To render graph visualizations, the Graphviz binaries are required in addition
to the Python `graphviz` package. Install them from
[graphviz.org/download](https://graphviz.org/download/) or with your preferred
package manager.

Optional extras:

| Extra | Install | Purpose |
| ----- | ------- | ------- |
| `test` | `pip install "citk[test]"` | pytest plus the `citk-test` runner |
| `dev` | `pip install "citk[dev]"` | tests plus packaging tools (`build`, `twine`, `hatch`) |

## Quick start

```python
from citk import CausalGraph, Pr, SymbolicSCM, variables

x, y, z, ux, uy, uz = variables("x y z ux uy uz")

scm = SymbolicSCM(
    f={x: ux, z: x | ~uz, y: z & uy},
    pu={ux: 0.5, uz: 0.5, uy: 0.5},
)

scm.query({y: 1}, given={x: 1})
scm.do({x: 1}).query({y: 1})

graph = CausalGraph(
    v=[x, y, z],
    directed_edges=[(x, z), (z, y)],
)
graph.is_d_separator(x, y, given={z})
```

The worked examples in `chapter2.ipynb`, `chapter4.ipynb`, and `chapter5.ipynb`
correspond to the tests in `src/citk/__tests__/chapter_n_tests.py`.

## Documentation

The public API is exported from `citk`:

```python
from citk import (
    CausalGraph,
    Pr,
    Summation,
    SymbolicSCM,
    SymbolContainer,
    Variable,
    variables,
)
```

### `variables(names)`

Create one or more `Variable` instances from a space-separated name string.

**Parameters**

- **names** (`str`): Variable names separated by spaces, for example `"x y z"`.

**Returns**

- A single `Variable` if one name is given, otherwise a list of `Variable`s.

### `Variable`

A `sympy.Symbol` that can carry intervention subscripts (for example `y_{x=1}`).

**Constructor parameters**

- **main** (`str`): Base name of the variable.
- **interventions** (`dict` or `set`, optional): Mapping or set of interventions.
  Defaults to no interventions.

**Methods**

- **`update_interventions(interventions)`**
  - **interventions** (`dict` or `set`): Interventions to add.
  - **Returns:** a new `Variable` with the combined intervention subscript.
- **`remove_interventions(interventions)`**
  - **interventions** (`set` of `Variable`): Interventions to drop.
  - **Returns:** a new `Variable` with those interventions removed.

### `SymbolicSCM`

A discrete structural causal model \(\langle U, V, F, P(U)\rangle\).

**Constructor parameters**

- **f** (`dict[Variable, Expr]`): Structural assignments, one per endogenous
  variable. Equations must be given in topological order.
- **pu** (`dict[Variable, list | float]`): Exogenous distributions. A `float` in
  `[0, 1]` is treated as Bernoulli; a list of probabilities is treated as
  categorical (must sum to 1).
- **precision** (`int`, default `4`): Digits used when displaying results.

**Attributes**

- **v**, **u**: `SymbolContainer` of endogenous and exogenous variables.
- **f**, **pu**: Structural equations and exogenous distributions.
- **graph**: The `CausalGraph` induced by the SCM.
- **syn**: Mapping from base variables to their current (possibly
  counterfactual) versions.

**Methods**

- **`get_probability_table(symbols=None, include_u=False)`**
  - **symbols** (`list[Variable]`, optional): Columns to keep. Defaults to all
    endogenous variables.
  - **include_u** (`bool`, default `False`): If true, also include exogenous
    variables.
  - **Returns:** `pandas.DataFrame` with a `probability` column.
- **`sample(n=1)`**
  - **n** (`int`, default `1`): Number of draws with replacement, weighted by
    the joint distribution.
  - **Returns:** `pandas.DataFrame` of samples.
- **`query(x, given=None, latex=False)`**
  - **x** (`dict[Variable, int]`): Event, for example `{y: 1}`.
  - **given** (`dict[Variable, int]`, optional): Conditioning assignment.
  - **latex** (`bool`, default `False`): If true, return an IPython `Latex`
    object instead of a float.
  - **Returns:** probability of `x` given `given`, or a `Latex` display object.
- **`query_exp(expr, latex=False)`**
  - **expr** (`sympy` expression): Combination of `Pr` / `Summation` objects,
    arithmetic, or a relational expression such as `Pr(...) > Pr(...)`.
  - **latex** (`bool`, default `False`): If true, return a `Latex` object.
  - **Returns:** `float`, `bool` (for relations), or `Latex`.
- **`do(x)`**
  - **x** (`dict[Variable, Expr | int | Variable]`): Intervention. Each key is
    replaced by the given value in a new submodel.
  - **Returns:** a new `SymbolicSCM` whose endogenous variables carry
    intervention subscripts. The original SCM remembers the mapping so
    counterfactual symbols can be queried on either object.

### `CausalGraph`

A causal diagram with directed edges \(X \rightarrow Y\) and bidirected edges
\(X \leftrightarrow Y\) (unobserved confounding). `CausalGraph` combines
d-separation, adjustment, do-calculus / ctf-calculus, accessors, counterfactual
networks, and Graphviz display.

**Constructor parameters**

- **v** (`list[Variable]`, optional): Endogenous nodes. Defaults to `[]`.
- **directed_edges** (`list[tuple[Variable, Variable]]`, optional): Directed
  edges. Must form a DAG with no self-loops.
- **bidirected_edges** (`list[tuple[Variable, Variable]]`, optional):
  Bidirected edges. No self-loops.

**Class methods**

- **`from_scm(scm)`**
  - **scm** (`SymbolicSCM`): Model whose assignments and shared exogenous
    parents define directed and bidirected edges.
  - **Returns:** `CausalGraph`.

**Properties**

- **de_graph**: `networkx.DiGraph` of directed edges.
- **be_graph**: `networkx.MultiGraph` of bidirected edges.
- **combined_graph**: Directed graph that replaces bidirected edges with
  explicit exogenous parents.
- **v**, **u**: Endogenous and exogenous `SymbolContainer`s.
- **syn**: Mapping from a base variable to its current graph copy.
- **cc**: Connected components of the bidirected graph.

**Graph operators**

- **`do(x)`**
  - **x** (`Variable`, set, or dict): Nodes to intervene on. Incoming directed
    edges to `x` are removed; variable names are updated with intervention
    subscripts.
  - **Returns:** a new `CausalGraph`.
- **`__and__(other)` / `__or__(other)`**: Intersection / union of edge sets.
  Variable sets must match.
- **`draw(node_positions=None, include_u=False)`**
  - **node_positions** (`dict[Variable, tuple[int, int]]`, optional): Fixed
    Graphviz positions.
  - **include_u** (`bool`, default `False`): Draw exogenous nodes from the
    combined graph.
  - **Returns:** an IPython display of the Graphviz figure.

**Accessors**

Shared parameters unless noted: **node** is a variable or collection of
variables; **include_self** (`bool`, default `False`) keeps the query nodes;
**as_graph** (`bool`, default `False`) returns a subgraph instead of a
`SymbolContainer`.

- **`get_parents(node, include_self=False)`**
- **`get_children(node, include_self=False)`**
- **`get_ancestors(node, include_self=False, as_graph=False)`**
- **`get_descendants(node, include_self=False, as_graph=False)`**
- **`get_neighbors(node, include_self=False, as_graph=False)`**: Bidirected
  neighbors.
- **`get_connected_components(node, as_graph=False)`**: Bidirected components
  that intersect `node`.
- **`get_nodes(node, as_graph=False)`**: Resolve `node` through `syn`.
- **`get_ctf_ancestors(node)`**, **`get_ctf_descendants(node)`**,
  **`get_ctf_parents(node)`**: Counterfactual-aware relatives. Irrelevant
  interventions are dropped when they do not lie on a path to the query node.

**D-separation**

- **`is_d_separator(x, y, given=None)`**
  - **x**, **y**: Variable or collection on each side of the query.
  - **given**: Conditioning set; defaults to empty.
  - **Returns:** `True` if \(x \perp y \mid given\) in the combined graph.
- **`is_minimal_d_separator(x, y, given=None)`**: Whether `given` is a minimal
  d-separator of `x` and `y`.
- **`find_minimal_d_separator(x, y, included=None, restricted=None)`**
  - **included**: Variables that must appear in the separator.
  - **restricted**: Candidate pool; defaults to \(V \setminus \{x, y\}\).
  - **Returns:** a `SymbolContainer`, or `None`.
- **`find_all_d_separators(x, y, included=None, restricted=None)`**: All
  valid separators as a list of `SymbolContainer`s.
- **`find_all_proper_causal_paths(x, y, full_path=True)`**: Directed paths from
  `x` to `y` that do not pass through other members of `x` or `y`. If
  `full_path` is false, only the first edge of each path is returned.
- **`is_ctf_d_separator(x, y, given=None)`**: D-separation in the ancestral
  multi-world network of the query.

**Adjustment**

- **`is_backdoor_adjustment(x, y, z=None, drop_z=None, latex=False)`**
  - **x**, **y**: Treatment and outcome.
  - **z**: Candidate adjustment set.
  - **drop_z**: Variables to delete from the graph before the test.
  - **latex**: If true, return the adjustment formula instead of `True`.
  - **Returns:** `bool` or `Latex`.
- **`get_backdoor_adjustment_formula(x, y, z=None, given=None)`**: LaTeX
  backdoor formula if `z ∪ given` is a valid adjustment set, else `None`.
- **`find_backdoor_adjustment(x, y, included=None, restricted=None, drop_z=None, latex=False)`**:
  One valid backdoor set (or formula).
- **`find_all_backdoor_adjustments(x, y, included=None, restricted=None)`**:
  All valid backdoor sets, or `None`.
- **`get_adjustment_backdoor_graph(x, y, drop_z=None)`**: Graph used by the
  backdoor test (proper causal paths from `x` to `y` truncated).
- **`get_backdoor_graph(x)`**: Graph with outgoing edges from `x` removed.
- **`is_frontdoor_adjustment(x, y, z=None, xz=None, zy=None, latex=False)`**
  - **z**: Mediator set.
  - **xz**, **zy**: Adjustment sets for \(x \rightarrow z\) and \(z \rightarrow y\).
- **`get_frontdoor_adjustment_formula(x, y, z=None, xz=None, zy=None)`**
- **`find_frontdoor_adjustment(x, y, restricted=None, latex=False)`**: One
  triple `(Z, XZ, ZY)`.
- **`find_all_frontdoor_adjustments(x, y, restricted=None)`**: All such triples.
- **`get_frontdoor_graph(x, y, z)`**: Intersection of the X→Z and Z→Y backdoor
  graphs.

**Counterfactual calculus**

- **`apply_r1(pr, target_var, intervention_var, method="remove")`**: Rule 1
  (consistency). Alias of `apply_consistency`.
- **`apply_r2(pr, target_var=None, method="remove")`**: Rule 2 (independence).
  Alias of `apply_independence`. If `target_var` is omitted and `method` is
  `"remove"`, every variable in the conditioning set is tested.
- **`apply_r3(pr, target_var, intervention_var=None, method="remove")`**: Rule 3
  (exclusion). Alias of `apply_exclusion`.
- **`apply_consistency(pr, target_var, intervention_var, method="remove")`**
  - **pr** (`Pr`): Probability expression.
  - **target_var**: Event variables whose interventions should change.
  - **intervention_var**: Intervention being added or removed.
  - **method**: `"remove"` or `"add"`.
- **`apply_independence(pr, target_var=None, method="remove")`**: Add or drop
  conditioning variables when the ancestral multi-world network d-separates
  them from the event.
- **`apply_exclusion(pr, target_var, intervention_var=None, method="remove")`**:
  Drop (or add) interventions that are not ancestors of the target.
- **`apply_exclusion_var(var)`**: Exclusion reduction of a single counterfactual
  variable.
- **`apply_ctf_unnest(pr, target_var=None)`**: Replace nested counterfactual
  assignments with a `Summation` over explicit latent values.

**Counterfactual networks**

- **`build_TWN(interventions)`**: Twin network for one intervention set
  (self-interventions are kept).
- **`build_MWN(*interventions)`**: Multi-world network for several intervention
  sets (self-interventions are dropped).
- **`build_AMWN(counterfactuals)`**: Ancestral multi-world network containing
  only the query variables and their counterfactual ancestors.

**Display helpers**

- **`convert_to_dot(node_positions=None)`**: DOT source for the endogenous
  graph (directed plus dashed bidirected edges).
- **`convert_to_dot_combined_graph(node_positions=None)`**: DOT source that
  includes explicit exogenous parents.

### `Pr`

A `sympy` symbol that stores an event, optional conditioning set, and optional
intervention.

**Constructor parameters**

- **event** (`dict` or `set`): Event variables, optionally with values
  (`{y: 1}` or `{y}`).
- **given** (`dict` or `set`, optional): Conditioning assignment.
- **do** (`dict` or `set`, optional): Intervention.

**Methods**

- **`get_event()`**, **`get_condition()`**, **`get_action()`**: The stored
  event, condition, and `do` dictionaries.
- **`get_id()`**: Unique symbol id used internally by sympy.
- **`vars`**: `SymbolContainer` of variables appearing in the event and
  condition.
- **`apply_value_map(map)`**
  - **map** (`dict[Variable, int | Variable]`): Replace matching values in
    event, condition, and `do`.
  - **Returns:** a new `Pr`.
- **`apply_bayes(flip=None)`**
  - **flip** (`set[Variable]`, optional): Subset of the condition to move into
    the event. Defaults to the full condition.
  - **Returns:** `Pr(event ∪ flip | rest) / Pr(flip | rest)`.
- **`apply_bayes_inverse(expr)`** (static)
  - **expr**: A ratio of two `Pr` objects, `P(A, B) / P(B)`.
  - **Returns:** `P(A | B)` (times leftover factors when the conditions differ).

### `Summation`

Symbolic sum of an expression over one or more index variables.

**Constructor parameters**

- **expr**: Summand, typically built from `Pr` objects.
- **\*limits**: Index symbols, or `(symbol, domain)` pairs. A missing domain is
  treated as the universal set (filled in from the SCM when evaluated).

**Properties**

- **expr**: The summand.
- **symbols**: Index variables.
- **domains**: Mapping from each index to its domain.

`SymbolicSCM.query_exp` evaluates `Summation` by enumerating each index over
the endogenous domain (binary `{0, 1}` unless a domain was supplied).

### `SymbolContainer`

Read-mostly collection of `Variable`s returned by graph accessors and
separator searches. It supports indexing by position or by base name
(`graph.v.x` or `graph.v["x"]`), iteration, and set operations (`+`, `-`, `&`,
`|`, `in`). When a base name has several counterfactual copies, `container[name]`
is a list sorted by intervention depth.

## Contributing

Download this repository for the most up-to-date version of the library.

Open a pull request with `numpy`-style documented code together with relevant
book examples. For large multi-function classes, use Python ABC inheritance to
separate components that any class can mix in once it defines the abstract
properties those components require.

### Writing additional tests

Chapter notebooks are the source of truth for library behavior. Tests live next
to the package in `src/citk/__tests__/` and are named after the notebook they
encode:

| Notebook | Test module |
| -------- | ----------- |
| `chapter2.ipynb` | `src/citk/__tests__/chapter_2_tests.py` |
| `chapter4.ipynb` | `src/citk/__tests__/chapter_4_tests.py` |
| `chapter5.ipynb` | `src/citk/__tests__/chapter_5_tests.py` |

When you add a `chapterN.ipynb` example that exercises library tools:

1. Add `src/citk/__tests__/chapter_N_tests.py` (or extend the existing file).
2. Name tests `test_<example_or_definition>_<behavior>` so they are collected
   by pytest (`python_functions = test_*`).
3. Import the public API from `citk` (not from `src`). Shared set-comparison
   helpers live in `src/citk/__tests__/helpers.py`.
4. Assert the same numeric values, independence facts, adjustment sets, and
   symbolic `Pr` / `Summation` strings that the notebook produces. Use
   `pytest.approx` for probabilities. Skip display-only cells such as
   `graph.draw(...)` and seaborn plots; cover the underlying method
   (`query`, `is_d_separator`, `convert_to_dot`, …) instead.
5. Keep new library code in NumPy docstring style, and include the book
   example that motivated the change.

Install test dependencies and run the suite from the repository root:

```bash
pip install -e "./citk[test]"
citk-test
# or equivalently:
pytest citk/src/__tests__
python -m citk.src.__tests__
```

`citk-test` is the console script shipped with the package. Extra pytest flags
are forwarded, for example `citk-test -k chapter_2`. Hatch users can run
`hatch run test`.

Publish a release with the standard PyPI flow:

```bash
pip install -e "./citk[dev]"
python -m build
twine upload citk/dist/*
```

## Attribution

CITK is authored by Ryan Sherby at the Columbia Causal AI Lab, Columbia
University (director: Elias Bareinboim). The library accompanies examples from
the Causal Artificial Intelligence book.

- **Contact:** ryan.sherby@columbia.edu
- **Lab:** https://causalai.net/
- **Copyright:** Copyright 2025 Columbia Causal AI Lab (MIT License)
- **Citation:** see [`CITATION.cff`](CITATION.cff)
