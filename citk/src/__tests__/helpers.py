"""Shared helpers for chapter-aligned citk tests."""

from __future__ import annotations

from typing import Iterable


def as_names(container) -> set[str]:
    """Return the string names of symbols in a container-like object."""
    return {str(symbol) for symbol in container}


def as_name_sets(containers: Iterable) -> set[frozenset[str]]:
    """Return a set of frozensets of symbol names, ignoring container order."""
    return {frozenset(str(symbol) for symbol in container) for container in containers}


def as_name_triples(triples: Iterable) -> set[tuple[frozenset[str], ...]]:
    """Normalize a sequence of adjustment triples to comparable name sets."""
    return {
        tuple(frozenset(str(symbol) for symbol in part) for part in triple)
        for triple in triples
    }


def intervention_mains(var) -> set[str]:
    """Return the base names of interventions attached to a variable."""
    return {key.main for key in var.interventions}


def assert_contains_all(text: str, *parts: str) -> None:
    """Assert that every substring appears in ``text`` (order-independent)."""
    missing = [part for part in parts if part not in text]
    assert not missing, f"Missing {missing} in {text}"
