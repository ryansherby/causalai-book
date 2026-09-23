from importlib.resources import files

from .causal_graph import CausalGraph
from .scm import SymbolicSCM
from .sympy_classes import variables, Variable, Pr, Summation
from .return_classes import SymbolContainer

__author__ = "Ryan Sherby, Columbia Causal AI Lab"
__copyright__ = "Copyright 2025 Columbia Causal AI Lab"
_LOGO_FILENAME = "CITK Logo.png"


def get_logo_resource():
    """Return a traversable handle to the bundled CITK logo file."""
    return files(__name__).joinpath("assets", _LOGO_FILENAME)


def get_logo_bytes() -> bytes:
    """Read and return the bundled CITK logo bytes."""
    return get_logo_resource().read_bytes()

__all__ = [
    "CausalGraph",
    "SymbolicSCM",
    "variables",
    "Variable",
    "Pr",
    "Summation",
    "SymbolContainer",
    "get_logo_resource",
    "get_logo_bytes",
]
