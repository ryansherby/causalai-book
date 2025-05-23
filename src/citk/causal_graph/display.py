from typing import Dict, List, Optional, Union, overload, Tuple

import networkx as nx
import sympy as sp

from src.return_classes import SymbolContainer

class Display:
    
    @property
    def de_graph(self):
        return nx.DiGraph()
    
    @property
    def be_graph(self):
        return nx.MultiGraph()
    
    @property
    def combined_graph(self):
        return nx.DiGraph()

    @property
    def v(self):
        return SymbolContainer()
    
    @property
    def u(self):
        return SymbolContainer()
    
    @property
    def syn(self):
        # Mapping from a symbol to its counterfactual variant
        return {}
    
    @property
    def cc(self):
        return set(set())
    
    
    def convert_to_dot(self, node_positions: Dict[sp.Symbol, Tuple[int, int]] = {}) -> str:
        """
        Converts the graph to the DOT format for visualization.

        Parameters
        ----------
        node_positions : Dict[sp.Symbol, Tuple[int, int]], optional
            A dictionary mapping nodes to their positions in the graph (default is empty dictionary).

        Returns
        -------
        str
            A string representing the graph in DOT format.
        """
        _node_positions = {
            self.syn.get(node, node): pos
            for node, pos in node_positions.items()
        }
        
        dot_str = "digraph G {\n  rankdir=LR;\n"
        
        for node in self.v:
            pos = (
                f'pos="{_node_positions[node][0]},{_node_positions[node][1]}!"'
                if node in _node_positions.keys()
                else ""
            )
            fillcolor = "style=filled, fillcolor=lightgray"
            dot_str += f'  "{node}" [label="{node}" {pos} {fillcolor}];\n'
        
        style = f"penwidth=2.0"
        
        for edge in self.de_graph.edges:
            arrow_type = f"[{style}]"
            dot_str += f'  "{edge[0]}" -> "{edge[1]}" {arrow_type};\n'
        
        for edge in self.be_graph.edges(keys=False):
            arrow_type = f"[dir=both, style=dashed, constraint=false, splines=curved, {style}]"
            dot_str += f'  "{edge[0]}" -> "{edge[1]}" {arrow_type};\n'
        
        return dot_str + "}"
    
    
    def convert_to_dot_combined_graph(self, node_positions: Dict[sp.Symbol, Tuple[int, int]] = {}) -> str:
        """
        Converts the combined graph to the DOT format for visualization.

        Parameters
        ----------
        node_positions : Dict[sp.Symbol, Tuple[int, int]], optional
            A dictionary mapping nodes to their positions in the graph (default is empty dictionary).

        Returns
        -------
        str
            A string representing the combined graph in DOT format.
        """
        _node_positions = node_positions
        
        dot_str = "digraph G {\n  rankdir=LR;\n"
        
        for node in set(self.v | self.u):
            pos = (
                f'pos="{_node_positions[node][0]},{_node_positions[node][1]}!"'
                if node in _node_positions.keys()
                else ""
            )
            fillcolor = "style=filled, fillcolor=lightgray"
            dot_str += f'  "{node}" [label="{node}" {pos} {fillcolor}];\n'
        
        style = f"penwidth=2.0"
        
        for edge in self.combined_graph.edges:
            if edge[0] in self.u:
                arrow_type = f"[style=dashed, constraint=false, splines=curved, {style}]"
            else:
                arrow_type = f"[{style}]"
            dot_str += f'  "{edge[0]}" -> "{edge[1]}" {arrow_type};\n'
        
        
        return dot_str + "}"
            
        
        
        
        