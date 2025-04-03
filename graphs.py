from typing import List, Dict, Tuple

class Graph:
    """Base class for a directed graph"""
    def __init__(self):
        self.adj_list = {}

    def get_adj_nodes(self, node: int) -> List[int]:
        return [neighbor for neighbor, _ in self.adj_list.get(node, [])]

    def add_node(self, node: int):
        if node not in self.adj_list:
            self.adj_list[node] = []
    
    def add_edge(self, start: int, end: int, w: int):
        """Adds a directed edge from start to end with weight w"""
        self.adj_list.setdefault(start, []).append((end, 0))

    def get_num_of_nodes(self) -> int:
        return len(self.adj_list)
    
    def w(node: int ) -> float:
        return 0.0



class WeightedGraph(Graph):
    """Directed weighted graph"""
    def __init__(self):
        super().__init__()
        self.weights: Dict[Tuple[int, int], float] = {}

    def add_edge(self, start: int, end: int, w: float):
        """Adds a directed edge from start to end with weight w"""
        super().add_edge(start, end, w)
        self.weights[(start, end)] = w  # Directed edge only

    def w(self, node1: int, node2: int) -> float:
        """Returns weight of the directed edge from node1 to node2"""
        return self.weights.get((node1, node2), float('inf'))  # No reverse lookup

    def get_edges(self) -> List[Tuple[int, int, float]]:
        """Returns a list of all directed edges with weights"""
        return [(start, end, self.weights[(start, end)]) for (start, end) in self.weights]


class HeuristicGraph(WeightedGraph):
    """Directed weighted graph with heuristic values for A*"""
    def __init__(self):
        super().__init__()
        self.heuristic: Dict[int, float] = {}

    def get_heuristic(self) -> Dict[int, float]:
        """Returns the heuristic values"""
        return self.heuristic

    def set_heuristic(self, node: int, h: float):
        """Sets the heuristic value for a given node"""
        self.heuristic[node] = h
