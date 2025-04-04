from graphs import Graph, WeightedGraph, HeuristicGraph
from abc import ABC, abstractmethod
from typing import List, Dict


class SPAlgorithm(ABC):
    """Abstract class for shortest path algorithms"""
    @abstractmethod
    def calc_sp(self, graph: Graph, source: int, dest: int) -> float:
        pass


class Dijkstra(SPAlgorithm):
    """Dijkstra’s algorithm implementation"""
    def __init__(self):
        self.distance = {}
        self.previous = {}

    def calc_sp(self, graph: WeightedGraph, source: int, dest: int) -> float:
        # Implement Dijkstra's algorithm for directed graphs
        pass

class BellmanFord(SPAlgorithm):
    """Bellman-Ford algorithm implementation"""
    def __init__(self):
        self.distance = {}
        self.previous = {}
    
    def calc_sp(self, graph: WeightedGraph, source: int, dest: int) -> float:
        # Initialize distances and previous nodes
        self.distance = {node: float('infinity') for node in range(graph.num_vertices())}
        self.previous = {node: None for node in range(graph.num_vertices())}
        # Set distance to source as 0
        self.distance[source] = 0
        # Relax edges |V| - 1 times
        for _ in range(graph.num_vertices() - 1):
            # For each edge in the graph
            for u in range(graph.num_vertices()):
                for v in graph.neighbors(u):
                    weight = graph.get_edge_weight(u, v)
                    # If we can improve the shortest path to v through u
                    if self.distance[u] != float('infinity') and self.distance[u] + weight < self.distance[v]:
                        self.distance[v] = self.distance[u] + weight
                        self.previous[v] = u
        # Check for negative weight cycles
        for u in range(graph.num_vertices()):
            for v in graph.neighbors(u):
                weight = graph.get_edge_weight(u, v)
                if self.distance[u] != float('infinity') and self.distance[u] + weight < self.distance[v]:
                    # Negative weight cycle detected
                    raise ValueError("Graph contains a negative weight cycle")
        # Return the shortest distance to destination
        return self.distance[dest]
    
    def get_path(self, dest: int) -> list:
        """Return the shortest path to the destination as a list of vertices"""
        if self.distance.get(dest, float('infinity')) == float('infinity'):
            return []  # No path exists
        
        path = []
        current = dest
        
        # Reconstruct the path from destination to source
        while current is not None:
            path.append(current)
            current = self.previous[current]
        
        # Return the path in correct order (source to destination)
        return path[::-1]

class AStar(SPAlgorithm):
    """A* algorithm implementation"""
    def __init__(self):
        self.distance = {}
        self.previous = {}
    def calc_sp(self, graph: HeuristicGraph, source: int, dest: int) -> float:
        # Implement A* algorithm for directed graphs using heuristic
        pass

class ShortPathFinder:
    """Class that finds the shortest path using a given algorithm"""
    def __init__(self):
        self.graph = None
        self.algorithm = None

    def set_graph(self, graph: Graph):
        self.graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm

    def calc_short_path(self, source: int, dest: int) -> float:
        if not self.graph or not self.algorithm:
            raise ValueError("Graph or Algorithm not set")
        return self.algorithm.calc_sp(self.graph, source, dest)
