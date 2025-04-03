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
        # Implement Bellman-Ford for directed graphs
        pass

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
