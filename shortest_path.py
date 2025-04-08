from graphs import Graph, WeightedGraph, HeuristicGraph
from abc import ABC, abstractmethod
from typing import List, Dict
import heapq


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
        self.distance = {node: float('inf') for node in graph.adj_list}
        self.previous = {node: None for node in graph.adj_list}
        self.distance[source] = 0

        # Priority queue: (distance, node)
        queue = [(0, source)]

        while queue:
            # Get the current distance from the source to the node at the top of the priority queue
            current_dist, current_node = heapq.heappop(queue)

            # If we are at the destination end it
            if current_node == dest:
                break
            
            # If the current distance is greater than a distance to this node we have already found skip
            if current_dist > self.distance[current_node]:
                continue 
            
            # For every adjacent node
            for neighbor in graph.neighbors(current_node):
                # Add the edge weight to the current distance
                distance_through_node = current_dist + graph.get_edge_weight(current_node, neighbor)
                # If the distance_through_node variable is better than the current distance replace it
                if distance_through_node < self.distance[neighbor]:
                    self.distance[neighbor] = distance_through_node
                    self.previous[neighbor] = current_node
                    heapq.heappush(queue, (distance_through_node, neighbor))
        # Return the distance
        return self.distance[dest] if self.distance[dest] != float('inf') else -1

    def get_shortest_path(self, dest: int) -> list:
        """Reconstruct the shortest path to the destination after calling calc_sp"""
        path = []
        while dest is not None:
            path.append(dest)
            dest = self.previous[dest]
        return path[::-1]
    

    
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
        # Initialize predecessor dictionary
        for node in graph.adj_list:
            self.previous[node] = None
            self.distance[node] = float('inf')
        self.distance[source] = 0
        # Initialize heap and marked lists
        heap = []
        marked = []

        # Initialize g and h
        g = self.distance[source]
        h = graph.heuristic[source][dest]
        f = g + h
        heapq.heappush(heap, (f, source))
        while len(heap) > 0:
            # Get node off the top of the heap (node with lowest f value)
            current_f, current = heapq.heappop(heap)

            # Get g value for current node
            g = self.distance[current]

            # If the current node is the destination we are done and we reconstruct the path from the predecessor dict
            if current == dest:
                return self.previous, self.reconstruct(source, dest, self.previous)

            # Otherwise add the current node to marked
            marked.append(current)

            # For every edge on the current node
            for x in graph.adj_list[current]:
                neighbour = x[0]
                # If the nieghbour is in marked skip it
                if neighbour in marked:
                    continue

                # Compute g and f values for the neighbour node
                # maybe_g = g + the edge to neighbour
                maybe_g = g + graph.get_edge_weight(current, neighbour)
                # maybe_f = maybe_g + the distance estimate from neighbour to dest
                maybe_f = maybe_g + graph.heuristic[neighbour][dest]
                
                # If the maybe_g value is better than the current distance replace and push onto heap.
                if maybe_g < self.distance[neighbour]:
                    self.distance[neighbour] = maybe_g
                    self.previous[neighbour] = current
                    heapq.heappush(heap, (maybe_f, neighbour))
        return False
    def reconstruct(self, source: int, dest: int, pred: dict[int, int]):
        node = dest
        path = [dest]
        while node != source:
            path.append(pred[node])
            node = pred[node]
        path.reverse()
        return path

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
