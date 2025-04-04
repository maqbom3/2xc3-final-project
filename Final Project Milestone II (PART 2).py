import heapq
import random
import matplotlib
import matplotlib.pyplot as plt
import time
from graphs import WeightedGraph  # Custom WeightedGraph class
from shortest_path import BellmanFord  # Custom Bellman-Ford solver class
from typing import Dict, List, Tuple

# Part 2.1 — Dijkstra's algorithm with k-relaxation limit
def dijkstra(graph: WeightedGraph, source: int, k: int) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    # Initialize distances to infinity, except the source node
    distances = {node: float('inf') for node in range(graph.num_vertices())}
    distances[source] = 0
    # Track shortest paths as lists of nodes
    paths = {node: [] for node in range(graph.num_vertices())}
    paths[source] = [source]
    # Keep count of how many times each node has been relaxed
    relaxation_count = {node: 0 for node in range(graph.num_vertices())}
    # Priority queue stores (distance, node)
    priority_queue = [(0, source)]

    while priority_queue:
        # Pop the node with the smallest known distance
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue
        # Relax each neighbor
        for neighbor in graph.neighbors(current_node):
            weight = graph.w(current_node, neighbor)
            if relaxation_count[neighbor] >= k:
                continue
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance  # Update shorter distance
                paths[neighbor] = paths[current_node] + [neighbor]  # Update path
                relaxation_count[neighbor] += 1  # Count the relaxation
                heapq.heappush(priority_queue, (distance, neighbor))  # Push to queue

    return distances, paths

# Part 2.2 — Bellman-Ford algorithm with k-relaxation
def bellman_ford(graph: WeightedGraph, source: int, k: int) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    # Initialize distances and paths
    distances = {node: float('infinity') for node in range(graph.num_vertices())}
    distances[source] = 0
    paths = {node: [] for node in range(graph.num_vertices())}
    paths[source] = [source]
    relaxation_count = {node: 0 for node in range(graph.num_vertices())}

    # Relax edges up to k times
    for _ in range(k):
        updated = False
        for u, v, weight in graph.get_edges():  # Loop through all edges
            if relaxation_count[v] >= k:
                continue
            if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                paths[v] = paths[u] + [v]
                relaxation_count[v] += 1
                updated = True
        if not updated:
            break  # Early stop if no updates

    return distances, paths

# Random graph generator
def generate_random_graph(n: int, density: float = 0.3, weight_range: Tuple[int, int] = (1, 10)) -> WeightedGraph:
    graph = WeightedGraph()
    vertex_set = set()
    max_edges = n * (n - 1)  # Max edges in directed graph
    num_edges = int(max_edges * density)  # Target number of edges
    edges_added = 0
    added = set()  # Track which edges have been added to avoid duplicates

    while edges_added < num_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and (u, v) not in added:
            weight = random.randint(weight_range[0], weight_range[1])
            graph.add_edge(u, v, weight)
            vertex_set.add(u)
            vertex_set.add(v)
            added.add((u, v))
            edges_added += 1

    graph.num_vertices = lambda: len(vertex_set)  # Dynamically define num_vertices
    return graph

# Part 2.3 — Run experiment on varying graph sizes
def experiment_vary_graph_size():
    graph_sizes = [10, 30, 50, 70, 100]  # Sizes to test
    k = 3  # Max relaxations
    density = 0.3  # Edge density
    num_trials = 3  # Trials per size

    dijkstra_times = []
    bellman_times = []
    dijkstra_acc = []
    bellman_acc = []

    for size in graph_sizes:
        print(f"\nRunning experiments for graph size: {size}")
        d_time_total = 0
        b_time_total = 0
        d_acc_total = 0
        b_acc_total = 0
        valid_trials = 0

        while valid_trials < num_trials:
            graph = generate_random_graph(size, density)
            source = 0  # Start from node 0

            bf_solver = BellmanFord()
            try:
                bf_solver.calc_sp(graph, source, source)  # Ground truth using full Bellman-Ford
            except Exception as e:
                print(f"  Skipping graph due to: {e}")
                continue

            ground_truth = bf_solver.distance

            # Dijkstra runtime and accuracy
            start = time.time()
            d_out, _ = dijkstra(graph, source, k)
            d_time_total += (time.time() - start) * 1000  # Convert to ms

            # Bellman-Ford runtime and accuracy
            start = time.time()
            b_out, _ = bellman_ford(graph, source, k)
            b_time_total += (time.time() - start) * 1000

            # Calculate accuracy by comparing with ground truth
            correct_d = sum(1 for node in ground_truth if d_out[node] == ground_truth[node])
            correct_b = sum(1 for node in ground_truth if b_out[node] == ground_truth[node])

            d_acc_total += correct_d / size * 100
            b_acc_total += correct_b / size * 100

            valid_trials += 1

        # Store average times and accuracies
        dijkstra_times.append(d_time_total / num_trials)
        bellman_times.append(b_time_total / num_trials)
        dijkstra_acc.append(d_acc_total / num_trials)
        bellman_acc.append(b_acc_total / num_trials)

    return graph_sizes, dijkstra_times, bellman_times, dijkstra_acc, bellman_acc

# Plotting the runtime graph
def draw_performance_graph(sizes, d_times, b_times):
    print("Drawing performance graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, d_times, marker='o', label="Dijkstra Runtime")
    plt.plot(sizes, b_times, marker='s', label="Bellman-Ford Runtime")
    plt.xlabel("Graph Size (n)")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime Comparison of Dijkstra vs Bellman-Ford with k=3")
    plt.grid(True)
    plt.legend()
    plt.savefig("runtime_plot.png")
    plt.show()

# Plotting the accuracy graph
def draw_accuracy_graph(sizes, d_acc, b_acc):
    print("Drawing accuracy graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, d_acc, marker='o', label="Dijkstra Accuracy")
    plt.plot(sizes, b_acc, marker='s', label="Bellman-Ford Accuracy")
    plt.xlabel("Graph Size (n)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison of Dijkstra vs Bellman-Ford with k=3")
    plt.grid(True)
    plt.legend()
    plt.savefig("accuracy_plot.png")
    plt.show()


if __name__ == "__main__":
    print("Starting experiment...")
    sizes, d_times, b_times, d_acc, b_acc = experiment_vary_graph_size()
    draw_performance_graph(sizes, d_times, b_times)
    draw_accuracy_graph(sizes, d_acc, b_acc)
    print("Experiment complete. Plots saved.")
