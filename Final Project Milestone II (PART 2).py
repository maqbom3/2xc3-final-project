import heapq  #to use a priority queue (min-heap)
import random
import matplotlib.pyplot as plt
import heapq
import time

def draw_plot_part2_3(graph_sizes, dijkstra_times, bellman_times):
    plt.figure(figsize=(10, 6))
    plt.plot(graph_sizes, dijkstra_times, marker='o', linestyle='-', label="Dijkstra (k-relax)")
    plt.plot(graph_sizes, bellman_times, marker='s', linestyle='--', label="Bellman-Ford (k-relax)")
    plt.xlabel("Graph Size (n)")
    plt.ylabel("Runtime (ms)")
    plt.title("Performance Comparison: Dijkstra vs Bellman-Ford with Relaxation Limit k")
    plt.grid(True)
    plt.legend()
    plt.show()

# Part 2.1
def dijkstra(graph, source, k):
    # init the shortest known distances to all nodes as infinity
    distances = {node: float('inf') for node in graph}
    distances[source] = 0  # Distance to the source is 0
    # init path tracking so each node maps to the path taken to reach it
    paths = {node: [] for node in graph}
    paths[source] = [source]  # Start path from source node
    # Track how many times each node has been relaxed
    relaxation_count = {node: 0 for node in graph}
    # Set up the priority queue with the source node and distance 0
    priority_queue = [(0, source)]
    
    # continue exploring until the queue is empty
    while priority_queue:
        # get the node with the smallest known distance
        current_distance, current_node = heapq.heappop(priority_queue)
        # skip this node if we’ve already found a better path before
        if current_distance > distances[current_node]:
            continue
        # explore all neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            # If we've already relaxed this neighbor k times, skip it
            if relaxation_count[neighbor] >= k:
                continue
            # Calculate the new distance through the current node
            distance = current_distance + weight
            # If the new path is shorter, update distance and path
            if distance < distances[neighbor]:
                distances[neighbor] = distance  # Update shortest distance
                paths[neighbor] = paths[current_node] + [neighbor]  # Update path
                relaxation_count[neighbor] += 1  # Increase relaxation count
                heapq.heappush(priority_queue, (distance, neighbor))  # Add neighbor to queue
                
    return distances, paths

#Part 2.2
def bellman_ford(graph, source, k):
    # set shortest distances from the source to all nodes as infinity
    distances = {node: float('infinity') for node in graph}
    distances[source] = 0  # Distance to source is 0

    # init paths dictionary to track actual shortest paths
    paths = {node: [] for node in graph}
    paths[source] = [source]  
    # init a counter for how many times each node has been relaxed
    relaxation_count = {node: 0 for node in graph}
    # Repeat the process up to k times 
    
    for _ in range(k):
        updated = False  # Track if any distance was updated during this iteration
        # Iterate through each node and its neighbors
        for node in graph:
            for neighbor, weight in graph[node].items():
                # Skip relaxing this neighbor if it has already been relaxed k times
                if relaxation_count[neighbor] >= k:
                    continue
                # Relax the edge if a shorter path is found
                if distances[node] != float('infinity') and distances[node] + weight < distances[neighbor]:
                    # update the shortest distance
                    distances[neighbor] = distances[node] + weight
                    # ipdate the path to this neighbor
                    paths[neighbor] = paths[node] + [neighbor]
                    # increase the relaxation count for the neighbor
                    relaxation_count[neighbor] += 1
                    # Mark that we had at least one update this round
                    updated = True
        # If no update happened in this pass, stop early
        if not updated:
            break
        
    return distances, paths


#Part 2.3 

# Helper: Ground Truth 
def bellman_ford_true(graph, source):
    distances = {node: float('infinity') for node in graph}
    distances[source] = 0
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u].items():
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
    return distances

# Generate Random Graph 
def generate_random_graph(n, density=0.3, weight_range=(1, 10)):
    graph = {i: {} for i in range(n)}
    max_edges = n * (n - 1)
    num_edges = int(max_edges * density)
    edges_added = 0
    while edges_added < num_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and v not in graph[u]:
            graph[u][v] = random.randint(*weight_range)
            edges_added += 1
    return graph

# Experiment
def experiment_vary_graph_size():
    graph_sizes = [10, 30, 50, 70, 100]
    k = 3
    density = 0.3
    num_trials = 3

    dijkstra_times = []
    bellman_times = []
    dijkstra_acc = []
    bellman_acc = []

    for size in graph_sizes:
        d_time_total = 0
        b_time_total = 0
        d_acc_total = 0
        b_acc_total = 0

        for _ in range(num_trials):
            graph = generate_random_graph(size, density)
            source = 0
            ground_truth = bellman_ford_true(graph, source)

            start = time.time()
            d_out, _ = dijkstra(graph, source, k)
            d_time_total += (time.time() - start) * 1000

            start = time.time()
            b_out, _ = bellman_ford(graph, source, k)
            b_time_total += (time.time() - start) * 1000

            correct_d = sum(1 for node in graph if d_out[node] == ground_truth[node])
            correct_b = sum(1 for node in graph if b_out[node] == ground_truth[node])

            d_acc_total += correct_d / size * 100
            b_acc_total += correct_b / size * 100

        dijkstra_times.append(d_time_total / num_trials)
        bellman_times.append(b_time_total / num_trials)
        dijkstra_acc.append(d_acc_total / num_trials)
        bellman_acc.append(b_acc_total / num_trials)

    return graph_sizes, dijkstra_times, bellman_times, dijkstra_acc, bellman_acc

def draw_performance_graph(sizes, d_times, b_times):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, d_times, marker='o', label="Dijkstra Runtime")
    plt.plot(sizes, b_times, marker='s', label="Bellman-Ford Runtime")
    plt.xlabel("Graph Size (n)")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime Comparison of Dijkstra vs Bellman-Ford with k=3")
    plt.grid(True)
    plt.legend()
    plt.show()

def draw_accuracy_graph(sizes, d_acc, b_acc):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, d_acc, marker='o', label="Dijkstra Accuracy")
    plt.plot(sizes, b_acc, marker='s', label="Bellman-Ford Accuracy")
    plt.xlabel("Graph Size (n)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison of Dijkstra vs Bellman-Ford with k=3")
    plt.grid(True)
    plt.legend()
    plt.show()

# ------------------ Run the One Experiment ------------------
if __name__ == "__main__":
    sizes, d_times, b_times, d_acc, b_acc = experiment_vary_graph_size()
    draw_performance_graph(sizes, d_times, b_times)
    draw_accuracy_graph(sizes, d_acc, b_acc)