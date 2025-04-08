import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict
from graphs import WeightedGraph, HeuristicGraph
from shortest_path import Dijkstra, AStar

# ------------------ Setup & Utilities ------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def build_heuristic_graph(dest, heuristic_type):
    hg = HeuristicGraph()
    for u, v, w in wg.get_edges():
        hg.add_edge(u, v, w)

    lat_d, lon_d = id_to_coords[dest]
    for node in id_to_coords:
        lat, lon = id_to_coords[node]
        if heuristic_type == 'haversine':
            h = haversine(lat, lon, lat_d, lon_d)
        elif heuristic_type == 'squared':
            h = (lat - lat_d) ** 2 + (lon - lon_d) ** 2
        elif heuristic_type == 'manhattan':
            h = abs(lat - lat_d) + abs(lon - lon_d)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic_type}")
        hg.set_heuristic(node, {dest: h})
    return hg

# ------------------ Categorization Logic ------------------

def build_line_to_stations_map(connections_df):
    mapping = defaultdict(set)
    for _, row in connections_df.iterrows():
        mapping[row['line']].add(int(row['station1']))
        mapping[row['line']].add(int(row['station2']))
    return mapping


def build_station_to_lines_map(connections_df):
    mapping = defaultdict(set)
    for _, row in connections_df.iterrows():
        mapping[int(row['station1'])].add(row['line'])
        mapping[int(row['station2'])].add(row['line'])
    return mapping

def lines_are_adjacent(lines1, lines2):
    for l1 in lines1:
        for l2 in lines2:
            if l1 == l2:
                continue
            # Check if they share any station (transfer station)
            if line_to_stations[l1] & line_to_stations[l2]:
                return True
    return False

def categorize_station_pair(s1, s2):
    lines1 = station_to_lines[s1]
    lines2 = station_to_lines[s2]
    if lines1 & lines2:
        return "same_line"
    elif lines_are_adjacent(lines1, lines2):
        return "adjacent_line"
    else:
        return "multi_transfer"

def sample_station_pairs_by_category(station_ids, sample_size=50):
    same, adjacent, multi = [], [], []
    attempts = 0
    while len(same) < sample_size or len(adjacent) < sample_size or len(multi) < sample_size:
        s1, s2 = random.sample(station_ids, 2)
        if s1 == s2:
            continue
        category = categorize_station_pair(s1, s2)
        if category == "same_line" and len(same) < sample_size:
            same.append((s1, s2))
        elif category == "adjacent_line" and len(adjacent) < sample_size:
            adjacent.append((s1, s2))
        elif category == "multi_transfer" and len(multi) < sample_size:
            multi.append((s1, s2))
        attempts += 1
        if attempts > 10000:
            break
    return {"same_line": same, "adjacent_line": adjacent, "multi_transfer": multi}

# ------------------ Run Experiments ------------------

def run_experiment_on_pairs(pairs, wg, id_to_coords, heuristics=['haversine']):
    results = []
    dijkstra = Dijkstra()
    astar = AStar()
    for source, dest in tqdm(pairs):
        try:
            start = time.time()
            d_distance = dijkstra.calc_sp(wg, source, dest)
            d_time = time.time() - start
            results.append({
                'source': source, 'dest': dest, 'algorithm': 'dijkstra', 'heuristic': 'none',
                'distance': d_distance, 'time': d_time
            })
            for h in heuristics:
                hg = build_heuristic_graph(dest, h)
                start = time.time()
                a_distance, _ = astar.calc_sp(hg, source, dest)
                a_time = time.time() - start
                results.append({
                    'source': source, 'dest': dest, 'algorithm': 'astar', 'heuristic': h,
                    'distance': astar.distance[dest], 'time': a_time
                })
        except Exception:
            continue
    return results

def run_all_pairs_experiment(wg, station_ids, id_to_coords, heuristics=['haversine', 'squared', 'manhattan'], max_pairs=1000):
    results = []
    dijkstra = Dijkstra()
    astar = AStar()
    pair_count = 0

    print(f"\nRunning all-pairs shortest path comparison (max {max_pairs or '∞'} pairs)...")
    for i, source in enumerate(tqdm(station_ids)):
        for dest in station_ids[i + 1:]:
            if max_pairs and pair_count >= max_pairs:
                return results
            try:
                start = time.time()
                d_distance = dijkstra.calc_sp(wg, source, dest)
                d_time = time.time() - start
                results.append({
                    'source': source, 'dest': dest, 'algorithm': 'dijkstra', 'heuristic': 'none',
                    'distance': d_distance, 'time': d_time
                })
                for h in heuristics:
                    hg = build_heuristic_graph(dest, h)
                    start = time.time()
                    a_distance, _ = astar.calc_sp(hg, source, dest)
                    a_time = time.time() - start
                    results.append({
                        'source': source, 'dest': dest, 'algorithm': 'astar', 'heuristic': h,
                        'distance': astar.distance[dest], 'time': a_time
                    })
                pair_count += 1
            except Exception:
                continue
    return results

# ------------------ Plotting ------------------

def plot_category_results(df):
    plt.figure(figsize=(12, 6))
    colors = {'same_line': 'blue', 'adjacent_line': 'green', 'multi_transfer': 'red'}

    for category in df['category'].unique():
        subset = df[(df['algorithm'] == 'astar') & (df['heuristic'] == 'haversine') & (df['category'] == category)]
        subset = subset.dropna(subset=['distance', 'time'])
        plt.scatter(subset['distance'], subset['time'], label=f"A* ({category})", color=colors[category], alpha=0.5, s=10)

        if len(subset) > 1:
            x = subset['distance'].values
            y = subset['time'].values
            m, b = np.polyfit(x, y, 1)
            x_sorted = np.sort(x)
            plt.plot(x_sorted, m * x_sorted + b, color=colors[category], linestyle='--', linewidth=2)

    plt.xlabel("Distance (km)")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs Distance for A* with Haversine Heuristic (by Line Category)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_pairs_results(df):
    plt.figure(figsize=(12, 6))
    colors = {'haversine': 'blue', 'squared': 'green', 'manhattan': 'orange', 'dijkstra': 'black'}
    heuristics = ['haversine', 'squared', 'manhattan']

    for h in heuristics:
        subset = df[(df['algorithm'] == 'astar') & (df['heuristic'] == h)]
        plt.scatter(subset['distance'], subset['time'], label=f"A* ({h})", alpha=0.4, s=10, c=colors[h])
        if len(subset) > 1:
            m, b = np.polyfit(subset['distance'], subset['time'], 1)
            plt.plot(np.sort(subset['distance']), m * np.sort(subset['distance']) + b, color=colors[h], linestyle='--')

    subset = df[df['algorithm'] == 'dijkstra']
    plt.scatter(subset['distance'], subset['time'], label='Dijkstra', alpha=0.4, s=10, c=colors['dijkstra'])
    if len(subset) > 1:
        m, b = np.polyfit(subset['distance'], subset['time'], 1)
        plt.plot(np.sort(subset['distance']), m * np.sort(subset['distance']) + b, color=colors['dijkstra'], linestyle='--')

    plt.xlabel("Distance (km)")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs Distance for Dijkstra and A* with Different Heuristics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Main Logic ------------------

if __name__ == "__main__":
    SAMPLE_SIZE = 1000
    MAX_PAIRS = 1000

    stations_df = pd.read_csv("data/london_stations.csv")
    connections_df = pd.read_csv("data/london_connections.csv")
    station_ids = stations_df['id'].tolist()
    id_to_coords = {
        row['id']: (row['latitude'], row['longitude'])
        for _, row in stations_df.iterrows()
    }
    line_to_stations = build_line_to_stations_map(connections_df)
    station_to_lines = build_station_to_lines_map(connections_df)

    wg = WeightedGraph()
    for _, row in connections_df.iterrows():
        s1, s2 = int(row['station1']), int(row['station2'])
        if s1 in id_to_coords and s2 in id_to_coords:
            lat1, lon1 = id_to_coords[s1]
            lat2, lon2 = id_to_coords[s2]
            dist = haversine(lat1, lon1, lat2, lon2)
            wg.add_edge(s1, s2, dist)
            wg.add_edge(s2, s1, dist)

    # Run categorized experiment
    sampled = sample_station_pairs_by_category(station_ids, sample_size=SAMPLE_SIZE)
    all_results = []
    for category, pairs in sampled.items():
        print(f"\nRunning for category: {category}")
        results = run_experiment_on_pairs(pairs, wg, id_to_coords)
        for r in results:
            r["category"] = category
        all_results.extend(results)
    df_category = pd.DataFrame(all_results)
    plot_category_results(df_category)

    # Run all-pairs experiment
    all_pair_results = run_all_pairs_experiment(wg, station_ids, id_to_coords, max_pairs=MAX_PAIRS)
    df_all = pd.DataFrame(all_pair_results)
    plot_all_pairs_results(df_all)
