import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import random

seed = 2025
random.seed(seed)
np.random.seed(seed)
def count_undirected_simple_paths(graph: nx.Graph) -> int:
    """
    Calculates the number of undirected simple paths in a given graph.

    Args:
        graph: The networkx graph object.

    Returns:
        The total number of undirected simple paths.
    """
    order = graph.number_of_nodes()
    if order < 2:
        return 0
        
    # This is a computationally intensive operation.
    num_paths = sum(
        sum(1 for _ in nx.all_simple_paths(graph, x, y)) 
        for x, y in combinations(graph.nodes(), 2)
    )
    return num_paths

def run_cumulative_pruning_simulation(order: int, rates: np.ndarray) -> list[int]:
    """
    Simulates the PC pruning process with cumulative edge removal to ensure
    monotonic decay of path counts.

    Args:
        order: The number of vertices (variables) in the graph.
        rates: A numpy array of independence rates to test.

    Returns:
        A list of path counts corresponding to each rate.
    """
    G_initial = nx.complete_graph(order)
    all_variable_pairs = list(combinations(G_initial.nodes(), 2))
    
    # Shuffle the list of pairs ONCE to create a deterministic removal order.
    random.shuffle(all_variable_pairs)
    
    path_counts_over_time = []

    # Iterate through each rate, removing a cumulative number of edges.
    for rate in rates:
        G_pruned = G_initial.copy()
        
        # Calculate how many edges to remove for the current rate.
        num_pairs_to_remove = int(round(len(all_variable_pairs) * rate))
        edges_to_remove = all_variable_pairs[:num_pairs_to_remove]
        
        if edges_to_remove:
            G_pruned.remove_edges_from(edges_to_remove)
            
        path_counts_over_time.append(count_undirected_simple_paths(G_pruned))
        
    return path_counts_over_time


def visualize_decay_curves(results: dict, rates: np.ndarray):
    """
    Generates a line plot showing the decay of simple paths as independence rate increases.
    
    Args:
        results: A dictionary where keys are graph sizes and values are lists of path counts.
        rates: The array of independence rates tested.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i, (order, path_counts) in enumerate(results.items()):
        plot_values = np.array(path_counts) + 1
        ax.plot(rates * 100, plot_values, marker='o', linestyle='-', label=f'K{order}', color=colors[i])

    ax.set_xlabel('Percentage of Independent Pairs (%)')
    # Update the label to reflect the +1 adjustment for the log scale.
    ax.set_ylabel('Number of Remaining Simple Paths + 1 (Log Scale)')
    ax.set_title('Path Count Decay by Percentage of Independent Pairs', fontsize=16)
    
    ax.set_yscale('log')

    ax.set_ylim(bottom=-10)
    ax.grid(True, which="both", ls="--")
    ax.legend(title="Graph Size")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    GRAPH_SIZES = range(5, 10)  # Test for K5 through K9
    INDEPENDENCE_RATES = np.linspace(0, 1.0, 20) 

    simulation_results = {size: [] for size in GRAPH_SIZES}

    print("Running cumulative pruning simulations...")
    for n in GRAPH_SIZES:
        print(f"Processing K{n}...")
        path_counts = run_cumulative_pruning_simulation(n, INDEPENDENCE_RATES)
        simulation_results[n] = path_counts
    
    print(simulation_results)
    visualize_decay_curves(simulation_results, INDEPENDENCE_RATES)
