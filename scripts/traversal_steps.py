from typing import Iterable
import networkx as nx
import matplotlib.pyplot as plt


def get_all_simple_paths_loops(G, source, targets):
    # Copied and modified from networkx's implementation
    # https://github.com/networkx/networkx/blob/36e8a1ee85ca0ab4195a486451ca7d72153e2e00/networkx/algorithms/simple_paths.py#L362
    if isinstance(targets, int):
        targets = {targets}
    elif isinstance(targets, Iterable):
        targets = set(targets)
    else:
        raise TypeError("targets must be an int or an iterable of ints")

    cutoff = G.order()
    get_edges = (
        (lambda node: G.edges(node, keys=True))
        if G.is_multigraph()
        else (lambda node: G.edges(node))
    )

    current_path = {None: None}
    stack = [iter([(None, source)])]
    loops = 0
    while stack:
        next_edge = next((e for e in stack[-1] if e[1] not in current_path), None)
        if next_edge is None:
            stack.pop()
            current_path.popitem()
            continue
        loops += 1
        previous_node, next_node, *_ = next_edge

        if len(current_path) - 1 < cutoff and (
            targets - current_path.keys() - {next_node}
        ):
            current_path[next_node] = next_edge
            stack.append(iter(get_edges(next_node)))
    return loops


def compare_loops(order: int) -> float:
    assert order > 1, "Order must be greater than 1"
    G = nx.complete_graph(order)
    single_target_loops = get_all_simple_paths_loops(G, 0, 1)
    multiple_targets_loops = get_all_simple_paths_loops(G, 0, range(1, order))
    return multiple_targets_loops / (single_target_loops * (order - 1))


if __name__ == "__main__":
    orders = range(2, 13)
    ratios = [compare_loops(order) for order in orders]

    plt.figure()
    plt.plot(orders, ratios, marker='o')
    plt.xlabel("Order of the complete graph")
    plt.ylabel("Ratio")
    plt.axhline(y=0.5, color='red', linestyle='--', label='y=0.5')
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.xticks(orders)
    plt.xlim(min(orders), max(orders))
    plt.title("Ratio of node traversals for multiple targets vs. n-1 single target")
    plt.grid(True)
    # plt.show()
    plt.savefig("traversal_ratio.svg", format="svg")
