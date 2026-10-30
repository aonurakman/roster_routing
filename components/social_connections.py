import networkx as nx
import numpy as np


def generate_social_network(
    agents,
    topology="small_world",
    mean_degree=6,
    rewiring_prob=0.1,
    weight_sigma=0.5,
    ensure_connected=True,
    seed=42,
) -> dict:
    ids = sorted((agent.id for agent in agents), key=int)

    n = len(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    if topology == "small_world":
        k = min(n - 1, max(2, 2 * round(mean_degree / 2)))
        k -= k % 2

        graph = (
            nx.watts_strogatz_graph(n, k, rewiring_prob, seed=seed)
            if k >= 2 else nx.path_graph(n)
        )

    elif topology == "random":
        p = min(1.0, mean_degree / (n - 1))
        graph = nx.fast_gnp_random_graph(n, p, seed=seed)

    elif topology == "scale_free":
        m = min(n - 1, max(1, round(mean_degree / 2)))
        graph = nx.barabasi_albert_graph(n, m, seed=seed)

    else:
        raise ValueError(f"Unknown topology: {topology}")

    if ensure_connected:
        components = list(nx.connected_components(graph))
        for left, right in zip(components, components[1:]):
            u = rng.choice(list(left))
            v = rng.choice(list(right))
            graph.add_edge(int(u), int(v))

    adjacency = {agent_id: {} for agent_id in ids}

    for u, v in graph.edges():
        # Positive, lognormally distributed tie strength (mean 1).
        weight = float(rng.lognormal(
            mean=-weight_sigma**2 / 2,
            sigma=weight_sigma,
        ))

        id_u, id_v = ids[u], ids[v]
        adjacency[id_u][id_v] = weight
        adjacency[id_v][id_u] = weight
        
    adjacency = normalize_social_connections(adjacency)
    return adjacency


def normalize_social_connections(adjacency: dict) -> dict:
    """Normalize each traveler's influence weights to sum to 1."""
    normalized = {}

    for agent_id, neighbors in adjacency.items():
        total = sum(neighbors.values())
        normalized[agent_id] = (
            {other: weight / total for other, weight in neighbors.items()}
            if total > 0 else {}
        )

    return normalized
