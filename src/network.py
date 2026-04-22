"""
Co-Author Network Analysis
============================
Build and analyze co-authorship networks from DBLP data.
"""

from collections import Counter
from itertools import combinations
from typing import Optional

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def build_coauthor_graph(
    df: pd.DataFrame,
    min_papers: int = 2,
) -> nx.Graph:
    """
    Build a co-authorship graph from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'authors' column (list of author names)
    min_papers : int
        Minimum number of co-authored papers for an edge to be included

    Returns
    -------
    nx.Graph
        Undirected weighted graph where:
        - nodes = authors
        - edge weight = number of co-authored papers
    """
    edge_counter = Counter()

    for authors in df["authors"]:
        if isinstance(authors, list) and len(authors) > 1:
            # Create edges for all author pairs in the paper
            for a1, a2 in combinations(sorted(authors), 2):
                edge_counter[(a1, a2)] += 1

    # Build graph with filtered edges
    G = nx.Graph()
    for (a1, a2), weight in edge_counter.items():
        if weight >= min_papers:
            G.add_edge(a1, a2, weight=weight)

    print(f"Co-author graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def compute_centralities(G: nx.Graph, top_n: int = 20) -> pd.DataFrame:
    """
    Compute centrality measures for the graph.

    Parameters
    ----------
    G : nx.Graph
        Co-authorship graph
    top_n : int
        Number of top authors to return

    Returns
    -------
    pd.DataFrame
        DataFrame with centrality measures, sorted by degree centrality
    """
    degree_cent = nx.degree_centrality(G)

    # Betweenness is expensive on large graphs — sample if needed
    if G.number_of_nodes() > 10000:
        betweenness_cent = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
    else:
        betweenness_cent = nx.betweenness_centrality(G)

    # Build results DataFrame
    records = []
    for node in G.nodes():
        records.append({
            "author": node,
            "degree": G.degree(node),
            "degree_centrality": degree_cent[node],
            "betweenness_centrality": betweenness_cent.get(node, 0),
            "weighted_degree": sum(d["weight"] for _, _, d in G.edges(node, data=True)),
        })

    df_cent = pd.DataFrame(records)
    df_cent = df_cent.sort_values("degree_centrality", ascending=False)

    return df_cent.head(top_n)


def get_top_subgraph(G: nx.Graph, top_n: int = 50) -> nx.Graph:
    """
    Extract subgraph of the top-N most connected authors.

    Parameters
    ----------
    G : nx.Graph
        Full co-authorship graph
    top_n : int
        Number of top authors to include

    Returns
    -------
    nx.Graph
        Subgraph induced by the top-N authors
    """
    # Get top-N by degree
    top_authors = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:top_n]
    return G.subgraph(top_authors).copy()


def get_author_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-author publication statistics.

    Returns DataFrame with: author, paper_count, first_year, last_year, venues
    """
    author_records = []

    # Explode authors list
    df_exploded = df.explode("authors").dropna(subset=["authors"])

    author_groups = df_exploded.groupby("authors")

    for author, group in author_groups:
        author_records.append({
            "author": author,
            "paper_count": len(group),
            "first_year": group["year"].min(),
            "last_year": group["year"].max(),
            "num_venues": group["venue"].nunique(),
            "top_venue": group["venue"].mode().iloc[0] if not group["venue"].mode().empty else "Unknown",
        })

    df_authors = pd.DataFrame(author_records)
    return df_authors.sort_values("paper_count", ascending=False)


def plot_network(
    G: nx.Graph,
    title: str = "Co-Author Network",
    figsize: tuple = (14, 10),
    node_size_factor: float = 5.0,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize a co-authorship network graph.

    Parameters
    ----------
    G : nx.Graph
        Graph to visualize
    title : str
        Plot title
    figsize : tuple
        Figure size
    node_size_factor : float
        Multiplier for node sizes based on degree
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # Node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [degrees[n] * node_size_factor + 20 for n in G.nodes()]

    # Node colors based on degree
    degree_values = [degrees[n] for n in G.nodes()]
    norm = mcolors.Normalize(vmin=min(degree_values), vmax=max(degree_values))

    # Edge widths based on weight
    edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.3 + (w / max_weight) * 2 for w in edge_weights]

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.3,
        edge_color="#888888",
    )

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=degree_values,
        cmap=plt.cm.YlOrRd,
        alpha=0.85,
        edgecolors="#333333",
        linewidths=0.5,
    )

    # Labels for top nodes
    top_nodes = sorted(G.nodes(), key=lambda n: degrees[n], reverse=True)[:15]
    labels = {n: n for n in top_nodes}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7,
        font_weight="bold",
        font_color="#222222",
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")

    plt.colorbar(nodes, ax=ax, label="Degree", shrink=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def degree_distribution(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the degree distribution of the graph.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (degrees, counts) arrays
    """
    degrees = [d for _, d in G.degree()]
    unique, counts = np.unique(degrees, return_counts=True)
    return unique, counts
