"""
Topic Discovery & Clustering
==============================
Dimensionality reduction and density-based clustering of paper embeddings.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Reduce embedding dimensionality for visualization.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embeddings of shape (n_samples, n_features)
    method : str
        'umap' or 'pca'
    n_components : int
        Target dimensionality (2 or 3)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Reduced embeddings of shape (n_samples, n_components)
    """
    if method == "umap":
        import umap

        print(f"Running UMAP (n_components={n_components})...")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            metric=kwargs.get("metric", "cosine"),
            random_state=random_state,
            verbose=True,
        )
        reduced = reducer.fit_transform(embeddings)

    elif method == "pca":
        from sklearn.decomposition import PCA

        print(f"Running PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components, random_state=random_state)
        reduced = pca.fit_transform(embeddings)
        explained = sum(pca.explained_variance_ratio_) * 100
        print(f"Explained variance: {explained:.1f}%")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'pca'.")

    print(f"Reduced to shape {reduced.shape}")
    return reduced


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    **kwargs,
) -> np.ndarray:
    """
    Cluster embeddings using density-based methods.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embeddings (can be full-dim or reduced)
    method : str
        'hdbscan' or 'dbscan'

    Returns
    -------
    np.ndarray
        Cluster labels (-1 = noise)
    """
    if method == "hdbscan":
        import hdbscan

        print("Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=kwargs.get("min_cluster_size", 50),
            min_samples=kwargs.get("min_samples", 10),
            metric=kwargs.get("metric", "euclidean"),
            cluster_selection_method=kwargs.get("cluster_selection_method", "eom"),
        )
        labels = clusterer.fit_predict(embeddings)

    elif method == "dbscan":
        from sklearn.cluster import DBSCAN

        print("Running DBSCAN clustering...")
        clusterer = DBSCAN(
            eps=kwargs.get("eps", 0.5),
            min_samples=kwargs.get("min_samples", 10),
            metric=kwargs.get("metric", "euclidean"),
        )
        labels = clusterer.fit_predict(embeddings)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'hdbscan' or 'dbscan'.")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise:,} noise points ({n_noise/len(labels)*100:.1f}%)")

    return labels


def extract_cluster_keywords(
    titles: list[str],
    labels: np.ndarray,
    top_n: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """
    Extract top keywords per cluster using TF-IDF.

    Parameters
    ----------
    titles : list[str]
        List of paper titles
    labels : np.ndarray
        Cluster labels
    top_n : int
        Number of top keywords per cluster

    Returns
    -------
    dict[int, list[tuple[str, float]]]
        Mapping from cluster_id to list of (keyword, tfidf_score)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    unique_labels = sorted(set(labels))
    cluster_keywords = {}

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise

        mask = labels == label
        cluster_titles = [t for t, m in zip(titles, mask) if m]

        if len(cluster_titles) < 5:
            continue

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 2),
        )

        try:
            tfidf = vectorizer.fit_transform(cluster_titles)
            feature_names = vectorizer.get_feature_names_out()

            # Average TF-IDF scores across documents in cluster
            avg_scores = tfidf.mean(axis=0).A1
            top_indices = avg_scores.argsort()[::-1][:top_n]

            keywords = [(feature_names[i], avg_scores[i]) for i in top_indices]
            cluster_keywords[label] = keywords
        except Exception as e:
            print(f"Could not extract keywords for cluster {label}: {e}")

    return cluster_keywords


def plot_clusters_matplotlib(
    reduced: np.ndarray,
    labels: np.ndarray,
    title: str = "Topic Clusters",
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
) -> None:
    """Plot clusters using matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(
                reduced[mask, 0], reduced[mask, 1],
                c="lightgray", alpha=0.1, s=1, label="Noise",
            )
        else:
            ax.scatter(
                reduced[mask, 0], reduced[mask, 1],
                c=[cmap(i)], alpha=0.5, s=3, label=f"Cluster {label}",
            )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Show legend only if few clusters
    if len(unique_labels) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_clusters_interactive(
    reduced: np.ndarray,
    labels: np.ndarray,
    titles: Optional[list[str]] = None,
    title: str = "Topic Clusters (Interactive)",
) -> None:
    """Plot clusters using plotly for interactive exploration."""
    df_plot = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "cluster": labels.astype(str),
    })

    if titles is not None:
        df_plot["title"] = [t[:80] + "..." if len(t) > 80 else t for t in titles]
        hover_data = ["title"]
    else:
        hover_data = None

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="cluster",
        hover_data=hover_data,
        title=title,
        opacity=0.5,
        width=1000,
        height=700,
    )

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        template="plotly_dark",
        legend_title="Cluster",
    )
    fig.show()


def summarize_clusters(
    labels: np.ndarray,
    cluster_keywords: dict,
) -> pd.DataFrame:
    """
    Create a summary table of clusters.

    Returns DataFrame with: cluster_id, size, percentage, top_keywords
    """
    total = len(labels)
    records = []

    for label in sorted(set(labels)):
        count = (labels == label).sum()
        keywords = cluster_keywords.get(label, [])
        keyword_str = ", ".join([k for k, _ in keywords[:5]])

        records.append({
            "cluster": label if label != -1 else "Noise",
            "size": count,
            "percentage": f"{count/total*100:.1f}%",
            "top_keywords": keyword_str,
        })

    return pd.DataFrame(records)
