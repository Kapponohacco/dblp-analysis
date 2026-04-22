"""
Title Embedding Generation
============================
Generate and cache sentence embeddings for paper titles.
Supports sentence-transformers and TF-IDF fallback.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def generate_embeddings_transformer(
    titles: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    show_progress: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Generate embeddings using sentence-transformers.

    Parameters
    ----------
    titles : list[str]
        List of paper titles
    model_name : str
        Sentence-transformer model name
    batch_size : int
        Encoding batch size
    show_progress : bool
        Show progress bar
    device : str, optional
        Device to use ('cuda', 'cpu'). Auto-detected if None.

    Returns
    -------
    np.ndarray
        Embeddings matrix of shape (n_titles, embedding_dim)
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    print(f"Encoding {len(titles):,} titles...")
    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    print(f"Generated embeddings: shape {embeddings.shape}")
    return embeddings


def generate_embeddings_tfidf(
    titles: list[str],
    max_features: int = 10000,
) -> np.ndarray:
    """
    Generate TF-IDF embeddings as a fallback.

    Parameters
    ----------
    titles : list[str]
        List of paper titles
    max_features : int
        Maximum vocabulary size

    Returns
    -------
    np.ndarray
        TF-IDF matrix of shape (n_titles, max_features)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"Computing TF-IDF embeddings (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
    )

    tfidf_matrix = vectorizer.fit_transform(titles)
    print(f"TF-IDF matrix: shape {tfidf_matrix.shape}")

    return tfidf_matrix.toarray(), vectorizer


def load_or_compute_embeddings(
    titles: list[str],
    cache_path: str | Path,
    method: str = "transformer",
    model_name: str = "all-MiniLM-L6-v2",
    force_recompute: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Load cached embeddings or compute and cache them.

    Parameters
    ----------
    titles : list[str]
        List of paper titles
    cache_path : str or Path
        Path to cache file (.npy)
    method : str
        'transformer' or 'tfidf'
    model_name : str
        Model name for transformer method
    force_recompute : bool
        If True, recompute even if cache exists

    Returns
    -------
    np.ndarray
        Embeddings matrix
    """
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_recompute:
        print(f"Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(titles):
            print(f"Loaded embeddings: shape {embeddings.shape}")
            return embeddings
        else:
            print(f"Cache size mismatch ({embeddings.shape[0]} vs {len(titles)}). Recomputing...")

    # Compute embeddings
    if method == "transformer":
        embeddings = generate_embeddings_transformer(titles, model_name=model_name, **kwargs)
    elif method == "tfidf":
        embeddings, _ = generate_embeddings_tfidf(titles, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'transformer' or 'tfidf'.")

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Cached embeddings to {cache_path}")

    return embeddings


def find_similar(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the top-k most similar items to a query embedding.

    Parameters
    ----------
    query_embedding : np.ndarray
        Query vector of shape (1, dim) or (dim,)
    corpus_embeddings : np.ndarray
        Corpus matrix of shape (n, dim)
    top_k : int
        Number of results to return

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (indices, scores) of top-k most similar items
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices, top_scores


def embed_query(
    query: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed a single query string using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return embedding[0]
