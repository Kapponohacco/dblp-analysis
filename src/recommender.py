"""
Content-Based Recommendation Engine
======================================
Paper recommendation using cosine similarity over title embeddings.
"""


import numpy as np
import pandas as pd

from src.embeddings import find_similar, embed_query


class ContentRecommender:
    """
    Content-based paper recommender using title embeddings.

    Supports:
    - Search by paper title (find similar papers)
    - Search by free-text query
    - Author-based recommendations
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the recommender.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with paper metadata (title, authors, year, venue)
        embeddings : np.ndarray
            Pre-computed embeddings for all papers, shape (n_papers, dim)
        model_name : str
            Sentence-transformer model name for query encoding
        """
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.model_name = model_name

        assert len(df) == embeddings.shape[0], (
            f"DataFrame ({len(df)}) and embeddings ({embeddings.shape[0]}) size mismatch!"
        )

        print(f"Recommender initialized with {len(df):,} papers, "
              f"embedding dim={embeddings.shape[1]}")

    def recommend_by_title(
        self,
        title: str,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """
        Find papers similar to a given title.

        Parameters
        ----------
        title : str
            Exact or partial title to search for
        top_k : int
            Number of recommendations
        exclude_self : bool
            Whether to exclude the exact match from results

        Returns
        -------
        pd.DataFrame
            Recommended papers with similarity scores
        """
        # Find the paper in the dataset
        mask = self.df["title"].str.contains(title, case=False, na=False)
        matches = self.df[mask]

        if len(matches) == 0:
            print(f"Title not found in dataset. Using free-text search instead.")
            return self.recommend_by_query(title, top_k=top_k)

        # Use the first match
        idx = matches.index[0]
        query_embedding = self.embeddings[idx]

        indices, scores = find_similar(query_embedding, self.embeddings, top_k=top_k + 1)

        # Exclude the query paper itself
        if exclude_self:
            keep = indices != idx
            indices = indices[keep][:top_k]
            scores = scores[keep][:top_k]

        results = self.df.iloc[indices].copy()
        results["similarity_score"] = scores
        results = results[["title", "authors", "year", "venue", "similarity_score"]]

        return results.reset_index(drop=True)

    def recommend_by_query(
        self,
        query: str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Find papers relevant to a free-text query.

        Parameters
        ----------
        query : str
            Free-text search query
        top_k : int
            Number of recommendations

        Returns
        -------
        pd.DataFrame
            Recommended papers with similarity scores
        """
        query_embedding = embed_query(query, model_name=self.model_name)
        indices, scores = find_similar(query_embedding, self.embeddings, top_k=top_k)

        results = self.df.iloc[indices].copy()
        results["similarity_score"] = scores
        results = results[["title", "authors", "year", "venue", "similarity_score"]]

        return results.reset_index(drop=True)

    def recommend_by_author(
        self,
        author_name: str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Find papers by a specific author, ranked by recency.

        Parameters
        ----------
        author_name : str
            Author name (partial match supported)
        top_k : int
            Number of results

        Returns
        -------
        pd.DataFrame
            Papers by the author
        """
        mask = self.df["authors"].apply(
            lambda authors: any(author_name.lower() in a.lower() for a in authors)
            if isinstance(authors, list) else False
        )

        results = self.df[mask].copy()
        results = results.sort_values("year", ascending=False)
        results = results[["title", "authors", "year", "venue"]].head(top_k)

        return results.reset_index(drop=True)

    def hybrid_recommend(
        self,
        title: str,
        author_weight: float = 0.3,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Hybrid recommendation combining title similarity and co-authorship.

        Parameters
        ----------
        title : str
            Paper title to find similar papers for
        author_weight : float
            Weight for author overlap bonus (0 to 1)
        top_k : int
            Number of recommendations

        Returns
        -------
        pd.DataFrame
            Recommended papers with hybrid scores
        """
        # Get more candidates for re-ranking
        candidates = self.recommend_by_title(title, top_k=top_k * 3, exclude_self=True)

        if len(candidates) == 0:
            return candidates

        # Find the query paper's authors
        mask = self.df["title"].str.contains(title, case=False, na=False)
        if mask.sum() == 0:
            return candidates.head(top_k)

        query_authors = set(self.df[mask].iloc[0]["authors"])

        # Compute author overlap bonus
        def author_bonus(authors):
            if not isinstance(authors, list) or not query_authors:
                return 0
            overlap = len(set(authors) & query_authors)
            return overlap / max(len(query_authors), 1)

        candidates["author_bonus"] = candidates["authors"].apply(author_bonus)
        candidates["hybrid_score"] = (
            (1 - author_weight) * candidates["similarity_score"]
            + author_weight * candidates["author_bonus"]
        )

        candidates = candidates.sort_values("hybrid_score", ascending=False).head(top_k)
        return candidates.reset_index(drop=True)

    def format_recommendations(self, results: pd.DataFrame, query: str = "") -> str:
        """Format recommendations as a readable string."""
        lines = []
        if query:
            lines.append(f"Recommendations for: \"{query}\"")
            lines.append("=" * 60)

        for i, row in results.iterrows():
            score = row.get("similarity_score", row.get("hybrid_score", "N/A"))
            if isinstance(score, float):
                score = f"{score:.4f}"
            authors = ", ".join(row["authors"][:3]) if isinstance(row["authors"], list) else "N/A"
            if isinstance(row["authors"], list) and len(row["authors"]) > 3:
                authors += f" (+{len(row['authors'])-3} more)"

            lines.append(f"\n{i+1}. {row['title']}")
            lines.append(f"   Authors: {authors}")
            lines.append(f"   Year: {row.get('year', 'N/A')} | Venue: {row.get('venue', 'N/A')}")
            lines.append(f"   Score: {score}")

        return "\n".join(lines)
