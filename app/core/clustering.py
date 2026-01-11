"""
ML Clustering Module for Customer Intelligence AI.

This module implements complaint clustering using classical ML techniques:
- TF-IDF vectorization for text representation
- K-Means clustering for grouping similar complaints
- Theme extraction to identify top issues

Why clustering?
- Automatically group similar complaints together
- Discover themes without manual labeling
- Identify top issues at scale
- Track how complaint types change over time

Technical choices:
- TF-IDF: Classic, interpretable, works well for topic discovery
- K-Means: Fast, scalable, produces clear clusters
- Silhouette score: Helps determine optimal cluster count
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.core.data_ingestion import FeedbackRecord
from app.core.data_cleaning import clean_for_clustering
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """
    Represents the result of clustering analysis.
    
    Attributes:
        cluster_id: Numeric identifier for the cluster
        size: Number of documents in the cluster
        top_keywords: Most representative terms for this cluster
        sample_documents: Example documents from this cluster
        theme_label: Human-readable theme description (optional)
    """
    cluster_id: int
    size: int
    top_keywords: List[str]
    sample_documents: List[str]
    theme_label: Optional[str] = None


@dataclass
class ClusteringOutput:
    """
    Complete output from a clustering run.
    
    Attributes:
        clusters: List of ClusterResult objects
        labels: Cluster assignment for each input document
        silhouette: Quality score (-1 to 1, higher is better)
        n_documents: Total documents processed
    """
    clusters: List[ClusterResult]
    labels: List[int]
    silhouette: float
    n_documents: int


class FeedbackClusterer:
    """
    Clusters customer feedback to identify themes and common complaints.
    
    This class handles the full pipeline:
    1. Text preprocessing for ML
    2. TF-IDF vectorization
    3. K-Means clustering
    4. Theme extraction from clusters
    
    Example:
        >>> clusterer = FeedbackClusterer(n_clusters=5)
        >>> records = load_csv("feedback.csv")
        >>> result = clusterer.fit(records)
        >>> for cluster in result.clusters:
        ...     print(f"Theme: {cluster.top_keywords}")
    """
    
    def __init__(
        self,
        n_clusters: int = None,
        max_features: int = 1000,
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        """
        Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters (None = auto-detect)
            max_features: Maximum vocabulary size for TF-IDF
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
        """
        self.n_clusters = n_clusters or settings.default_num_clusters
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize vectorizer with English stop words removed
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
            ngram_range=(1, 2),  # Include bigrams for better themes
        )
        
        self.kmeans = None
        self.feature_names = None
        self._fitted = False
    
    def fit(self, records: List[FeedbackRecord]) -> ClusteringOutput:
        """
        Cluster the feedback records and extract themes.
        
        Args:
            records: List of FeedbackRecord objects to cluster
            
        Returns:
            ClusteringOutput with cluster information
            
        Raises:
            ValueError: If not enough records for clustering
        """
        if len(records) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} records for {self.n_clusters} clusters. "
                f"Got {len(records)}."
            )
        
        # Preprocess texts for clustering
        texts = [clean_for_clustering(r.text) for r in records]
        original_texts = [r.text for r in records]
        
        # Filter out empty texts after cleaning
        valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) > 0]
        texts = [texts[i] for i in valid_indices]
        original_texts = [original_texts[i] for i in valid_indices]
        
        if len(texts) < self.n_clusters:
            raise ValueError(f"Only {len(texts)} valid texts after cleaning")
        
        logger.info(f"Clustering {len(texts)} feedback records into {self.n_clusters} clusters")
        
        # Step 1: TF-IDF Vectorization
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.debug(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Step 2: K-Means Clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        labels = self.kmeans.fit_predict(tfidf_matrix)
        
        # Step 3: Calculate clustering quality
        if len(set(labels)) > 1:
            sil_score = silhouette_score(tfidf_matrix, labels)
        else:
            sil_score = 0.0
        
        logger.info(f"Clustering complete. Silhouette score: {sil_score:.3f}")
        
        # Step 4: Extract cluster information
        clusters = self._extract_cluster_info(
            texts=original_texts,
            cleaned_texts=texts,
            labels=labels,
            tfidf_matrix=tfidf_matrix,
        )
        
        self._fitted = True
        
        return ClusteringOutput(
            clusters=clusters,
            labels=labels.tolist(),
            silhouette=sil_score,
            n_documents=len(texts),
        )
    
    def _extract_cluster_info(
        self,
        texts: List[str],
        cleaned_texts: List[str],
        labels: np.ndarray,
        tfidf_matrix,
    ) -> List[ClusterResult]:
        """Extract detailed information about each cluster."""
        clusters = []
        
        for cluster_id in range(self.n_clusters):
            # Get indices of documents in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Get top keywords based on centroid
            centroid = self.kmeans.cluster_centers_[cluster_id]
            top_indices = centroid.argsort()[-10:][::-1]
            top_keywords = [self.feature_names[i] for i in top_indices]
            
            # Get sample documents (up to 3)
            sample_indices = cluster_indices[:3]
            sample_docs = [texts[i] for i in sample_indices]
            
            clusters.append(ClusterResult(
                cluster_id=cluster_id,
                size=len(cluster_indices),
                top_keywords=top_keywords,
                sample_documents=sample_docs,
                theme_label=None,  # Can be set later or by LLM
            ))
        
        # Sort by size (largest first)
        clusters.sort(key=lambda c: c.size, reverse=True)
        
        return clusters
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Assign new texts to existing clusters.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            List of cluster IDs
        """
        if not self._fitted:
            raise ValueError("Clusterer must be fit before predicting")
        
        cleaned = [clean_for_clustering(t) for t in texts]
        tfidf = self.vectorizer.transform(cleaned)
        return self.kmeans.predict(tfidf).tolist()


def find_optimal_clusters(
    records: List[FeedbackRecord],
    min_k: int = 2,
    max_k: int = 10,
) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using silhouette score.
    
    This helps when you don't know how many themes exist in the data.
    
    Args:
        records: Feedback records to analyze
        min_k: Minimum clusters to try
        max_k: Maximum clusters to try
        
    Returns:
        Tuple of (optimal_k, list of scores for each k)
    """
    texts = [clean_for_clustering(r.text) for r in records]
    texts = [t for t in texts if t.strip()]
    
    # Limit max_k based on data size
    max_k = min(max_k, len(texts) - 1)
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    scores = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)
        scores.append(score)
        logger.debug(f"k={k}: silhouette={score:.3f}")
    
    # Find best k
    best_idx = np.argmax(scores)
    optimal_k = min_k + best_idx
    
    logger.info(f"Optimal cluster count: {optimal_k} (score: {scores[best_idx]:.3f})")
    
    return optimal_k, scores


def get_complaint_summary(clustering_output: ClusteringOutput) -> Dict[str, Any]:
    """
    Generate a structured summary of complaints from clustering results.
    
    This is useful for quick dashboard views or API responses.
    
    Args:
        clustering_output: Result from FeedbackClusterer.fit()
        
    Returns:
        Dictionary with summary information
    """
    themes = []
    for cluster in clustering_output.clusters:
        themes.append({
            "id": cluster.cluster_id,
            "size": cluster.size,
            "percentage": round(cluster.size / clustering_output.n_documents * 100, 1),
            "keywords": cluster.top_keywords[:5],
            "examples": cluster.sample_documents[:2],
        })
    
    return {
        "total_feedback": clustering_output.n_documents,
        "num_themes": len(clustering_output.clusters),
        "quality_score": round(clustering_output.silhouette, 3),
        "themes": themes,
    }
