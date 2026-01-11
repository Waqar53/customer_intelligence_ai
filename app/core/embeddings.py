"""
Embeddings Module for Customer Intelligence AI.

This module handles generating semantic embeddings for text using
sentence-transformers. Embeddings allow us to:

1. Semantic Search: Find similar feedback based on meaning, not keywords
2. RAG Retrieval: Retrieve relevant context for LLM queries
3. Similarity Analysis: Measure how similar two pieces of feedback are

Model choice:
- all-MiniLM-L6-v2: Fast, good quality, 384 dimensions
- Small enough to run on CPU
- Produces normalized embeddings (good for cosine similarity)
"""

import logging
from typing import List, Union, Optional
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy loading to avoid import overhead when not needed
_model = None


def get_embedding_model():
    """
    Lazy-load the sentence transformer model.
    
    This prevents loading the model until it's actually needed,
    speeding up application startup.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _model


class EmbeddingModel:
    """
    Wrapper for sentence-transformer embedding generation.
    
    This class provides a clean interface for generating embeddings
    and handles batch processing efficiently.
    
    Example:
        >>> embedder = EmbeddingModel()
        >>> embeddings = embedder.encode(["Great product!", "Terrible service"])
        >>> print(embeddings.shape)  # (2, 384)
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       If None, uses the default from settings.
        """
        self.model_name = model_name or settings.embedding_model
        self._model = None
    
    @property
    def model(self):
        """Lazy-load the model on first access."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension size."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single string or list of strings to embed
            normalize: If True, normalize embeddings to unit length
                      (required for cosine similarity with FAISS)
            batch_size: Batch size for processing large lists
            show_progress: If True, show progress bar
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        logger.debug(f"Encoding {len(texts)} texts")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.encode([text1, text2])
        # Dot product of normalized vectors = cosine similarity
        return float(np.dot(embeddings[0], embeddings[1]))


def encode_texts(
    texts: List[str],
    model_name: str = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convenience function to encode texts without creating an EmbeddingModel instance.
    
    Uses a cached global model for efficiency.
    
    Args:
        texts: List of texts to encode
        model_name: Model to use (default from settings)
        normalize: Whether to normalize embeddings
        
    Returns:
        numpy array of embeddings
    """
    model = get_embedding_model()
    
    if not texts:
        return np.array([])
    
    return model.encode(
        texts,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
