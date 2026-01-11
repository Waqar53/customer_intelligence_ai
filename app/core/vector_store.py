"""
Vector Store Module for Customer Intelligence AI.

This module manages the FAISS vector index for semantic search:
- Indexing customer feedback embeddings
- Fast similarity search for relevant documents
- Persistence (save/load index to disk)

Why FAISS?
- Extremely fast similarity search (millions of vectors in milliseconds)
- Works well on CPU (no GPU required)
- Supports different index types for different use cases
- Production-tested by Facebook/Meta

Index choice:
- IndexFlatIP: Exact inner product search (best accuracy)
- Fast enough for < 1M documents
- Can upgrade to IVF/HNSW for larger datasets
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from app.config import settings
from app.core.data_ingestion import FeedbackRecord

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    A document stored in the vector database.
    
    Attributes:
        id: Unique document identifier
        text: The original text content
        source: Where this document came from
        metadata: Additional information
    """
    id: str
    text: str
    source: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(**data)
    
    @classmethod
    def from_feedback(cls, record: FeedbackRecord) -> "Document":
        """Convert a FeedbackRecord to a Document."""
        return cls(
            id=record.id,
            text=record.text,
            source=record.source,
            metadata=record.metadata,
        )


@dataclass
class SearchResult:
    """
    A single search result from the vector store.
    
    Attributes:
        document: The matched document
        score: Similarity score (higher is better for cosine similarity)
        rank: Position in search results (1-indexed)
    """
    document: Document
    score: float
    rank: int


class VectorStore:
    """
    FAISS-based vector store for semantic search.
    
    This class manages:
    - Adding documents with their embeddings
    - Searching for similar documents
    - Saving/loading the index to disk
    
    Example:
        >>> from app.core.embeddings import EmbeddingModel
        >>> embedder = EmbeddingModel()
        >>> store = VectorStore(dimension=embedder.dimension)
        >>> 
        >>> # Add documents
        >>> texts = ["Great product!", "Terrible support"]
        >>> embeddings = embedder.encode(texts)
        >>> docs = [Document(id=str(i), text=t, source="test", metadata={}) 
        ...         for i, t in enumerate(texts)]
        >>> store.add_documents(docs, embeddings)
        >>> 
        >>> # Search
        >>> query_embedding = embedder.encode("What do customers like?")
        >>> results = store.search(query_embedding, top_k=5)
    """
    
    def __init__(self, dimension: int = 384, store_path: str = None):
        """
        Initialize the vector store.
        
        Args:
            dimension: Embedding dimension (384 for MiniLM)
            store_path: Path to save/load index (default from settings)
        """
        self.dimension = dimension
        self.store_path = store_path or settings.vector_store_path
        
        # Lazy import faiss to avoid loading at module import
        import faiss
        
        # Use Inner Product index (works with normalized vectors for cosine sim)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Document storage (maps internal index to documents)
        self.documents: List[Document] = []
        
        logger.info(f"Initialized VectorStore with dimension={dimension}")
    
    @property
    def size(self) -> int:
        """Number of documents in the store."""
        return len(self.documents)
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add documents with their embeddings to the store.
        
        Args:
            documents: List of Document objects
            embeddings: numpy array of embeddings (n_docs, dimension)
            
        Returns:
            Number of documents added
            
        Raises:
            ValueError: If documents and embeddings length mismatch
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )
        
        if len(documents) == 0:
            return 0
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store document metadata
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents. Total: {self.size}")
        return len(documents)
    
    def add_feedback_records(
        self,
        records: List[FeedbackRecord],
        embeddings: np.ndarray,
    ) -> int:
        """
        Convenience method to add FeedbackRecords directly.
        
        Args:
            records: List of FeedbackRecord objects
            embeddings: numpy array of embeddings
            
        Returns:
            Number of documents added
        """
        documents = [Document.from_feedback(r) for r in records]
        return self.add_documents(documents, embeddings)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Embedding of the query (1D or 2D array)
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects, sorted by similarity
        """
        if self.size == 0:
            logger.warning("Search called on empty store")
            return []
        
        # Handle 1D input
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure correct type
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
        
        # Limit top_k to available documents
        k = min(top_k, self.size)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append(SearchResult(
                document=self.documents[idx],
                score=float(score),
                rank=rank,
            ))
        
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def clear(self) -> None:
        """Remove all documents from the store."""
        import faiss
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        logger.info("Vector store cleared")
    
    def save(self, path: str = None) -> str:
        """
        Save the index and documents to disk.
        
        Args:
            path: Directory to save to (uses self.store_path if None)
            
        Returns:
            Path where files were saved
        """
        import faiss
        
        path = path or self.store_path
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(path, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save documents as JSON
        docs_path = os.path.join(path, "documents.json")
        docs_data = [doc.to_dict() for doc in self.documents]
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Saved vector store to {path} ({self.size} documents)")
        return path
    
    def load(self, path: str = None) -> int:
        """
        Load index and documents from disk.
        
        Args:
            path: Directory to load from (uses self.store_path if None)
            
        Returns:
            Number of documents loaded
            
        Raises:
            FileNotFoundError: If index files don't exist
        """
        import faiss
        
        path = path or self.store_path
        
        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "documents.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        
        # Load documents
        with open(docs_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        self.documents = [Document.from_dict(d) for d in docs_data]
        
        logger.info(f"Loaded vector store from {path} ({self.size} documents)")
        return self.size
    
    def exists(self, path: str = None) -> bool:
        """Check if a saved index exists at the given path."""
        path = path or self.store_path
        index_path = os.path.join(path, "index.faiss")
        return os.path.exists(index_path)


def create_vector_store_from_records(
    records: List[FeedbackRecord],
    embedding_model=None,
) -> VectorStore:
    """
    Convenience function to create a populated vector store from feedback records.
    
    Args:
        records: List of FeedbackRecord objects
        embedding_model: EmbeddingModel instance (creates one if None)
        
    Returns:
        Populated VectorStore
    """
    from app.core.embeddings import EmbeddingModel
    
    if embedding_model is None:
        embedding_model = EmbeddingModel()
    
    # Generate embeddings
    texts = [r.text for r in records]
    embeddings = embedding_model.encode(texts, show_progress=len(texts) > 100)
    
    # Create and populate store
    store = VectorStore(dimension=embedding_model.dimension)
    store.add_feedback_records(records, embeddings)
    
    return store
