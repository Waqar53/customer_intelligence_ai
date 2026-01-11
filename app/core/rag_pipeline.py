"""
RAG Pipeline Module for Customer Intelligence AI.

This module orchestrates the complete Retrieval-Augmented Generation flow:
1. Process user query
2. Retrieve relevant documents from vector store
3. Build context with evidence
4. Generate grounded response with LLM
5. Return response with citations

Why RAG?
- Prevents hallucination by grounding answers in real data
- Enables answering questions about specific customer feedback
- Scales to large datasets (only retrieves relevant docs)
- Provides evidence/citations for trustworthy responses
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from app.config import settings
from app.core.embeddings import EmbeddingModel
from app.core.vector_store import VectorStore, SearchResult, Document
from app.core.llm_client import LLMClient, LLMResponse
from app.core.data_ingestion import FeedbackRecord

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """
    A citation linking a response to source evidence.
    
    Attributes:
        document_id: ID of the source document
        text: The relevant excerpt
        source: Original source file
        score: Relevance score from retrieval
    """
    document_id: str
    text: str
    source: str
    score: float


@dataclass
class RAGResponse:
    """
    Complete response from the RAG pipeline.
    
    Attributes:
        answer: The generated answer text
        citations: List of source documents used
        query: The original query
        num_sources: Number of sources retrieved
        tokens_used: Total LLM tokens consumed
        confidence: Optional confidence score
    """
    answer: str
    citations: List[Citation]
    query: str
    num_sources: int
    tokens_used: int
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "answer": self.answer,
            "citations": [
                {
                    "document_id": c.document_id,
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "source": c.source,
                    "relevance_score": round(c.score, 3),
                }
                for c in self.citations
            ],
            "query": self.query,
            "num_sources": self.num_sources,
            "tokens_used": self.tokens_used,
        }


class RAGPipeline:
    """
    Orchestrates retrieval-augmented generation for customer intelligence.
    
    This class ties together:
    - Embedding model for query encoding
    - Vector store for document retrieval
    - LLM client for answer generation
    
    Example:
        >>> pipeline = RAGPipeline()
        >>> 
        >>> # Add data
        >>> records = load_csv("feedback.csv")
        >>> pipeline.index_feedback(records)
        >>> 
        >>> # Query
        >>> response = pipeline.query("What are customers complaining about?")
        >>> print(response.answer)
        >>> print(f"Sources: {len(response.citations)}")
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        vector_store: VectorStore = None,
        llm_client: LLMClient = None,
        top_k: int = None,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Model for encoding queries and documents
            vector_store: Store for document retrieval
            llm_client: Client for LLM generation
            top_k: Number of documents to retrieve per query
        """
        self.top_k = top_k or settings.retrieval_top_k
        
        # Initialize components (lazy loading for embedding model)
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._llm_client = llm_client
        
        self._indexed = False
        logger.info("RAG Pipeline initialized")
    
    @property
    def embedding_model(self) -> EmbeddingModel:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel()
        return self._embedding_model
    
    @property
    def vector_store(self) -> VectorStore:
        """Lazy-load vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                dimension=self.embedding_model.dimension
            )
        return self._vector_store
    
    @property
    def llm_client(self) -> LLMClient:
        """Lazy-load LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    @property
    def is_ready(self) -> bool:
        """Check if the pipeline has indexed data and is ready for queries."""
        return self._indexed or (self._vector_store and self._vector_store.size > 0)
    
    def index_feedback(
        self,
        records: List[FeedbackRecord],
        clear_existing: bool = False,
    ) -> int:
        """
        Index feedback records for retrieval.
        
        Args:
            records: List of FeedbackRecord objects to index
            clear_existing: If True, clear existing index first
            
        Returns:
            Number of documents indexed
        """
        if clear_existing:
            self.vector_store.clear()
        
        if not records:
            logger.warning("No records to index")
            return 0
        
        # Generate embeddings
        texts = [r.text for r in records]
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.embedding_model.encode(texts, show_progress=len(texts) > 50)
        
        # Add to vector store
        count = self.vector_store.add_feedback_records(records, embeddings)
        self._indexed = True
        
        logger.info(f"Indexed {count} documents. Total in store: {self.vector_store.size}")
        return count
    
    def retrieve(self, query: str, top_k: int = None) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of SearchResult objects
        """
        k = top_k or self.top_k
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Search
        results = self.vector_store.search(query_embedding, top_k=k)
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
    
    def query(
        self,
        question: str,
        top_k: int = None,
    ) -> RAGResponse:
        """
        Complete RAG pipeline: retrieve and generate answer.
        
        Args:
            question: The user's question
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and citations
        """
        if not self.is_ready:
            return RAGResponse(
                answer="No data has been indexed yet. Please upload customer feedback first.",
                citations=[],
                query=question,
                num_sources=0,
                tokens_used=0,
            )
        
        logger.info(f"Processing query: {question}")
        
        # Step 1: Retrieve relevant documents
        results = self.retrieve(question, top_k=top_k)
        
        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant feedback to answer this question.",
                citations=[],
                query=question,
                num_sources=0,
                tokens_used=0,
            )
        
        # Step 2: Build citations
        citations = [
            Citation(
                document_id=r.document.id,
                text=r.document.text,
                source=r.document.source,
                score=r.score,
            )
            for r in results
        ]
        
        # Step 3: Prepare context for LLM
        context_docs = [r.document.text for r in results]
        
        # Step 4: Generate answer
        try:
            llm_response = self.llm_client.answer_question(
                question=question,
                context_documents=context_docs,
            )
            
            return RAGResponse(
                answer=llm_response.content,
                citations=citations,
                query=question,
                num_sources=len(results),
                tokens_used=llm_response.total_tokens,
            )
        except ValueError as e:
            # API key not configured
            logger.warning(f"LLM generation failed: {e}")
            return RAGResponse(
                answer=(
                    "LLM is not configured. Based on the retrieved feedback, here are "
                    f"the top {len(results)} relevant items:\n\n" +
                    "\n\n".join(f"• {r.document.text[:200]}..." for r in results[:3])
                ),
                citations=citations,
                query=question,
                num_sources=len(results),
                tokens_used=0,
            )
    
    def get_themes_summary(self) -> str:
        """
        Get a summary of themes in the indexed data.
        
        Uses clustering + LLM to summarize main themes.
        """
        if not self.is_ready:
            return "No data indexed."
        
        # Get sample documents for summary
        sample_texts = [doc.text for doc in self.vector_store.documents[:20]]
        
        try:
            response = self.llm_client.summarize(
                feedback_items=sample_texts,
                focus="main customer complaints and themes"
            )
            return response.content
        except ValueError:
            return f"Based on {self.vector_store.size} indexed documents."
    
    def save(self, path: str = None) -> str:
        """Save the vector store to disk."""
        return self.vector_store.save(path)
    
    def load(self, path: str = None) -> int:
        """Load vector store from disk."""
        count = self.vector_store.load(path)
        self._indexed = count > 0
        return count


# Global pipeline instance for the application
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """
    Get the global RAG pipeline instance.
    
    Creates one if it doesn't exist. Used by the API endpoints.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def reset_pipeline() -> None:
    """Reset the global pipeline (useful for testing)."""
    global _pipeline
    _pipeline = None
