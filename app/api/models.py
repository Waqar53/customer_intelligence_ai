"""
Pydantic Models for API Request/Response Validation.

These models define the contract between the API and its clients.
Using Pydantic ensures:
- Automatic validation of incoming data
- Clear documentation via OpenAPI/Swagger
- Type hints for IDE support
- Serialization/deserialization handling
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# =========================================================
# Request Models
# =========================================================

class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to ask about customer feedback",
        examples=["What are customers unhappy about?"],
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve for context",
    )


class ClusterRequest(BaseModel):
    """Request body for clustering analysis."""
    
    n_clusters: Optional[int] = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of clusters to create",
    )


# =========================================================
# Response Models
# =========================================================

class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    
    status: str = Field(
        description="Health status of the service",
        examples=["healthy"],
    )
    version: str = Field(
        description="Application version",
    )
    documents_indexed: int = Field(
        description="Number of documents currently indexed",
    )
    ready: bool = Field(
        description="Whether the system is ready to handle queries",
    )


class CitationModel(BaseModel):
    """A citation linking an answer to source evidence."""
    
    document_id: str = Field(description="Unique ID of the source document")
    text: str = Field(description="Excerpt from the source document")
    source: str = Field(description="Original file/source name")
    relevance_score: float = Field(description="How relevant this source is (0-1)")


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""
    
    answer: str = Field(
        description="The generated answer to the question",
    )
    citations: List[CitationModel] = Field(
        default_factory=list,
        description="Source documents used to generate the answer",
    )
    query: str = Field(
        description="The original question asked",
    )
    num_sources: int = Field(
        description="Number of sources retrieved",
    )
    tokens_used: int = Field(
        description="LLM tokens consumed for this query",
    )


class UploadResponse(BaseModel):
    """Response from the /upload-data endpoint."""
    
    success: bool = Field(
        description="Whether the upload was successful",
    )
    message: str = Field(
        description="Status message",
    )
    records_processed: int = Field(
        description="Number of feedback records processed",
    )
    total_indexed: int = Field(
        description="Total documents now in the index",
    )
    filename: str = Field(
        description="Name of the uploaded file",
    )


class ThemeModel(BaseModel):
    """A theme/cluster identified in the feedback."""
    
    id: int = Field(description="Cluster ID")
    size: int = Field(description="Number of items in this cluster")
    percentage: float = Field(description="Percentage of total feedback")
    keywords: List[str] = Field(description="Top keywords for this theme")
    examples: List[str] = Field(description="Sample feedback from this cluster")


class ClusteringResponse(BaseModel):
    """Response from clustering analysis."""
    
    success: bool
    total_feedback: int
    num_themes: int
    quality_score: float = Field(description="Silhouette score (-1 to 1)")
    themes: List[ThemeModel]


class MetricsResponse(BaseModel):
    """Response from the /metrics endpoint."""
    
    documents_indexed: int = Field(
        description="Total documents in the vector store",
    )
    queries_processed: int = Field(
        description="Total queries processed since startup",
    )
    total_tokens_used: int = Field(
        description="Total LLM tokens consumed",
    )
    evaluation_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results from evaluation suite if run",
    )


class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details",
    )
