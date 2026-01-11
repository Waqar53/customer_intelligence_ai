"""
API Routes for Customer Intelligence AI.

This module defines all REST API endpoints:
- POST /upload-data: Upload customer feedback files
- POST /query: Ask questions about feedback
- POST /cluster: Run clustering analysis
- GET /health: Health check
- GET /metrics: Application metrics
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from app import __version__
from app.config import settings
from app.api.models import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    HealthResponse,
    MetricsResponse,
    ClusterRequest,
    ClusteringResponse,
    ThemeModel,
    ErrorResponse,
    CitationModel,
)
from app.core.data_ingestion import ingest_file, DataIngestionError
from app.core.data_cleaning import preprocess_feedback
from app.core.rag_pipeline import get_pipeline, RAGPipeline
from app.core.clustering import FeedbackClusterer, get_complaint_summary

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Track metrics
_metrics = {
    "queries_processed": 0,
    "total_tokens_used": 0,
    "files_uploaded": 0,
}


def get_rag_pipeline() -> RAGPipeline:
    """Dependency to get the RAG pipeline."""
    return get_pipeline()


# =========================================================
# Health Check
# =========================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Check the health status of the service.
    
    Returns:
        - status: "healthy" if all systems operational
        - version: Current application version
        - documents_indexed: Number of documents in the index
        - ready: Whether the system can handle queries
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        documents_indexed=pipeline.vector_store.size if pipeline.is_ready else 0,
        ready=pipeline.is_ready,
    )


# =========================================================
# Data Upload
# =========================================================

@router.post(
    "/upload-data",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Data"],
    summary="Upload customer feedback data",
)
async def upload_data(
    file: UploadFile = File(..., description="CSV, TXT, or PDF file with feedback"),
    clear_existing: bool = False,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    Upload a file containing customer feedback.
    
    Supported formats:
    - **CSV**: Must have a column with feedback text (auto-detected)
    - **TXT**: Plain text, one feedback per line
    - **PDF**: Text extracted from all pages
    
    The file will be processed, cleaned, and indexed for semantic search.
    
    Args:
        file: The file to upload
        clear_existing: If true, clear existing data before indexing
        
    Returns:
        Upload status with record counts
    """
    # Validate file type
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in [".csv", ".txt", ".pdf"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use CSV, TXT, or PDF.",
        )
    
    try:
        # Save uploaded file temporarily
        upload_dir = settings.upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        
        temp_path = os.path.join(upload_dir, filename)
        content = await file.read()
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file: {temp_path}")
        
        # Ingest and process the file
        records = ingest_file(temp_path)
        cleaned_records = preprocess_feedback(records)
        
        if not cleaned_records:
            raise HTTPException(
                status_code=400,
                detail="No valid feedback records found in the file.",
            )
        
        # Index the records
        count = pipeline.index_feedback(cleaned_records, clear_existing=clear_existing)
        
        # Update metrics
        _metrics["files_uploaded"] += 1
        
        return UploadResponse(
            success=True,
            message=f"Successfully processed {count} feedback records",
            records_processed=count,
            total_indexed=pipeline.vector_store.size,
            filename=filename,
        )
        
    except DataIngestionError as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# =========================================================
# Query
# =========================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Intelligence"],
    summary="Ask a question about customer feedback",
)
async def query_feedback(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    Ask a question about the indexed customer feedback.
    
    Uses RAG (Retrieval-Augmented Generation) to:
    1. Find relevant feedback using semantic search
    2. Generate an answer using LLM with retrieved context
    3. Include citations to source documents
    
    Example questions:
    - "What are customers complaining about?"
    - "What issues increased this month?"
    - "Summarize problems with the mobile app"
    
    Args:
        request: Query request with question and optional parameters
        
    Returns:
        Answer with citations to source feedback
    """
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No data indexed. Please upload feedback data first using /upload-data.",
        )
    
    try:
        # Run RAG pipeline
        rag_response = pipeline.query(
            question=request.question,
            top_k=request.top_k,
        )
        
        # Update metrics
        _metrics["queries_processed"] += 1
        _metrics["total_tokens_used"] += rag_response.tokens_used
        
        # Convert to API response
        return QueryResponse(
            answer=rag_response.answer,
            citations=[
                CitationModel(
                    document_id=c.document_id,
                    text=c.text[:300] + "..." if len(c.text) > 300 else c.text,
                    source=c.source,
                    relevance_score=round(c.score, 3),
                )
                for c in rag_response.citations
            ],
            query=rag_response.query,
            num_sources=rag_response.num_sources,
            tokens_used=rag_response.tokens_used,
        )
        
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# =========================================================
# Clustering
# =========================================================

@router.post(
    "/cluster",
    response_model=ClusteringResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["Intelligence"],
    summary="Analyze complaints using clustering",
)
async def cluster_feedback(
    request: ClusterRequest = None,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """
    Run clustering analysis on indexed feedback.
    
    Uses K-Means clustering on TF-IDF vectors to identify
    common themes and complaint categories.
    
    Args:
        request: Clustering parameters
        
    Returns:
        Identified themes with keywords and examples
    """
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No data indexed. Please upload feedback data first.",
        )
    
    try:
        from app.core.data_ingestion import FeedbackRecord
        
        # Get documents from vector store
        documents = pipeline.vector_store.documents
        n_clusters = request.n_clusters if request else 5
        
        if len(documents) < n_clusters:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {n_clusters} documents for clustering. "
                       f"Currently have {len(documents)}.",
            )
        
        # Convert to FeedbackRecords
        records = [
            FeedbackRecord(
                id=doc.id,
                text=doc.text,
                source=doc.source,
                metadata=doc.metadata,
            )
            for doc in documents
        ]
        
        # Run clustering
        clusterer = FeedbackClusterer(n_clusters=n_clusters)
        result = clusterer.fit(records)
        summary = get_complaint_summary(result)
        
        return ClusteringResponse(
            success=True,
            total_feedback=summary["total_feedback"],
            num_themes=summary["num_themes"],
            quality_score=summary["quality_score"],
            themes=[
                ThemeModel(
                    id=t["id"],
                    size=t["size"],
                    percentage=t["percentage"],
                    keywords=t["keywords"],
                    examples=t["examples"],
                )
                for t in summary["themes"]
            ],
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Clustering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


# =========================================================
# Metrics
# =========================================================

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["System"],
    summary="Get application metrics",
)
async def get_metrics(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Get application metrics and statistics.
    
    Includes:
    - Document counts
    - Query statistics
    - Token usage
    - Evaluation results (if available)
    """
    # Get evaluation results if available
    evaluation_results = None
    try:
        from app.evaluation.evaluator import get_last_evaluation
        evaluation_results = get_last_evaluation()
    except Exception:
        pass
    
    return MetricsResponse(
        documents_indexed=pipeline.vector_store.size if pipeline.is_ready else 0,
        queries_processed=_metrics["queries_processed"],
        total_tokens_used=_metrics["total_tokens_used"],
        evaluation_results=evaluation_results,
    )
