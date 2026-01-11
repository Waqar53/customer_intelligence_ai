"""
FastAPI Application Entry Point.

This is the main application file that:
- Creates the FastAPI app instance
- Configures middleware (CORS, error handling)
- Includes API routers
- Sets up lifespan events (startup/shutdown)

Run the application with:
    uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app import __version__
from app.config import settings
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    
    Startup:
    - Load any saved vector store
    - Initialize ML models (optional eager loading)
    
    Shutdown:
    - Save vector store to disk
    - Clean up resources
    """
    # Startup
    logger.info(f"Starting Customer Intelligence AI v{__version__}")
    logger.info(f"Environment: {settings.app_env}")
    
    # Try to load existing vector store
    from app.core.rag_pipeline import get_pipeline
    pipeline = get_pipeline()
    
    if pipeline.vector_store.exists():
        try:
            count = pipeline.load()
            logger.info(f"Loaded existing index with {count} documents")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down...")
    
    # Save vector store if we have data
    if pipeline.is_ready:
        try:
            pipeline.save()
            logger.info("Saved vector store")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")


# Create FastAPI application
app = FastAPI(
    title="Customer Intelligence AI",
    description=(
        "Production-grade AI system for analyzing customer feedback at scale.\n\n"
        "## Features\n"
        "- **Data Ingestion**: Upload CSV, TXT, or PDF files with customer feedback\n"
        "- **Semantic Search**: Find relevant feedback using embeddings\n"
        "- **RAG-powered Q&A**: Ask questions grounded in real customer data\n"
        "- **Clustering**: Automatically identify complaint themes\n"
        "- **Evaluation**: Built-in quality metrics and regression testing\n\n"
        "## Quick Start\n"
        "1. Upload feedback data via `POST /upload-data`\n"
        "2. Ask questions via `POST /query`\n"
        "3. Check health via `GET /health`"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# =========================================================
# Middleware
# =========================================================

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Error Handlers
# =========================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with clear messages."""
    errors = exc.errors()
    
    # Format error messages
    messages = []
    for error in errors:
        loc = " -> ".join(str(x) for x in error["loc"])
        messages.append(f"{loc}: {error['msg']}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Invalid request data",
            "detail": messages,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully."""
    logger.exception(f"Unhandled error: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.app_env == "development" else None,
        },
    )


# =========================================================
# Include Routers
# =========================================================

# Include the main API router
app.include_router(api_router)


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Customer Intelligence AI",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


# =========================================================
# Run with Python (for development)
# =========================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
    )
