"""
Data Cleaning Module for Customer Intelligence AI.

This module preprocesses raw customer feedback text to improve
the quality of ML models and search results.

Cleaning steps:
1. Normalize whitespace and line breaks
2. Remove special characters (but keep punctuation for meaning)
3. Fix common encoding issues
4. Standardize case for consistent processing

Why clean text?
- Reduces noise in embeddings
- Improves clustering quality
- Better search matching
- Prevents encoding issues in downstream processing
"""

import re
import logging
from typing import List
import unicodedata

from app.core.data_ingestion import FeedbackRecord

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize a single text string.
    
    This function applies multiple cleaning steps while preserving
    the semantic meaning of the text.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text string
        
    Example:
        >>> clean_text("  Hello\\n\\nWorld!!!   ")
        'Hello World!!!'
    """
    if not text:
        return ""
    
    # Step 1: Normalize unicode characters
    # Converts characters like "café" → "cafe" for consistency
    text = unicodedata.normalize("NFKD", text)
    
    # Step 2: Remove null bytes and control characters
    # These can cause issues in databases and APIs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    
    # Step 3: Normalize whitespace
    # Replace multiple spaces, tabs, newlines with single space
    text = re.sub(r"\s+", " ", text)
    
    # Step 4: Remove leading/trailing whitespace
    text = text.strip()
    
    # Step 5: Fix common encoding artifacts
    # These often appear when copying from Word or web pages
    replacements = {
        """: '"',
        """: '"',
        "'": "'",
        "'": "'",
        "–": "-",
        "—": "-",
        "…": "...",
        "\u200b": "",  # Zero-width space
        "\ufeff": "",  # Byte order mark
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def clean_for_embedding(text: str) -> str:
    """
    Prepare text specifically for embedding generation.
    
    This is more aggressive than basic cleaning - it removes
    elements that add noise to semantic embeddings.
    
    Args:
        text: Text to prepare for embedding
        
    Returns:
        Cleaned text optimized for embedding
    """
    text = clean_text(text)
    
    # Remove URLs - they don't add semantic meaning
    text = re.sub(r"https?://\S+", "", text)
    
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    
    # Remove excessive punctuation (keep 1-2 for emphasis)
    text = re.sub(r"([!?.]){3,}", r"\1\1", text)
    
    # Remove hash symbols from hashtags but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # Remove @ mentions but keep the name
    text = re.sub(r"@(\w+)", r"\1", text)
    
    # Normalize whitespace again after removals
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def clean_for_clustering(text: str) -> str:
    """
    Prepare text for TF-IDF vectorization and clustering.
    
    This cleaning is focused on standardizing text for
    statistical analysis while preserving key terms.
    
    Args:
        text: Text to prepare for clustering
        
    Returns:
        Cleaned text optimized for TF-IDF
    """
    text = clean_text(text)
    
    # Convert to lowercase for consistent term matching
    text = text.lower()
    
    # Remove all punctuation for cleaner token matching
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Remove single character words (usually not meaningful)
    text = re.sub(r"\b\w\b", "", text)
    
    # Remove numbers (usually not helpful for topic clustering)
    text = re.sub(r"\b\d+\b", "", text)
    
    # Normalize whitespace after removals
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def preprocess_feedback(records: List[FeedbackRecord]) -> List[FeedbackRecord]:
    """
    Clean all feedback records for processing.
    
    This function applies basic cleaning to each record,
    updating the text in place. More specialized cleaning
    (for embeddings or clustering) should be done at
    the point of use.
    
    Args:
        records: List of FeedbackRecord objects
        
    Returns:
        List of cleaned FeedbackRecord objects
        
    Example:
        >>> records = load_csv("feedback.csv")
        >>> cleaned = preprocess_feedback(records)
        >>> print(cleaned[0].text)  # Now cleaned
    """
    cleaned_records = []
    skipped = 0
    
    for record in records:
        cleaned_text = clean_text(record.text)
        
        # Skip records that become empty after cleaning
        if not cleaned_text or len(cleaned_text) < 5:
            skipped += 1
            continue
        
        # Create new record with cleaned text
        cleaned_records.append(FeedbackRecord(
            id=record.id,
            text=cleaned_text,
            source=record.source,
            metadata=record.metadata,
        ))
    
    if skipped > 0:
        logger.info(f"Skipped {skipped} records (empty after cleaning)")
    
    logger.info(f"Preprocessed {len(cleaned_records)} feedback records")
    return cleaned_records


def validate_feedback(records: List[FeedbackRecord]) -> dict:
    """
    Validate a batch of feedback records and return quality metrics.
    
    Useful for understanding data quality before processing.
    
    Args:
        records: List of FeedbackRecord objects to validate
        
    Returns:
        Dictionary with validation metrics
    """
    if not records:
        return {"valid": False, "error": "No records provided"}
    
    total = len(records)
    text_lengths = [len(r.text) for r in records]
    empty_count = sum(1 for r in records if not r.text.strip())
    
    return {
        "valid": True,
        "total_records": total,
        "empty_records": empty_count,
        "avg_text_length": sum(text_lengths) / total,
        "min_text_length": min(text_lengths),
        "max_text_length": max(text_lengths),
        "sources": list(set(r.source for r in records)),
    }
