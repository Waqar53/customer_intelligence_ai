"""
Data Ingestion Module for Customer Intelligence AI.

This module handles loading customer feedback from various file formats:
- CSV files (structured feedback data)
- Plain text files (unstructured feedback)
- PDF files (documents, reports)

Why separate loaders?
- Single Responsibility: Each loader handles one format
- Easy to extend: Add new formats by adding new loaders
- Testable: Each loader can be unit tested independently
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """
    Represents a single piece of customer feedback.
    
    Attributes:
        id: Unique identifier for the feedback
        text: The actual feedback content
        source: Where this feedback came from (filename)
        metadata: Additional information (date, customer_id, etc.)
    """
    id: str
    text: str
    source: str
    metadata: Dict[str, Any]


class DataIngestionError(Exception):
    """Custom exception for data ingestion failures."""
    pass


def load_csv(file_path: str, text_column: str = "feedback") -> List[FeedbackRecord]:
    """
    Load customer feedback from a CSV file.
    
    This function expects a CSV with at least one column containing
    the feedback text. It will try common column names if the 
    specified column is not found.
    
    Args:
        file_path: Path to the CSV file
        text_column: Name of the column containing feedback text
        
    Returns:
        List of FeedbackRecord objects
        
    Raises:
        DataIngestionError: If file cannot be loaded or parsed
        
    Example:
        >>> records = load_csv("feedback.csv", text_column="comment")
        >>> print(len(records))
        150
    """
    try:
        # Read CSV with pandas
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
        
        # Try to find the text column
        possible_columns = [
            text_column,
            "feedback",
            "comment",
            "text",
            "review",
            "message",
            "content",
            "description",
        ]
        
        text_col = None
        for col in possible_columns:
            # Case-insensitive matching
            matching = [c for c in df.columns if c.lower() == col.lower()]
            if matching:
                text_col = matching[0]
                break
        
        if text_col is None:
            # Fall back to first string column
            string_cols = df.select_dtypes(include=["object"]).columns
            if len(string_cols) > 0:
                text_col = string_cols[0]
                logger.warning(f"Text column not found, using first string column: {text_col}")
            else:
                raise DataIngestionError(
                    f"Could not find text column in CSV. Columns: {list(df.columns)}"
                )
        
        # Convert rows to FeedbackRecord objects
        records = []
        source = Path(file_path).name
        
        for idx, row in df.iterrows():
            text = str(row[text_col]).strip()
            
            # Skip empty or NaN values
            if not text or text.lower() == "nan":
                continue
            
            # Collect all other columns as metadata
            metadata = {
                col: row[col] 
                for col in df.columns 
                if col != text_col and pd.notna(row[col])
            }
            
            records.append(FeedbackRecord(
                id=f"{source}_{idx}",
                text=text,
                source=source,
                metadata=metadata,
            ))
        
        logger.info(f"Extracted {len(records)} feedback records from {source}")
        return records
        
    except pd.errors.EmptyDataError:
        raise DataIngestionError(f"CSV file is empty: {file_path}")
    except Exception as e:
        raise DataIngestionError(f"Failed to load CSV: {str(e)}")


def load_text(file_path: str) -> List[FeedbackRecord]:
    """
    Load feedback from a plain text file.
    
    The file is treated as one document. For files with multiple
    feedback entries, each line is treated as a separate record
    if the file has more than 5 non-empty lines.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of FeedbackRecord objects
        
    Raises:
        DataIngestionError: If file cannot be read
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        source = Path(file_path).name
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        # If file has many lines, treat each as separate feedback
        if len(lines) > 5:
            records = [
                FeedbackRecord(
                    id=f"{source}_{i}",
                    text=line,
                    source=source,
                    metadata={"line_number": i + 1},
                )
                for i, line in enumerate(lines)
            ]
        else:
            # Small file = single document
            records = [
                FeedbackRecord(
                    id=f"{source}_0",
                    text=content.strip(),
                    source=source,
                    metadata={},
                )
            ]
        
        logger.info(f"Extracted {len(records)} feedback records from {source}")
        return records
        
    except UnicodeDecodeError:
        raise DataIngestionError(f"File encoding not supported: {file_path}")
    except Exception as e:
        raise DataIngestionError(f"Failed to load text file: {str(e)}")


def load_pdf(file_path: str) -> List[FeedbackRecord]:
    """
    Extract text from a PDF file.
    
    Each page is treated as a separate feedback record,
    allowing for better granularity in search and analysis.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of FeedbackRecord objects (one per page)
        
    Raises:
        DataIngestionError: If PDF cannot be parsed
    """
    try:
        reader = PdfReader(file_path)
        source = Path(file_path).name
        records = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            if text and text.strip():
                records.append(FeedbackRecord(
                    id=f"{source}_page{page_num + 1}",
                    text=text.strip(),
                    source=source,
                    metadata={"page": page_num + 1},
                ))
        
        if not records:
            logger.warning(f"No text extracted from PDF: {file_path}")
        else:
            logger.info(f"Extracted {len(records)} pages from {source}")
        
        return records
        
    except Exception as e:
        raise DataIngestionError(f"Failed to parse PDF: {str(e)}")


def ingest_file(file_path: str) -> List[FeedbackRecord]:
    """
    Auto-detect file type and load feedback using appropriate loader.
    
    This is the main entry point for file ingestion. It determines
    the file type from the extension and routes to the correct loader.
    
    Args:
        file_path: Path to the file to ingest
        
    Returns:
        List of FeedbackRecord objects
        
    Raises:
        DataIngestionError: If file type is unsupported or loading fails
        
    Example:
        >>> records = ingest_file("data/feedback.csv")
        >>> records = ingest_file("data/report.pdf")
        >>> records = ingest_file("data/notes.txt")
    """
    if not os.path.exists(file_path):
        raise DataIngestionError(f"File not found: {file_path}")
    
    # Determine file type from extension
    ext = Path(file_path).suffix.lower()
    
    loaders = {
        ".csv": load_csv,
        ".txt": load_text,
        ".pdf": load_pdf,
    }
    
    if ext not in loaders:
        raise DataIngestionError(
            f"Unsupported file type: {ext}. Supported: {list(loaders.keys())}"
        )
    
    logger.info(f"Ingesting file: {file_path} (type: {ext})")
    return loaders[ext](file_path)


def ingest_directory(dir_path: str) -> List[FeedbackRecord]:
    """
    Ingest all supported files from a directory.
    
    Useful for batch loading multiple feedback files at once.
    
    Args:
        dir_path: Path to directory containing feedback files
        
    Returns:
        Combined list of FeedbackRecord objects from all files
    """
    all_records = []
    supported_extensions = {".csv", ".txt", ".pdf"}
    
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        
        if os.path.isfile(file_path):
            ext = Path(file_name).suffix.lower()
            if ext in supported_extensions:
                try:
                    records = ingest_file(file_path)
                    all_records.extend(records)
                except DataIngestionError as e:
                    logger.error(f"Skipping file {file_name}: {e}")
    
    logger.info(f"Ingested {len(all_records)} total records from {dir_path}")
    return all_records
