"""
Tests for Data Ingestion Module.

Run with: pytest tests/test_ingestion.py -v
"""

import os
import tempfile
import pytest
from app.core.data_ingestion import (
    load_csv,
    load_text,
    ingest_file,
    DataIngestionError,
    FeedbackRecord,
)
from app.core.data_cleaning import (
    clean_text,
    clean_for_embedding,
    clean_for_clustering,
    preprocess_feedback,
)


class TestLoadCSV:
    """Tests for CSV loading functionality."""
    
    def test_load_csv_with_feedback_column(self):
        """Test loading CSV with standard 'feedback' column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,feedback,date\n")
            f.write("1,Great product!,2024-01-01\n")
            f.write("2,Needs improvement,2024-01-02\n")
            temp_path = f.name
        
        try:
            records = load_csv(temp_path)
            assert len(records) == 2
            assert records[0].text == "Great product!"
            assert records[1].text == "Needs improvement"
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_with_comment_column(self):
        """Test loading CSV with 'comment' column (alternate name)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,comment\n")
            f.write("1,This is a comment\n")
            temp_path = f.name
        
        try:
            records = load_csv(temp_path)
            assert len(records) == 1
            assert records[0].text == "This is a comment"
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_empty_file(self):
        """Test handling of empty CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            with pytest.raises(DataIngestionError):
                load_csv(temp_path)
        finally:
            os.unlink(temp_path)


class TestLoadText:
    """Tests for text file loading functionality."""
    
    def test_load_single_document(self):
        """Test loading small text file as single document."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is feedback.\nIt has two lines.")
            temp_path = f.name
        
        try:
            records = load_text(temp_path)
            assert len(records) == 1
            assert "feedback" in records[0].text.lower()
        finally:
            os.unlink(temp_path)
    
    def test_load_multi_line_as_separate(self):
        """Test loading large text file with each line as separate record."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for i in range(10):
                f.write(f"Feedback line {i}\n")
            temp_path = f.name
        
        try:
            records = load_text(temp_path)
            assert len(records) == 10
        finally:
            os.unlink(temp_path)


class TestIngestFile:
    """Tests for auto-detect file ingestion."""
    
    def test_ingest_csv(self):
        """Test auto-detection of CSV files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("feedback\nTest feedback\n")
            temp_path = f.name
        
        try:
            records = ingest_file(temp_path)
            assert len(records) == 1
        finally:
            os.unlink(temp_path)
    
    def test_ingest_unsupported_type(self):
        """Test handling of unsupported file types."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("data")
            temp_path = f.name
        
        try:
            with pytest.raises(DataIngestionError, match="Unsupported file type"):
                ingest_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_ingest_missing_file(self):
        """Test handling of missing files."""
        with pytest.raises(DataIngestionError, match="File not found"):
            ingest_file("/nonexistent/path/file.csv")


class TestTextCleaning:
    """Tests for text cleaning functions."""
    
    def test_clean_text_whitespace(self):
        """Test whitespace normalization."""
        result = clean_text("  Hello   World  \n\n Test  ")
        assert result == "Hello World Test"
    
    def test_clean_text_special_chars(self):
        """Test special character handling."""
        result = clean_text('Hello "World"')
        assert result == 'Hello "World"'
    
    def test_clean_for_embedding_urls(self):
        """Test URL removal for embeddings."""
        result = clean_for_embedding("Check https://example.com for info")
        assert "https" not in result
        assert "example" not in result
    
    def test_clean_for_clustering_lowercase(self):
        """Test lowercase conversion for clustering."""
        result = clean_for_clustering("HELLO World")
        assert result == "hello world"
    
    def test_preprocess_feedback_empty(self):
        """Test handling of empty feedback records."""
        records = [
            FeedbackRecord(id="1", text="Valid feedback", source="test", metadata={}),
            FeedbackRecord(id="2", text="   ", source="test", metadata={}),
        ]
        cleaned = preprocess_feedback(records)
        assert len(cleaned) == 1
        assert cleaned[0].text == "Valid feedback"
