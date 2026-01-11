"""
Evaluation Module for Customer Intelligence AI.

This module provides automatic evaluation capabilities:
- Load ground truth test questions with expected keywords
- Run evaluation against the RAG pipeline
- Calculate quality metrics (keyword recall, precision)
- Track results for regression detection

Why evaluation?
- Catch quality regressions when prompts/models change
- Quantify answer quality objectively
- Support CI/CD for AI systems
- Build confidence in production deployments
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)

# Path to ground truth file
GROUND_TRUTH_PATH = os.path.join(
    os.path.dirname(__file__),
    "ground_truth.json"
)

# Store last evaluation results
_last_evaluation: Optional[Dict[str, Any]] = None


@dataclass
class TestCase:
    """
    A single evaluation test case.
    
    Attributes:
        id: Unique identifier
        question: The question to ask
        expected_keywords: Keywords that should appear in a good answer
        description: Human-readable description of what this tests
    """
    id: str
    question: str
    expected_keywords: List[str]
    description: str = ""


@dataclass
class EvaluationResult:
    """
    Result from evaluating a single test case.
    
    Attributes:
        test_case_id: ID of the test case
        question: The question asked
        answer: The answer generated
        keywords_found: Expected keywords present in answer
        keywords_missing: Expected keywords not in answer
        keyword_recall: Fraction of expected keywords found
        passed: Whether the test passed threshold
    """
    test_case_id: str
    question: str
    answer: str
    keywords_found: List[str]
    keywords_missing: List[str]
    keyword_recall: float
    passed: bool


@dataclass
class EvaluationReport:
    """
    Complete evaluation report.
    
    Attributes:
        timestamp: When evaluation was run
        total_tests: Number of test cases
        passed_tests: Number of tests that passed
        failed_tests: Number of tests that failed
        avg_keyword_recall: Average keyword recall across all tests
        results: Individual test results
    """
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    avg_keyword_recall: float
    results: List[EvaluationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": round(self.passed_tests / self.total_tests * 100, 1) if self.total_tests > 0 else 0,
            "avg_keyword_recall": round(self.avg_keyword_recall, 3),
            "results": [
                {
                    "id": r.test_case_id,
                    "question": r.question,
                    "keyword_recall": round(r.keyword_recall, 3),
                    "passed": r.passed,
                    "keywords_found": r.keywords_found,
                    "keywords_missing": r.keywords_missing,
                }
                for r in self.results
            ]
        }


def load_ground_truth(path: str = None) -> List[TestCase]:
    """
    Load ground truth test cases from JSON file.
    
    Args:
        path: Path to ground truth file (default: built-in)
        
    Returns:
        List of TestCase objects
    """
    path = path or GROUND_TRUTH_PATH
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        test_cases = [
            TestCase(
                id=item["id"],
                question=item["question"],
                expected_keywords=item["expected_keywords"],
                description=item.get("description", ""),
            )
            for item in data
        ]
        
        logger.info(f"Loaded {len(test_cases)} test cases from {path}")
        return test_cases
        
    except FileNotFoundError:
        logger.warning(f"Ground truth file not found: {path}")
        return []
    except Exception as e:
        logger.error(f"Error loading ground truth: {e}")
        return []


def calculate_keyword_recall(
    answer: str,
    expected_keywords: List[str],
) -> tuple[float, List[str], List[str]]:
    """
    Calculate keyword recall for an answer.
    
    This measures what fraction of expected keywords appear
    in the answer (case-insensitive matching).
    
    Args:
        answer: The generated answer text
        expected_keywords: Keywords that should be present
        
    Returns:
        Tuple of (recall_score, found_keywords, missing_keywords)
    """
    if not expected_keywords:
        return 1.0, [], []
    
    answer_lower = answer.lower()
    
    found = []
    missing = []
    
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found.append(keyword)
        else:
            missing.append(keyword)
    
    recall = len(found) / len(expected_keywords)
    return recall, found, missing


class Evaluator:
    """
    Runs evaluation suite against the RAG pipeline.
    
    Example:
        >>> from app.core.rag_pipeline import get_pipeline
        >>> evaluator = Evaluator(get_pipeline())
        >>> report = evaluator.run_evaluation()
        >>> print(f"Pass rate: {report.passed_tests}/{report.total_tests}")
    """
    
    def __init__(
        self,
        pipeline=None,
        pass_threshold: float = 0.3,
    ):
        """
        Initialize the evaluator.
        
        Args:
            pipeline: RAGPipeline instance (uses global if None)
            pass_threshold: Minimum keyword recall to pass (0-1)
        """
        self.pipeline = pipeline
        self.pass_threshold = pass_threshold
        self.test_cases = load_ground_truth()
    
    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: The test case to evaluate
            
        Returns:
            EvaluationResult with metrics
        """
        # Get pipeline
        if self.pipeline is None:
            from app.core.rag_pipeline import get_pipeline
            self.pipeline = get_pipeline()
        
        # Run query
        try:
            response = self.pipeline.query(test_case.question)
            answer = response.answer
        except Exception as e:
            logger.error(f"Query failed for {test_case.id}: {e}")
            answer = ""
        
        # Calculate metrics
        recall, found, missing = calculate_keyword_recall(
            answer,
            test_case.expected_keywords,
        )
        
        passed = recall >= self.pass_threshold
        
        return EvaluationResult(
            test_case_id=test_case.id,
            question=test_case.question,
            answer=answer,
            keywords_found=found,
            keywords_missing=missing,
            keyword_recall=recall,
            passed=passed,
        )
    
    def run_evaluation(self) -> EvaluationReport:
        """
        Run full evaluation suite.
        
        Returns:
            EvaluationReport with all results
        """
        global _last_evaluation
        
        if not self.test_cases:
            logger.warning("No test cases to evaluate")
            return EvaluationReport(
                timestamp=datetime.now().isoformat(),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                avg_keyword_recall=0.0,
                results=[],
            )
        
        logger.info(f"Running evaluation with {len(self.test_cases)} test cases")
        
        results = []
        for test_case in self.test_cases:
            result = self.evaluate_single(test_case)
            results.append(result)
            
            status = "✓" if result.passed else "✗"
            logger.info(
                f"{status} {test_case.id}: recall={result.keyword_recall:.2f} "
                f"({len(result.keywords_found)}/{len(test_case.expected_keywords)})"
            )
        
        # Aggregate metrics
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        avg_recall = sum(r.keyword_recall for r in results) / len(results)
        
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            avg_keyword_recall=avg_recall,
            results=results,
        )
        
        # Store for /metrics endpoint
        _last_evaluation = report.to_dict()
        
        logger.info(
            f"Evaluation complete: {passed}/{len(results)} passed, "
            f"avg recall: {avg_recall:.2f}"
        )
        
        return report


def get_last_evaluation() -> Optional[Dict[str, Any]]:
    """Get the last evaluation results for the /metrics endpoint."""
    return _last_evaluation


def run_quick_evaluation(pipeline=None) -> Dict[str, Any]:
    """
    Convenience function to run evaluation and return results.
    
    Args:
        pipeline: RAGPipeline instance (optional)
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = Evaluator(pipeline=pipeline)
    report = evaluator.run_evaluation()
    return report.to_dict()
