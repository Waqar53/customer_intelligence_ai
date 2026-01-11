"""
LLM Client Module for Customer Intelligence AI.

This module provides a clean interface to OpenAI's API for:
- Text generation and reasoning
- Grounded summarization
- Question answering with context

Design principles:
- Clean abstraction over the OpenAI SDK
- Configurable via environment variables
- Proper error handling and retries
- Token usage tracking
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Structured response from the LLM.
    
    Attributes:
        content: The generated text
        model: Model used for generation
        prompt_tokens: Tokens used in the prompt
        completion_tokens: Tokens in the response
        total_tokens: Total tokens used
    """
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """
    Client for interacting with OpenAI's API.
    
    Handles:
    - API authentication
    - Message formatting
    - Response parsing
    - Error handling
    
    Example:
        >>> client = LLMClient()
        >>> response = client.generate(
        ...     "Summarize the following customer feedback:",
        ...     context="Customer 1: Great product! Customer 2: Slow support."
        ... )
        >>> print(response.content)
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the LLM client.
        
        Args:
            api_key: OpenAI API key (defaults to settings.openai_api_key)
            model: Model to use (defaults to settings.llm_model)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        
        if not self.api_key:
            logger.warning("OpenAI API key not set - LLM calls will fail")
            self._client = None
        else:
            self._client = OpenAI(api_key=self.api_key)
        
        # Track usage
        self.total_tokens_used = 0
    
    def generate(
        self,
        prompt: str,
        context: str = None,
        system_message: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user's question or instruction
            context: Optional context to include (e.g., retrieved documents)
            system_message: Optional system prompt to control behavior
            temperature: Creativity (0.0-1.0, lower = more focused)
            max_tokens: Maximum response length
            
        Returns:
            LLMResponse with the generated content
            
        Raises:
            ValueError: If API key is not configured
        """
        if not self._client:
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
            )
        
        # Build messages
        messages = []
        
        # System message sets the behavior
        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            messages.append({
                "role": "system",
                "content": (
                    "You are a customer intelligence analyst. "
                    "Your job is to analyze customer feedback and provide accurate, "
                    "evidence-based insights. Always cite specific feedback in your answers. "
                    "If you cannot answer based on the provided context, say so clearly."
                )
            })
        
        # Add context if provided
        if context:
            messages.append({
                "role": "user",
                "content": f"Here is the relevant customer feedback data:\n\n{context}"
            })
        
        # Add the main prompt
        messages.append({"role": "user", "content": prompt})
        
        logger.debug(f"Generating response with {len(messages)} messages")
        
        # Call OpenAI API
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Parse response
        content = response.choices[0].message.content
        usage = response.usage
        
        self.total_tokens_used += usage.total_tokens
        
        logger.info(f"Generated response: {usage.total_tokens} tokens used")
        
        return LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
    
    def summarize(
        self,
        feedback_items: List[str],
        focus: str = None,
    ) -> LLMResponse:
        """
        Summarize a list of customer feedback items.
        
        Args:
            feedback_items: List of feedback texts to summarize
            focus: Optional focus area (e.g., "complaints about speed")
            
        Returns:
            LLMResponse with the summary
        """
        # Format the feedback
        numbered = "\n".join(
            f"{i+1}. {item}" 
            for i, item in enumerate(feedback_items[:20])  # Limit to avoid token overflow
        )
        
        # Build prompt
        if focus:
            prompt = f"Summarize the following customer feedback, focusing on: {focus}\n"
        else:
            prompt = "Summarize the following customer feedback:\n"
        
        prompt += (
            "Identify the main themes and provide specific examples. "
            "Use direct quotes where helpful."
        )
        
        return self.generate(prompt=prompt, context=numbered)
    
    def answer_question(
        self,
        question: str,
        context_documents: List[str],
    ) -> LLMResponse:
        """
        Answer a question using provided context documents.
        
        This is the core RAG answering function.
        
        Args:
            question: The user's question
            context_documents: Relevant documents retrieved from vector search
            
        Returns:
            LLMResponse with the answer grounded in the context
        """
        if not context_documents:
            return LLMResponse(
                content="I don't have enough information to answer this question. "
                        "Please upload customer feedback data first.",
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
        
        # Format context with clear separation
        context_text = "\n\n---\n\n".join(
            f"Feedback {i+1}: {doc}"
            for i, doc in enumerate(context_documents)
        )
        
        prompt = (
            f"Based on the customer feedback provided, answer this question:\n\n"
            f"**Question:** {question}\n\n"
            f"Instructions:\n"
            f"- Base your answer ONLY on the provided feedback\n"
            f"- Include specific quotes as evidence\n"
            f"- If the answer cannot be determined from the data, say so\n"
            f"- Structure your answer clearly"
        )
        
        return self.generate(
            prompt=prompt,
            context=context_text,
            temperature=0.2,  # Lower temperature for factual responses
        )


def get_llm_client() -> LLMClient:
    """Get a configured LLM client instance."""
    return LLMClient()
