#!/usr/bin/env python3
"""
Utility functions for ABM LLM simulation.
"""

import re

def token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string for a given model.
    
    This is a simplified token counting function that approximates
    the number of tokens by counting words and characters.
    
    Args:
        text: Input text to count tokens for
        model: Model name to use for tokenization (default: gpt-3.5-turbo)
        
    Returns:
        Number of tokens in the text (approximated)
    """
    if not text:
        return 0
    
    # Simple approximation: count words and add some overhead for special tokens
    # This is a rough estimate - for exact counting, you'd need tiktoken or similar
    words = len(text.split())
    chars = len(text)
    
    # Rough approximation: 1 token ≈ 4 characters or 0.75 words
    token_estimate = max(chars // 4, int(words * 1.33))
    
    return token_estimate

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()

def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to a maximum length while trying to preserve word boundaries.
    
    Args:
        text: Input text to truncate
        max_length: Maximum length in characters
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can find a space reasonably close to the end
        return truncated[:last_space] + "..."
    else:
        return truncated + "..." 