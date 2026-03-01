"""
Helper utilities for RAG Web Crawler
"""

import re
from urllib.parse import urlparse, urljoin
from typing import Optional


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing fragments and trailing slashes
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL
        
    Examples:
        >>> normalize_url("https://example.com/page#section")
        'https://example.com/page'
        >>> normalize_url("https://example.com/page/")
        'https://example.com/page'
    """
    parsed = urlparse(url)
    # Remove fragment
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    # Remove trailing slash except for root
    if normalized.endswith('/') and parsed.path != '/':
        normalized = normalized[:-1]
    # Add query params back if they exist
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def get_domain(url: str) -> Optional[str]:
    """
    Extract the registrable domain from URL
    
    Args:
        url: Full URL
        
    Returns:
        Domain name (e.g., 'example.com') or None if invalid
        
    Examples:
        >>> get_domain("https://www.example.com/page")
        'example.com'
        >>> get_domain("https://api.example.com/v1/data")
        'example.com'
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        
        # Remove port if present
        if ':' in hostname:
            hostname = hostname.split(':')[0]
        
        # Remove 'www.' prefix
        if hostname.startswith('www.'):
            hostname = hostname[4:]
        
        return hostname
    except Exception:
        return None


def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs belong to the same domain
    
    Args:
        url1: First URL
        url2: Second URL
        
    Returns:
        True if same domain, False otherwise
    """
    domain1 = get_domain(url1)
    domain2 = get_domain(url2)
    return domain1 == domain2 if domain1 and domain2 else False


def is_valid_url(url: str) -> bool:
    """
    Check if URL is valid and uses http/https
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


if __name__ == "__main__":
    # Test utilities
    print(normalize_url("https://example.com/page#section"))
    print(get_domain("https://www.example.com/page"))
    print(is_same_domain("https://example.com", "https://www.example.com/page"))
    print(is_valid_url("https://example.com"))
    print(clean_text("  Multiple   spaces   here  "))
    print(truncate_text("This is a very long text that needs to be truncated", 30))
