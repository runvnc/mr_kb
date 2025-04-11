# url_fetcher.py
import trafilatura
import logging
import asyncio
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class URLFetchError(Exception):
    """Raised when URL fetching fails."""
    pass

async def fetch_content_from_url(url: str) -> str:
    """Fetch and extract content from a URL using trafilatura.
    
    Args:
        url: The URL to fetch and extract content from
        
    Returns:
        str: The extracted main content of the webpage
        
    Raises:
        URLFetchError: If fetching or extraction fails
    """
    try:
        # Run trafilatura in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        downloaded = await loop.run_in_executor(None, trafilatura.fetch_url, url)
        
        if downloaded is None:
            raise URLFetchError(f"Failed to download content from {url}")
        
        content = await loop.run_in_executor(
            None, 
            lambda: trafilatura.extract(downloaded, include_comments=False, include_tables=True, no_fallback=False)
        )
        
        if content is None or not content.strip():
            raise URLFetchError(f"No content could be extracted from {url}")
            
        return content
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        raise URLFetchError(f"Error fetching URL: {str(e)}") from e
