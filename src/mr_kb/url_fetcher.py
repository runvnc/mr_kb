# url_fetcher.py
import trafilatura
import logging
import asyncio
import re
from typing import Dict, Optional
import requests # Assuming requests is available

logger = logging.getLogger(__name__)

class URLFetchError(Exception):
    """Raised when URL fetching fails."""
    pass

async def fetch_content_from_url(url: str) -> str:
    GDRIVE_DOC_PATTERN = r"https://docs.google.com/document/d/([a-zA-Z0-9_-]+)/"
    
    """Fetch and extract content from a URL using trafilatura.
    
    Args:
        url: The URL to fetch and extract content from
        
    Returns:
        str: The extracted main content of the webpage
        
    Raises:
        URLFetchError: If fetching or extraction fails
    """
    try:
        loop = asyncio.get_event_loop()
        
        # Check if it's a Google Doc URL
        match = re.search(GDRIVE_DOC_PATTERN, url)
        if match:
            doc_id = match.group(1)
            export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
            logger.info(f"Detected Google Doc. Fetching plain text from: {export_url}")
            
            try:
                # Use run_in_executor for the synchronous requests call
                response = await loop.run_in_executor(
                    None, 
                    lambda: requests.get(export_url, timeout=30) # Added timeout
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                content = response.text
                if not content.strip():
                     raise URLFetchError(f"No content could be extracted from Google Doc export URL: {export_url}")
                return content
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching Google Doc URL {export_url}: {str(e)}")
                raise URLFetchError(f"Error fetching Google Doc URL: {str(e)}") from e
        else:
            # Fallback to trafilatura for other URLs
            logger.info(f"Using trafilatura to fetch URL: {url}")
            # Run trafilatura fetch in a thread to avoid blocking
            downloaded = await loop.run_in_executor(None, trafilatura.fetch_url, url)
            
            if downloaded is None:
                raise URLFetchError(f"Failed to download content from {url} using trafilatura")
            
            # Run trafilatura extract in a thread
            content = await loop.run_in_executor(
                None, 
                lambda: trafilatura.extract(downloaded, include_comments=False, include_tables=True, no_fallback=False)
            )
            
            if content is None or not content.strip():
                raise URLFetchError(f"No content could be extracted from {url} using trafilatura")
                
            return content
        
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error processing URL {url}: {str(e)}")
        # Re-raise as URLFetchError or allow specific exceptions if needed
        if not isinstance(e, URLFetchError):
             raise URLFetchError(f"Unexpected error processing URL {url}: {str(e)}") from e
        else:
             raise e # Re-raise the original URLFetchError
