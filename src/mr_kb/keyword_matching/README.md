# Enhanced Keyword Matching for Vector Search

## Overview

This module provides enhanced keyword matching capabilities for vector search results. It improves the relevance of search results by considering multiple occurrences of keywords, term proximity, and phrase matching.

## Key Features

1. **Term Frequency Analysis**: Boosts documents that contain multiple occurrences of query terms
2. **Term Proximity Scoring**: Rewards documents where matching terms appear close together
3. **Phrase Matching**: Gives higher scores to documents that match multi-word phrases from the query
4. **Configurable Parameters**: All weights and thresholds can be adjusted via the config module
5. **Two-Stage Filtering**: Retrieves a larger initial set of results, then filters to a smaller final set after enhancement

## Usage

The main function to use is `enhance_search_results()`, which takes raw vector search results and applies multiple scoring adjustments:

```python
from mr_kb.keyword_matching.enhanced_matching import enhance_search_results

# Get raw results from vector search
raw_results = [...] # List of (text, metadata, score, chunk_size) tuples

# Apply enhanced keyword matching
enhanced_results = enhance_search_results(
    query_text="your search query",
    raw_results=raw_results,
    initial_top_k=15,  # Consider top 15 initial results
    final_top_k=6,     # Return top 6 after enhancement
    min_score=0.65     # Minimum score threshold
)
```

## Configuration

All parameters can be configured in the `config.py` file:

```python
# Weights for different components
DEFAULT_FREQUENCY_WEIGHT = 0.4  # Weight for term frequency component
DEFAULT_PROXIMITY_WEIGHT = 0.3  # Weight for term proximity component
DEFAULT_PHRASE_WEIGHT = 0.3     # Weight for phrase matching component

# Thresholds and limits
DEFAULT_MIN_SCORE = 0.65        # Minimum score to include in results
DEFAULT_INITIAL_TOP_K = 15      # Number of initial results to consider
DEFAULT_FINAL_TOP_K = 6         # Number of results to return after enhancement
```

## How It Works

1. **Term Extraction**: Important terms are extracted from the query by removing stopwords and short terms
2. **Term Frequency Analysis**: The frequency of each term in the document is calculated
3. **Term Proximity Calculation**: The average distance between different matching terms is measured
4. **Phrase Detection**: Multi-word phrases from the query are identified in the document
5. **Score Combination**: All components are weighted and combined to produce a final boost factor
6. **Result Filtering**: Results are re-sorted by adjusted scores and filtered to the final set

## Benefits

- **Better Ranking**: Documents with multiple occurrences of the same keywords are ranked higher
- **Improved Relevance**: Documents where query terms appear close together receive higher scores
- **Phrase Awareness**: Multi-word phrases are given special consideration
- **Configurable Balance**: The relative importance of different factors can be adjusted
- **Efficient Processing**: Two-stage filtering reduces the number of documents that need detailed analysis
