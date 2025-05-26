"""
Configuration settings for keyword matching enhancements.

This module provides configurable parameters for the keyword matching algorithms.
"""

# Default weights for different components of the integrated keyword matching
DEFAULT_FREQUENCY_WEIGHT = 0.4  # Weight for term frequency component
DEFAULT_PROXIMITY_WEIGHT = 0.3  # Weight for term proximity component
DEFAULT_PHRASE_WEIGHT = 0.3     # Weight for phrase matching component

# Default thresholds and limits
#DEFAULT_MIN_SCORE = 0.65        # Minimum score to include in results
DEFAULT_MIN_SCORE = 0
DEFAULT_INITIAL_TOP_K = 15      # Number of initial results to consider
DEFAULT_FINAL_TOP_K = 6         # Number of results to return after enhancement

# Term extraction settings
MIN_TERM_LENGTH = 3             # Minimum length of terms to consider important
MAX_DISTANCE_FOR_PROXIMITY = 100  # Maximum character distance for proximity calculation

# Boost factors
COVERAGE_BOOST = 0.25            # Boost factor for term coverage
DENSITY_BOOST = 0.25             # Boost factor for term density
PHRASE_MATCH_BOOST = 0.35        # Boost factor for phrase matches
FILENAME_MATCH_BOOST = 1.2      # Boost factor for filename matches
FILE_TYPE_BOOST = 1.15          # Boost factor for relevant file types

# Document length normalization
MIN_CONTENT_LENGTH = 20         # Minimum content length before applying penalty
MAX_CONTENT_LENGTH = 5000       # Maximum content length for normalization
