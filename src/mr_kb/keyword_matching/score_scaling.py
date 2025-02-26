"""
Score scaling utilities for keyword matching.

This module provides functions to scale and normalize scores to prevent saturation
and create better differentiation between highly relevant documents.
"""

import math

# Set a reasonable ceiling for scores
MAX_SCORE_CEILING = 2.0

def apply_logarithmic_boost(base_score, boost_factor):
    """
    Apply logarithmic scaling to boost factors to prevent scores from easily reaching 1.0.
    
    Args:
        base_score: The original similarity score
        boost_factor: The calculated boost factor
        
    Returns:
        Adjusted score with logarithmic scaling
    """
    if boost_factor <= 1.0:
        return base_score
    
    # log(1+x) grows much slower than x as x increases
    # This creates a more gradual increase for higher boost values
    scaled_boost = 1.0 + (math.log1p(boost_factor - 1.0) * 0.3)
    
    # Apply the scaled boost to the base score
    adjusted_score = base_score * scaled_boost
    
    # Apply a reasonable ceiling
    return min(adjusted_score, MAX_SCORE_CEILING)

def normalize_scores(results):
    """
    Normalize scores to create more differentiation between top results.
    
    Args:
        results: List of (text, metadata, score, chunk_size) tuples
        
    Returns:
        List with normalized scores
    """
    if not results:
        return results
    
    # Find the maximum score
    max_score = max(result[2] for result in results)
    
    # If max score is already below our ceiling, no need to normalize
    if max_score < 0.9:
        return results
    
    # Normalize scores to create more spread
    normalized_results = []
    for text, metadata, score, chunk_size in results:
        # Keep the top result close to its original score but cap at ceiling
        if score == max_score:
            normalized_score = min(score, MAX_SCORE_CEILING)
        else:
            # Create more separation between scores
            # Map other scores to maintain their relative ordering
            # but creates more differentiation
            relative_score = score / max_score
            normalized_score = MAX_SCORE_CEILING * 0.7 + (0.25 * relative_score)
        
        normalized_results.append((text, metadata, normalized_score, chunk_size))
    
    return normalized_results

def compress_high_scores(results):
    """
    Compress the upper range of scores to create more differentiation.
    
    Args:
        results: List of (text, metadata, score, chunk_size) tuples
        
    Returns:
        List with compressed scores
    """
    compressed_results = []
    for text, metadata, score, chunk_size in results:
        # Apply more compression to scores above 0.8
        if score > 0.8:
            # Map 0.8-1.0 range to 0.8-0.95 range
            compressed = 0.8 + (score - 0.8) * 0.9  # Less compression
        else:
            compressed = score
        
        compressed_results.append((text, metadata, compressed, chunk_size))
    
    return compressed_results

def apply_z_score_normalization(results):
    """
    Apply statistical normalization using z-scores.
    
    Args:
        results: List of (text, metadata, score, chunk_size) tuples
        
    Returns:
        List with z-score normalized scores
    """
    if len(results) <= 1:
        return results
    
    scores = [result[2] for result in results]
    mean = sum(scores) / len(scores)
    std_dev = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
    
    if std_dev < 0.001:  # Avoid division by zero or very small values
        return results
    
    normalized_results = []
    for text, metadata, score, chunk_size in results:
        # Convert to z-score and then to a 0-1 scale
        z_score = (score - mean) / std_dev
        # Map z-scores (typically z-scores of -3 to +3 cover 99.7% of data)
        normalized_score = min(max((z_score + 3) / 6, 0), MAX_SCORE_CEILING)
        normalized_results.append((text, metadata, normalized_score, chunk_size))
    
    return normalized_results
