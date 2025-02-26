"""
Enhanced keyword matching for vector search results.

This module provides functions to improve the relevance of vector search results
by considering multiple occurrences of keywords, term proximity, and phrase matching.
"""

import re
from typing import Dict, List, Tuple, Set, Optional
from .config import *
import logging

logger = logging.getLogger(__name__)

# Common English stopwords
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
    'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 
    'by', 'about', 'against', 'between', 'into', 'through', 'during', 
    'before', 'after', 'above', 'below', 'from', 'up', 'down', 'of', 
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 
    'just', 'should', 'now'
}

def extract_important_terms(query_text: str, min_length: int = 3) -> List[str]:
    """
    Extract important terms from a query by removing stopwords and short terms.
    
    Args:
        query_text: The query text to extract terms from
        min_length: Minimum length of terms to consider important
        
    Returns:
        List of important terms
    """
    return [
        term.lower() for term in re.findall(r'\b\w+\b', query_text)
        if term.lower() not in STOP_WORDS and len(term) >= min_length
    ]

def calculate_term_frequencies(text: str, terms: List[str]) -> Dict[str, int]:
    """
    Calculate the frequency of each term in the text.
    
    Args:
        text: The text to search in
        terms: List of terms to count
        
    Returns:
        Dictionary mapping terms to their frequencies
    """
    text_lower = text.lower()
    term_frequencies = {}
    
    for term in terms:
        # Find all occurrences of the term
        matches = re.findall(r'\b' + re.escape(term) + r'\b', text_lower)
        term_frequencies[term] = len(matches)
    
    return term_frequencies

def calculate_term_proximity(node_text: str, important_terms: List[str], max_distance: int = MAX_DISTANCE_FOR_PROXIMITY) -> float:
    """
    Calculate a proximity score based on how close matching terms appear to each other.
    
    Args:
        node_text: The document text
        important_terms: List of important terms from the query
        
        max_distance: Maximum distance to consider for proximity calculation
    Returns:
        Proximity score between 0.0 and 1.0
    """
    node_text_lower = node_text.lower()
    
    # Find positions of all term occurrences
    term_positions = []
    for term in important_terms:
        for match in re.finditer(r'\b' + re.escape(term) + r'\b', node_text_lower):
            term_positions.append((match.start(), term))
    
    # Sort positions
    term_positions.sort()
    
    if len(term_positions) <= 1:
        return 0.0
    
    # Calculate average distance between consecutive terms
    total_distance = 0
    unique_term_pairs = 0
    
    for i in range(len(term_positions) - 1):
        pos1, term1 = term_positions[i]
        pos2, term2 = term_positions[i + 1]
        
        # Only count distances between different terms
        if term1 != term2:
            distance = pos2 - pos1
            total_distance += distance
            unique_term_pairs += 1
    
    if unique_term_pairs == 0:
        return 0.0
    
    avg_distance = total_distance / unique_term_pairs
    
    # Convert distance to proximity score (closer = higher score)
    # Use a logarithmic scale to handle varying document lengths
    proximity_score = max(0.0, 1.0 - (min(avg_distance, max_distance) / max_distance))
    
    return proximity_score

def detect_phrase_matches(query_text: str, node_text: str) -> float:
    """
    Detect when multi-word phrases from the query appear in the document.
    
    Args:
        query_text: The query text
        node_text: The document text
        
    Returns:
        Phrase match score between 0.0 and 1.0
    """
    # Extract potential phrases (2-3 consecutive words)
    query_words = query_text.lower().split()
    
    phrases = []
    # Extract 2-word phrases
    for i in range(len(query_words) - 1):
        phrases.append(' '.join(query_words[i:i+2]))
    
    # Extract 3-word phrases if possible
    for i in range(len(query_words) - 2):
        phrases.append(' '.join(query_words[i:i+3]))
    
    # Count phrase matches
    node_text_lower = node_text.lower()
    phrase_matches = 0
    
    for phrase in phrases:
        if len(phrase.split()) >= 2:  # Only consider actual phrases (2+ words)
            matches = re.findall(re.escape(phrase), node_text_lower)
            phrase_matches += len(matches)
    
    # Calculate phrase match score
    if phrases:
        phrase_score = min(phrase_matches / len(phrases), 1.0) * PHRASE_MATCH_BOOST
    else:
        phrase_score = 0.0
    
    return phrase_score

def integrated_keyword_matching(query_text: str, node_text: str, base_score: float,
                               frequency_weight: float = DEFAULT_FREQUENCY_WEIGHT,
                               proximity_weight: float = DEFAULT_PROXIMITY_WEIGHT,
                               phrase_weight: float = DEFAULT_PHRASE_WEIGHT,
                               debug: bool = False) -> float:
    """
    Integrated approach combining term frequency, proximity, and phrase matching.
    
    Args:
        query_text: The query text
        node_text: The document text
        base_score: The base similarity score from vector search
        frequency_weight: Weight for term frequency component
        proximity_weight: Weight for term proximity component
        phrase_weight: Weight for phrase matching component
        debug: Whether to print debug information
        
    Returns:
        Adjusted score between 0.0 and 1.0
    """
    # Extract important terms from query
    important_terms = extract_important_terms(query_text, MIN_TERM_LENGTH)
    
    if not important_terms:
        return base_score
    
    # 1. Term frequency component
    term_frequencies = calculate_term_frequencies(node_text, important_terms)
    
    total_matches = sum(term_frequencies.values())
    unique_matches = sum(1 for count in term_frequencies.values() if count > 0)
    
    # Calculate coverage and density
    coverage_ratio = unique_matches / len(important_terms)
    density_factor = min(total_matches / (len(important_terms) * 2), 1.0)
    
    # 2. Term proximity component
    proximity_score = calculate_term_proximity(node_text, important_terms)
    
    # 3. Phrase matching component
    phrase_score = detect_phrase_matches(query_text, node_text)
    
    # Combine all components for final boost
    # Weight the components based on their importance
    combined_boost = 1.0 + (
        (coverage_ratio * COVERAGE_BOOST + density_factor * DENSITY_BOOST) * frequency_weight +
        proximity_score * proximity_weight +
        phrase_score * phrase_weight
    )
    
    if debug:
        logger.debug(f"Important terms: {important_terms}")
        logger.debug(f"Term frequencies: {term_frequencies}")
        logger.debug(f"Coverage: {coverage_ratio:.2f}, Density: {density_factor:.2f}, "
                    f"Proximity: {proximity_score:.2f}, Phrase: {phrase_score:.2f}")
        logger.debug(f"Combined boost: {combined_boost:.2f}")
    
    return min(base_score * combined_boost, 1.0)

def apply_minimum_content_threshold(text: str, score: float, min_length: int = MIN_CONTENT_LENGTH) -> float:
    """
    Penalize documents that are too short to be meaningful.
    
    Args:
        text: The document text
        score: The current similarity score
        min_length: Minimum acceptable length
        
    Returns:
        Adjusted score
    """
    if len(text) < min_length:
        penalty = 0.5 * (len(text) / min_length)
        return score * penalty
    return score

def normalize_by_length(score: float, text_length: int, max_length: int = MAX_CONTENT_LENGTH) -> float:
    """
    Normalize score based on document length to prevent bias toward longer docs.
    
    Args:
        score: The current similarity score
        text_length: Length of the document text
        max_length: Maximum length to consider
        
    Returns:
        Adjusted score
    """
    # Penalize very short documents (might be noise) and very long documents
    length_factor = min(text_length, max_length) / max_length
    # Apply a sigmoid-like normalization that favors mid-length documents
    length_adjustment = (4 * length_factor * (1 - length_factor)) ** 0.5
    return score * length_adjustment

def adjust_score_by_metadata(metadata: Dict, query_text: str, base_score: float) -> float:
    """
    Adjust score based on document metadata.
    
    Args:
        metadata: Document metadata
        query_text: The query text
        base_score: The current similarity score
        
    Returns:
        Adjusted score
    """
    file_name = metadata.get('file_name', '').lower()
    file_type = metadata.get('file_type', '').lower()
    
    # Extract key terms from query
    query_terms = set(query_text.lower().split())
    
    # Check if filename contains query terms
    filename_match = any(term in file_name for term in query_terms if len(term) > 3)
    
    # Boost for relevant file types (e.g., PDF for forms)
    file_type_boost = 1.0
    if 'form' in query_text.lower() and 'pdf' in file_type:
        file_type_boost = FILE_TYPE_BOOST
    
    # Apply boosts
    adjusted_score = base_score
    if filename_match:
        adjusted_score *= FILENAME_MATCH_BOOST
    
    adjusted_score *= file_type_boost
    
    return min(adjusted_score, 1.0)  # Cap at 1.0

def enhance_search_results(query_text: str, 
                          raw_results: List[Tuple[str, Dict, float, int]],
                          initial_top_k: int = DEFAULT_INITIAL_TOP_K,
                          final_top_k: int = DEFAULT_FINAL_TOP_K,
                          min_score: float = DEFAULT_MIN_SCORE,
                          debug: bool = False) -> List[Tuple[str, Dict, float, int]]:
    """
    Enhance search results by applying multiple scoring adjustments and filtering.
    
    Args:
        query_text: The original query text
        raw_results: List of tuples (node_text, metadata, score, chunk_size)
        initial_top_k: Number of initial results to consider
        final_top_k: Number of results to return after enhancement
        min_score: Minimum score threshold
        debug: Whether to print debug information
        
    Returns:
        List of enhanced and filtered results
    """
    # Limit to initial_top_k results
    results_to_enhance = raw_results[:initial_top_k]
    
    # Apply score adjustments
    enhanced_results = []
    for node_text, metadata, score, chunk_size in results_to_enhance:
        # Penalize very short documents
        adjusted_score = apply_minimum_content_threshold(node_text, score)
        
        # Apply integrated keyword matching
        adjusted_score = integrated_keyword_matching(
            query_text, node_text, adjusted_score, debug=debug
        )
        
        # Normalize by length to prevent bias toward very long documents
        adjusted_score = normalize_by_length(adjusted_score, len(node_text))
        
        # Apply metadata-based adjustments
        adjusted_score = adjust_score_by_metadata(metadata, query_text, adjusted_score)
        
        enhanced_results.append((node_text, metadata, adjusted_score, chunk_size))
    
    # Re-sort by adjusted scores
    enhanced_results.sort(key=lambda x: x[2], reverse=True)
    
    # Filter by minimum score and limit to final_top_k
    filtered_results = [r for r in enhanced_results if r[2] >= min_score]
    return filtered_results[:final_top_k]
