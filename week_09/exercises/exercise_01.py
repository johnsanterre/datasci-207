"""
DATASCI 207: Module 9 - Exercise: Text Representations
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# EXERCISE 1: Build Vocabulary and BoW
# =============================================================================

def tokenize(text):
    """Lowercase and split on whitespace."""
    return text.lower().split()


def bag_of_words(documents):
    """
    Create bag-of-words representation.
    
    Args:
        documents: List of text strings
    
    Returns:
        vocab: List of vocabulary words (sorted)
        bow_matrix: Matrix of shape (n_docs, vocab_size)
    """
    # TODO:
    # 1. Build vocabulary from all documents
    # 2. Create word-to-index mapping
    # 3. Create count matrix
    
    vocab = None  # Replace
    bow_matrix = None  # Replace
    
    return vocab, bow_matrix


# =============================================================================
# EXERCISE 2: Compute TF-IDF
# =============================================================================

def tfidf(bow_matrix):
    """
    Compute TF-IDF from bag-of-words matrix.
    
    TF = count / doc_length
    IDF = log(n_docs / document_frequency)
    TF-IDF = TF * IDF
    
    Returns:
        TF-IDF matrix, same shape as bow_matrix
    """
    # TODO: Compute TF-IDF
    tfidf_matrix = None  # Replace
    
    return tfidf_matrix


# =============================================================================
# EXERCISE 3: Cosine Similarity
# =============================================================================

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    
    cos(a, b) = (a · b) / (||a|| * ||b||)
    """
    # TODO: Implement cosine similarity
    similarity = None  # Replace
    
    return similarity


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Text Representations\n")
    
    documents = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "cats and dogs are pets"
    ]
    
    # Test 1: Bag of Words
    print("=" * 50)
    print("Test 1: Bag-of-Words")
    print("=" * 50)
    
    vocab, bow = bag_of_words(documents)
    if vocab is not None and bow is not None:
        print(f"Vocabulary: {vocab}")
        print(f"BoW shape: {bow.shape}")
        print(f"BoW matrix:\n{bow}")
        if bow.shape == (3, len(vocab)):
            print("PASS")
    else:
        print("Not implemented yet")
    
    # Test 2: TF-IDF
    print("\n" + "=" * 50)
    print("Test 2: TF-IDF")
    print("=" * 50)
    
    if bow is not None:
        tfidf_mat = tfidf(bow)
        if tfidf_mat is not None:
            print(f"TF-IDF shape: {tfidf_mat.shape}")
            print("TF-IDF values should differ from raw counts")
            if not np.allclose(tfidf_mat, bow):
                print("PASS")
        else:
            print("Not implemented yet")
    
    # Test 3: Cosine Similarity
    print("\n" + "=" * 50)
    print("Test 3: Cosine Similarity")
    print("=" * 50)
    
    a = np.array([1, 0, 1])
    b = np.array([1, 0, 1])
    c = np.array([0, 1, 0])
    
    sim_ab = cosine_similarity(a, b)
    sim_ac = cosine_similarity(a, c)
    
    if sim_ab is not None:
        print(f"Similarity(a, b): {sim_ab:.4f} (expected: 1.0)")
        print(f"Similarity(a, c): {sim_ac:.4f} (expected: 0.0)")
        if abs(sim_ab - 1.0) < 0.01 and abs(sim_ac) < 0.01:
            print("PASS")
    else:
        print("Not implemented yet")
