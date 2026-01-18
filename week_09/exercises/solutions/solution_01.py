"""
DATASCI 207: Module 9 - Exercise SOLUTIONS
"""

import numpy as np
np.random.seed(42)


def tokenize(text):
    return text.lower().split()


def bag_of_words(documents):
    """Create bag-of-words representation."""
    # Build vocabulary
    vocab_set = set()
    for doc in documents:
        vocab_set.update(tokenize(doc))
    vocab = sorted(vocab_set)
    
    # Create mapping
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    # Build matrix
    bow_matrix = np.zeros((len(documents), len(vocab)))
    for i, doc in enumerate(documents):
        for token in tokenize(doc):
            if token in word_to_idx:
                bow_matrix[i, word_to_idx[token]] += 1
    
    return vocab, bow_matrix


def tfidf(bow_matrix):
    """Compute TF-IDF from bag-of-words matrix."""
    # TF
    doc_lengths = bow_matrix.sum(axis=1, keepdims=True)
    tf = bow_matrix / np.maximum(doc_lengths, 1)
    
    # IDF
    n_docs = bow_matrix.shape[0]
    df = (bow_matrix > 0).sum(axis=0)
    idf = np.log(n_docs / np.maximum(df, 1))
    
    return tf * idf


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    documents = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "cats and dogs are pets"
    ]
    
    vocab, bow = bag_of_words(documents)
    assert bow.shape == (3, len(vocab))
    print("Bag-of-Words: VERIFIED")
    
    tfidf_mat = tfidf(bow)
    assert not np.allclose(tfidf_mat, bow)
    print("TF-IDF: VERIFIED")
    
    a = np.array([1, 0, 1])
    b = np.array([1, 0, 1])
    c = np.array([0, 1, 0])
    assert abs(cosine_similarity(a, b) - 1.0) < 0.01
    assert abs(cosine_similarity(a, c)) < 0.01
    print("Cosine Similarity: VERIFIED")
