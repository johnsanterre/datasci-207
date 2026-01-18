"""
DATASCI 207: Applied Machine Learning
Module 9: Text and Embeddings

This module covers:
- Bag-of-words representation
- TF-IDF weighting
- Word embeddings concepts
- Document representation

Using NumPy and scikit-learn - no pandas.
"""

import numpy as np
np.random.seed(42)


# =============================================================================
# PART 1: BAG-OF-WORDS FROM SCRATCH
# =============================================================================

print("=== Part 1: Bag-of-Words ===\n")


def tokenize(text):
    """Simple tokenization: lowercase and split on whitespace."""
    return text.lower().split()


def build_vocabulary(documents):
    """Build vocabulary from list of documents."""
    vocab = set()
    for doc in documents:
        tokens = tokenize(doc)
        vocab.update(tokens)
    return sorted(vocab)


def bag_of_words(documents, vocab):
    """
    Create bag-of-words matrix.
    
    Args:
        documents: List of text strings
        vocab: List of vocabulary words
    
    Returns:
        Matrix of shape (n_docs, vocab_size)
    """
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(documents), len(vocab)))
    
    for i, doc in enumerate(documents):
        tokens = tokenize(doc)
        for token in tokens:
            if token in word_to_idx:
                matrix[i, word_to_idx[token]] += 1
    
    return matrix


# Example documents
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog are friends"
]

vocab = build_vocabulary(documents)
bow_matrix = bag_of_words(documents, vocab)

print("Vocabulary:", vocab)
print("\nBag-of-Words Matrix:")
print(bow_matrix)
print("\n(Each row is a document, each column is a word count)")


# =============================================================================
# PART 2: TF-IDF FROM SCRATCH
# =============================================================================

print("\n=== Part 2: TF-IDF ===\n")


def compute_tf(bow_matrix):
    """Term frequency: normalize by document length."""
    doc_lengths = bow_matrix.sum(axis=1, keepdims=True)
    return bow_matrix / np.maximum(doc_lengths, 1)


def compute_idf(bow_matrix):
    """Inverse document frequency: log(N / df)."""
    n_docs = bow_matrix.shape[0]
    df = (bow_matrix > 0).sum(axis=0)  # Document frequency
    return np.log(n_docs / np.maximum(df, 1))


def compute_tfidf(bow_matrix):
    """TF-IDF: TF * IDF."""
    tf = compute_tf(bow_matrix)
    idf = compute_idf(bow_matrix)
    return tf * idf


tfidf_matrix = compute_tfidf(bow_matrix)

print("TF-IDF Matrix:")
print(np.round(tfidf_matrix, 3))

print("\nWords with highest TF-IDF in each document:")
for i, doc in enumerate(documents):
    top_idx = np.argmax(tfidf_matrix[i])
    print(f"  Doc {i}: '{vocab[top_idx]}' (score: {tfidf_matrix[i, top_idx]:.3f})")


# =============================================================================
# PART 3: N-GRAMS
# =============================================================================

print("\n=== Part 3: N-grams ===\n")


def generate_ngrams(text, n):
    """Generate n-grams from text."""
    tokens = tokenize(text)
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = "_".join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams


text = "the quick brown fox jumps"

print(f"Text: '{text}'")
print(f"Unigrams: {generate_ngrams(text, 1)}")
print(f"Bigrams: {generate_ngrams(text, 2)}")
print(f"Trigrams: {generate_ngrams(text, 3)}")


# =============================================================================
# PART 4: WORD EMBEDDINGS CONCEPT
# =============================================================================

print("\n=== Part 4: Word Embeddings ===\n")

print("""
Word embeddings represent words as dense vectors.

Bag-of-Words: 
  "cat" -> [0, 1, 0, 0, ..., 0]  (sparse, 10000+ dims)
  
Word Embedding:
  "cat" -> [0.2, -0.5, 0.8, ...]  (dense, 100-300 dims)

Key properties:
1. Similar words have similar vectors
2. Relationships encoded as vector operations
   king - man + woman ≈ queen
3. Learned from LARGE text corpora
""")

# Simulated mini embeddings
word_embeddings = {
    'king': np.array([0.9, 0.8, 0.1]),
    'queen': np.array([0.85, 0.85, 0.15]),
    'man': np.array([0.5, 0.1, 0.05]),
    'woman': np.array([0.55, 0.15, 0.1]),
    'cat': np.array([0.1, 0.3, 0.9]),
    'dog': np.array([0.15, 0.35, 0.85])
}

print("Simulated 3D embeddings:")
for word, vec in word_embeddings.items():
    print(f"  {word}: {vec}")


# =============================================================================
# PART 5: COSINE SIMILARITY
# =============================================================================

print("\n=== Part 5: Cosine Similarity ===\n")


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


print("Cosine similarities (1 = identical, 0 = orthogonal):")
print(f"  king vs queen: {cosine_similarity(word_embeddings['king'], word_embeddings['queen']):.4f}")
print(f"  cat vs dog: {cosine_similarity(word_embeddings['cat'], word_embeddings['dog']):.4f}")
print(f"  king vs cat: {cosine_similarity(word_embeddings['king'], word_embeddings['cat']):.4f}")


# =============================================================================
# PART 6: DOCUMENT EMBEDDINGS
# =============================================================================

print("\n=== Part 6: Document Embeddings ===\n")


def document_embedding(text, embeddings, default_dim=3):
    """Average word embeddings to get document embedding."""
    tokens = tokenize(text)
    vectors = []
    for token in tokens:
        if token in embeddings:
            vectors.append(embeddings[token])
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(default_dim)


doc1 = "king queen"
doc2 = "cat dog"

emb1 = document_embedding(doc1, word_embeddings)
emb2 = document_embedding(doc2, word_embeddings)

print(f"Doc: '{doc1}' -> embedding: {emb1}")
print(f"Doc: '{doc2}' -> embedding: {emb2}")
print(f"Similarity: {cosine_similarity(emb1, emb2):.4f}")


# =============================================================================
# PART 7: SCIKIT-LEARN TEXT PROCESSING
# =============================================================================

print("\n=== Part 7: scikit-learn Text Processing ===\n")

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    
    # Bag of words
    count_vec = CountVectorizer()
    bow_sklearn = count_vec.fit_transform(documents)
    
    print("CountVectorizer vocabulary:", count_vec.get_feature_names_out())
    print(f"BoW shape: {bow_sklearn.shape}")
    
    # TF-IDF
    tfidf_vec = TfidfVectorizer()
    tfidf_sklearn = tfidf_vec.fit_transform(documents)
    
    print(f"\nTF-IDF shape: {tfidf_sklearn.shape}")
    print("TF-IDF example (first document):")
    print(np.round(tfidf_sklearn[0].toarray(), 3))

except ImportError:
    print("(scikit-learn not available)")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Key Takeaways from Module 9")
print("=" * 60)
print("""
1. BAG-OF-WORDS: Count word occurrences
   - Simple but sparse
   - Loses word order

2. TF-IDF: Weight by importance
   - TF = term frequency in document
   - IDF = inverse document frequency (rarity)
   - High score = important to this document

3. N-GRAMS: Capture word sequences
   - Bigrams, trigrams...
   - Preserve some order information

4. WORD EMBEDDINGS: Dense semantic vectors
   - Similar words -> similar vectors
   - Capture relationships (king-man+woman=queen)
   - Learned from context (Word2Vec, GloVe)

5. COSINE SIMILARITY: Compare vectors
   - Measure angle between vectors
   - 1 = identical direction, 0 = orthogonal

6. DOCUMENT EMBEDDINGS: 
   - Average word embeddings
   - Or use models like BERT
""")
