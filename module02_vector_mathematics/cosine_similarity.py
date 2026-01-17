# cosine_similarity.py
# Explanation of Cosine Similarity
# Cosine similarity measures the cosine of the angle between two vectors.
# It's used to determine how similar two documents or items are, regardless of their magnitude.
# Values range from -1 (opposite) to 1 (identical).

import numpy as np  # Import NumPy for vector operations
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine similarity from scikit-learn

def cosine_similarity_manual(vec1, vec2):  # Define function to calculate cosine similarity manually
    """
    Calculate cosine similarity manually.
    """
    dot_product = np.dot(vec1, vec2)  # Calculate dot product of the two vectors
    norm1 = np.linalg.norm(vec1)  # Calculate Euclidean norm (magnitude) of first vector
    norm2 = np.linalg.norm(vec2)  # Calculate Euclidean norm (magnitude) of second vector
    return dot_product / (norm1 * norm2)  # Return cosine similarity: dot product divided by product of magnitudes

# Example vectors (could represent document embeddings)
vec1 = np.array([1, 2, 3, 4])  # Create first example vector
vec2 = np.array([1, 2, 3, 5])  # Create second vector (similar to vec1)
vec3 = np.array([-1, -2, -3, -4])  # Create third vector (opposite direction to vec1)

print("Manual Cosine Similarity:")  # Print section header
print(f"vec1 vs vec2: {cosine_similarity_manual(vec1, vec2):.3f}")  # Print similarity between vec1 and vec2
print(f"vec1 vs vec3: {cosine_similarity_manual(vec1, vec3):.3f}")  # Print similarity between vec1 and vec3

# Using scikit-learn
vectors = np.array([vec1, vec2, vec3])  # Create array of all vectors for pairwise comparison
similarity_matrix = cosine_similarity(vectors)  # Calculate pairwise cosine similarities

print("\nCosine Similarity Matrix:")  # Print section header
print(similarity_matrix)  # Print the similarity matrix showing all pairwise similarities

# In the context of BM25Okapi, cosine similarity can be used to compare query and document vectors
# The get_scores() method in BM25 typically returns scores for unique keywords

from rank_bm25 import BM25Okapi  # Import BM25 ranking algorithm

# Sample documents and query
documents = [  # Define sample documents for BM25 demonstration
    "The cat sat on the mat",  # First document
    "The dog played in the park",  # Second document
    "Cats and dogs are pets"  # Third document
]

# Tokenize documents
tokenized_docs = [doc.lower().split() for doc in documents]  # Convert documents to lowercase and split into words

# Create BM25 model
bm25 = BM25Okapi(tokenized_docs)  # Initialize BM25 model with tokenized documents

# Query
query = "cat dog"  # Define search query
tokenized_query = query.lower().split()  # Tokenize the query

# Get BM25 scores
scores = bm25.get_scores(tokenized_query)  # Calculate BM25 relevance scores for the query against all documents
print(f"\nBM25 scores for query '{query}': {scores}")  # Print the BM25 scores for each document

# Note: get_scores() returns scores for each document, not just unique keywords
# For unique keywords, you might need to process the query differently