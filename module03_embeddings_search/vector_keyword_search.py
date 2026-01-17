# vector_keyword_search.py
# Explanation of Vector Keyword Search
# Vector keyword search uses techniques like TF-IDF and BM25 to rank documents based on keyword relevance.
# TF-IDF: Term Frequency-Inverse Document Frequency - measures importance of terms in documents
# BM25: Best Matching 25, an improved ranking function that considers term frequency and document length

# Import required libraries for text processing and similarity calculations
# TfidfVectorizer converts text documents to TF-IDF feature vectors
from sklearn.feature_extraction.text import TfidfVectorizer
# cosine_similarity calculates similarity between vectors using cosine distance
from sklearn.metrics.pairwise import cosine_similarity
# numpy for numerical operations and array handling
import numpy as np

# Sample documents to demonstrate keyword-based search
# These documents contain overlapping keywords for testing relevance ranking
documents = [
    "The cat sat on the mat",        # Contains "cat"
    "The dog played in the park",    # Contains "dog"
    "Cats and dogs are pets",        # Contains both "cat" and "dog"
    "The weather is nice today"      # Contains neither keyword
]

# Create TF-IDF vectorizer to convert text to numerical vectors
# TF-IDF weighs terms by importance: high in specific doc, low across all docs
vectorizer = TfidfVectorizer()
# Fit the vectorizer to documents and transform them into TF-IDF matrix
# Each row is a document, each column is a term, values are TF-IDF scores
tfidf_matrix = vectorizer.fit_transform(documents)

# Define TF-IDF search function
def tfidf_search(query, documents, vectorizer, tfidf_matrix):
    """
    Perform TF-IDF based search.
    Converts query to TF-IDF vector and finds most similar documents.
    """
    # Transform the query into TF-IDF vector using the same vectorizer
    # This ensures query and documents use the same vocabulary/features
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between query vector and all document vectors
    # cosine_similarity returns a matrix; flatten() converts to 1D array
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Sort indices in descending order (highest similarity first)
    # np.argsort gives indices in ascending order, [::-1] reverses it
    ranked_indices = np.argsort(similarities)[::-1]

    # Return list of (document, similarity_score) tuples in ranked order
    return [(documents[i], similarities[i]) for i in ranked_indices]

# BM25 implementation (simplified version for educational purposes)
class SimpleBM25:
    # Initialize BM25 model with documents and parameters
    def __init__(self, documents):
        # Store the original documents for scoring
        self.documents = documents

        # Dictionary to store document frequency for each term
        # Document frequency = number of documents containing the term
        self.doc_freq = {}

        # Calculate document lengths (number of words in each document)
        self.doc_lengths = [len(doc.split()) for doc in documents]

        # Calculate average document length across all documents
        self.avg_doc_length = np.mean(self.doc_lengths)

        # BM25 parameters: k1 controls term frequency scaling, b controls length normalization
        self.k1 = 1.5  # BM25 parameter for term frequency saturation
        self.b = 0.75  # BM25 parameter for document length normalization

        # Calculate document frequencies for all unique terms
        for doc in documents:
            # Convert to lowercase and split into words, use set to get unique terms
            words = set(doc.lower().split())
            # Count how many documents contain each term
            for word in words:
                self.doc_freq[word] = self.doc_freq.get(word, 0) + 1

    # Calculate BM25 scores for a query against all documents
    def score(self, query):
        # Split query into individual terms
        query_terms = query.lower().split()

        # List to store BM25 scores for each document
        scores = []

        # Calculate score for each document
        for doc_idx, doc in enumerate(self.documents):
            # Initialize score for this document
            score = 0

            # Get document length (number of words)
            doc_length = self.doc_lengths[doc_idx]

            # Convert document to lowercase words for term matching
            doc_words = doc.lower().split()

            # Calculate score contribution from each query term
            for term in query_terms:
                # Only score if the term appears in this document
                if term in doc_words:
                    # Term frequency: how many times term appears in this document
                    tf = doc_words.count(term)

                    # Document frequency: how many documents contain this term
                    df = self.doc_freq.get(term, 0)

                    # Inverse document frequency: rarer terms get higher weight
                    # Uses Laplace smoothing (+0.5) to avoid division by zero
                    idf = np.log((len(self.documents) - df + 0.5) / (df + 0.5))

                    # BM25 scoring formula: combines TF, IDF, and length normalization
                    # (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length/avg_length))
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length))

            # Add this document's score to the results list
            scores.append(score)

        # Return BM25 scores for all documents
        return scores

# Create BM25 model instance with our documents
bm25 = SimpleBM25(documents)

# Define BM25 search function
def bm25_search(query, bm25_model, documents):
    """
    Perform BM25 search.
    Uses BM25 scoring to rank documents by relevance to query terms.
    """
    # Get BM25 scores for all documents
    scores = bm25_model.score(query)

    # Sort document indices by BM25 score in descending order
    ranked_indices = np.argsort(scores)[::-1]

    # Return list of (document, score) tuples in ranked order
    return [(documents[i], scores[i]) for i in ranked_indices]

# Example usage demonstrating both TF-IDF and BM25 search
query = "cat dog"

# Perform TF-IDF search and display results
print("TF-IDF Search Results:")
tfidf_results = tfidf_search(query, documents, vectorizer, tfidf_matrix)
# Show top 3 results with formatted scores
for doc, score in tfidf_results[:3]:
    print(f"Score: {score:.3f} - {doc}")

# Perform BM25 search and display results
print("\nBM25 Search Results:")
bm25_results = bm25_search(query, bm25, documents)
# Show top 3 results with formatted scores
for doc, score in bm25_results[:3]:
    print(f"Score: {score:.3f} - {doc}")