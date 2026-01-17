# vector_databases.py
# Explanation of Vector Databases
# Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently.
# They enable fast similarity search, which is crucial for RAG applications.
# Unlike traditional databases that store text/numbers, vector DBs store embeddings and find similar vectors.

# Import required libraries for vector operations and similarity calculations
# numpy for numerical array operations and vector handling
import numpy as np
# cosine_similarity for measuring similarity between high-dimensional vectors
from sklearn.metrics.pairwise import cosine_similarity

# Simple in-memory vector database implementation for educational purposes
# This demonstrates core concepts without external dependencies
class SimpleVectorDB:
    # Initialize the vector database with empty storage
    def __init__(self):
        # List to store the actual vector embeddings (numpy arrays)
        self.vectors = []
        # List to store metadata associated with each vector (e.g., text, IDs)
        self.metadata = []

    # Method to add a vector and its metadata to the database
    def add_vector(self, vector, metadata=None):
        """
        Add a vector to the database.
        Vectors are stored as numpy arrays for efficient computation.
        Metadata can include text content, IDs, timestamps, etc.
        """
        # Convert input vector to numpy array for consistent handling
        # This ensures all vectors are numpy arrays regardless of input type
        self.vectors.append(np.array(vector))
        # Store metadata, defaulting to empty dict if none provided
        self.metadata.append(metadata or {})

    # Method to search for most similar vectors to a query vector
    def search(self, query_vector, top_k=5):
        """
        Search for most similar vectors using cosine similarity.
        Cosine similarity measures angle between vectors, ranging from -1 to 1.
        Higher values indicate greater similarity.
        """
        # Return empty list if no vectors have been added yet
        if not self.vectors:
            return []

        # Convert query vector to numpy array and reshape to 2D (required by cosine_similarity)
        # reshape(1, -1) makes it a row vector: [vector] -> [[vector]]
        query = np.array(query_vector).reshape(1, -1)

        # Convert stored vectors list to numpy matrix for batch similarity calculation
        # This creates a matrix where each row is one stored vector
        vectors_matrix = np.array(self.vectors)

        # Calculate cosine similarity between query and all stored vectors
        # Result is a matrix; [0] takes the first (and only) row for our single query
        similarities = cosine_similarity(query, vectors_matrix)[0]

        # Get indices of top-k most similar vectors (highest similarity first)
        # np.argsort sorts in ascending order, [::-1] reverses to descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results list with vector data, similarity scores, and metadata
        results = []
        for idx in top_indices:
            # Create result dictionary for each top match
            results.append({
                'vector': self.vectors[idx],        # The actual vector embedding
                'similarity': similarities[idx],    # Cosine similarity score
                'metadata': self.metadata[idx]      # Associated metadata (e.g., text)
            })

        # Return the top-k most similar vectors with their metadata
        return results

# Example usage demonstrating the vector database functionality
# Create a new vector database instance
db = SimpleVectorDB()

# Add sample vectors representing different documents
# These are 3D vectors for simplicity; real embeddings are typically 384-768 dimensions
db.add_vector([1, 0, 0], {'text': 'Document about cats'})      # Vector points strongly in x-direction
db.add_vector([0, 1, 0], {'text': 'Document about dogs'})      # Vector points strongly in y-direction
db.add_vector([0.5, 0.5, 0], {'text': 'Document about pets'})  # Vector points diagonally (both cats and dogs)
db.add_vector([0, 0, 1], {'text': 'Document about weather'})   # Vector points in z-direction (unrelated)

# Query vector representing a search for cat-related content
# This vector is similar to the cat document vector [1,0,0]
query = [0.8, 0.2, 0]  # Mostly x-direction with small y-component

# Search for top 3 most similar vectors to the query
results = db.search(query, top_k=3)

# Display the search results
print("Search results:")
for i, result in enumerate(results):
    # Print ranking number (1-based)
    print(f"{i+1}. Similarity: {result['similarity']:.3f}")
    # Print the associated text from metadata
    print(f"   Text: {result['metadata']['text']}")
    # Print the actual vector values
    print(f"   Vector: {result['vector']}")

# Additional explanation of real vector database features
# Real vector databases go beyond this simple implementation:
# - Support indexing for fast search (HNSW, IVF, LSH) - algorithms that speed up similarity search
# - Handle millions/billions of vectors - scale to large datasets
# - Provide filtering, metadata storage - search with additional constraints
# - Offer APIs for CRUD operations - create, read, update, delete vectors
# - Support real-time updates and batch operations - handle streaming data