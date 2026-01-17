# indexing_algorithms.py
# Explanation of Indexing Algorithms
# Hierarchical Navigable Small World (HNSW) is a graph-based indexing algorithm.
# It creates a multi-layer graph structure for efficient nearest neighbor search.
# HNSW is widely used in production vector databases for its speed-accuracy balance.

# Import required libraries for HNSW demonstration
# numpy for vector operations and array handling
import numpy as np
# cosine_similarity for measuring vector similarity
from sklearn.metrics.pairwise import cosine_similarity
# heapq for priority queue operations (though not used in this simplified version)
import heapq

# Simplified HNSW implementation for educational purposes
# Real HNSW is much more complex with multiple layers and sophisticated algorithms
class SimpleHNSW:
    # Initialize HNSW structure with maximum connections per node
    def __init__(self, max_connections=16):
        # List to store all vectors in the index
        self.vectors = []
        # Graph structure: each index contains list of connected neighbor indices
        self.graph = []  # List of neighbor lists
        # Maximum number of connections per node (controls graph density)
        self.max_connections = max_connections

    # Add a vector to the HNSW structure
    def add_vector(self, vector):
        """
        Add a vector to the HNSW structure.
        In real HNSW, this involves probabilistic insertion into multiple layers.
        """
        # Convert input to numpy array for consistent handling
        vector = np.array(vector)
        # Add vector to the storage
        self.vectors.append(vector)
        # Get the index of the newly added vector
        idx = len(self.vectors) - 1
        # Initialize empty neighbor list for this vector
        self.graph.append([])

        # Only create connections if we have more than one vector
        if len(self.vectors) > 1:
            # Find nearest neighbors to connect with (simplified approach)
            neighbors = self._find_nearest_neighbors(vector, k=self.max_connections)
            # Store neighbors for this vector
            self.graph[idx] = neighbors

            # Add bidirectional connections (undirected graph)
            for neighbor_idx in neighbors:
                # Only add if not already connected (avoid duplicates)
                if idx not in self.graph[neighbor_idx]:
                    self.graph[neighbor_idx].append(idx)

    # Helper method to find nearest neighbors (simplified brute force for demo)
    def _find_nearest_neighbors(self, query, k):
        """
        Simple nearest neighbor search (brute force for demonstration).
        Real HNSW uses hierarchical graph traversal for efficiency.
        """
        # Return empty list if no vectors exist yet
        if not self.vectors:
            return []

        # Calculate similarity to all existing vectors except the one being added
        similarities = cosine_similarity([query], self.vectors[:-1])[0]  # Exclude self
        # Get indices of top-k most similar vectors
        top_k_indices = np.argsort(similarities)[::-1][:k]
        # Convert to list for easier handling
        return top_k_indices.tolist()

    # Search method to find k nearest neighbors to a query vector
    def search(self, query, k=5):
        """
        Search for k nearest neighbors.
        In real HNSW, this uses hierarchical navigation for efficiency.
        """
        # Return empty if no vectors in index
        if not self.vectors:
            return []

        # Convert query to numpy array
        query = np.array(query)
        # Calculate similarity to all vectors in the index
        similarities = cosine_similarity([query], self.vectors)[0]
        # Get top-k most similar vectors
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Build results with detailed information
        results = []
        for idx in top_k_indices:
            results.append({
                'index': idx,                    # Index in the vectors list
                'vector': self.vectors[idx],     # The actual vector
                'similarity': similarities[idx]  # Similarity score
            })

        # Return the k most similar vectors
        return results

# Example usage demonstrating HNSW indexing
# Create HNSW index instance
hnsw = SimpleHNSW()

# Define sample vectors representing different concepts
# These are 3D vectors for simplicity; real embeddings are much higher dimensional
vectors = [
    [1, 0, 0],      # Vector pointing strongly in x-direction
    [0, 1, 0],      # Vector pointing strongly in y-direction
    [0, 0, 1],      # Vector pointing strongly in z-direction
    [0.5, 0.5, 0],  # Vector in xy-plane
    [0, 0.5, 0.5],  # Vector in yz-plane
    [0.5, 0, 0.5]   # Vector in xz-plane
]

# Add all vectors to the HNSW index
for vec in vectors:
    hnsw.add_vector(vec)

# Perform search with a query vector
query = [0.6, 0.4, 0]  # Query vector similar to [0.5, 0.5, 0]
results = hnsw.search(query, k=3)

# Display search results
print("HNSW Search Results:")
for result in results:
    print(f"Similarity: {result['similarity']:.3f}, Vector: {result['vector']}")

# Key characteristics of HNSW algorithm:
# - Multi-layer graph structure: Different layers for coarse and fine search
# - Hierarchical navigation: Start from top layer, drill down to find neighbors
# - Probabilistic construction: Random decisions for graph connectivity
# - Excellent search quality vs. speed trade-off: Fast and accurate

# Other important indexing approaches used in vector databases:
# - IVF (Inverted File): Partition vector space into clusters, search within relevant clusters
# - LSH (Locality Sensitive Hashing): Hash similar vectors to same buckets for quick lookup