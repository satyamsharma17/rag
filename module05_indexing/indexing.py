# indexing.py
# Explanation of Indexing
# Indexing in vector databases refers to data structures that enable fast similarity search.
# Without indexing, searching would require comparing the query to every vector (brute force).
# Indexing algorithms organize vectors to reduce search time from O(n) to O(log n) or better.

# Import required libraries for indexing demonstration
# KDTree provides exact nearest neighbor search for low-dimensional data
from sklearn.neighbors import KDTree
# numpy for numerical operations and random vector generation
import numpy as np

# Generate sample vectors to demonstrate indexing
# Set random seed for reproducible results
np.random.seed(42)
# Create 100 random 2D vectors (points in 2D space)
vectors = np.random.rand(100, 2)  # 100 vectors in 2D

# Build KD-Tree index from the vectors
# KD-Tree partitions space into regions for efficient nearest neighbor search
kdtree = KDTree(vectors)

# Define query point for nearest neighbor search
# This represents a query vector we want to find similar vectors for
query = np.array([[0.5, 0.5]])

# Find k nearest neighbors using the KD-Tree index
# query() returns distances and indices of the k closest vectors
distances, indices = kdtree.query(query, k=5)

# Display the indexing results
print("KD-Tree Indexing Example:")
print(f"Query point: {query[0]}")
print("Nearest neighbors:")
# Iterate through the results and display each neighbor
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i+1}. Distance: {dist:.3f}, Vector: {vectors[idx]}")

# Types of indexing algorithms used in vector databases:
# - Exact indexing: KD-Tree, Ball-Tree (work well for low dimensions like 2D-20D)
# - Approximate indexing: HNSW, IVF, LSH (optimized for high dimensions like 384D-1536D)
# - Trade-off: Approximate methods sacrifice some accuracy for massive speed improvements

# Important note about dimensionality:
# In high-dimensional spaces (>100D), traditional tree structures become ineffective
# This is known as the "curse of dimensionality" - distances become meaningless
# Approximate methods like HNSW (Hierarchical Navigable Small World) are used instead

# Key benefits of indexing for RAG systems:
# - Faster search: Reduces time complexity from O(n) linear scan to O(log n) or better
# - Scalability: Enables handling millions/billions of vectors efficiently
# - Memory efficiency: Uses compressed representations and approximate structures
# - Real-time performance: Essential for production RAG systems requiring low latency