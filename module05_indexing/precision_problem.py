# precision_problem.py
# Explanation of Precision Problem
# In vector databases and similarity search, precision refers to the accuracy of retrieval.
# Approximate indexing methods trade some precision for speed.
# The precision problem occurs when the approximate results differ significantly from exact results.
# This is a fundamental trade-off in large-scale similarity search.

# Import required libraries for precision analysis
# numpy for numerical operations and random number generation
import numpy as np
# cosine_similarity for measuring vector similarity
from sklearn.metrics.pairwise import cosine_similarity

# Exact k-nearest neighbor search using brute force approach
def exact_search(query, vectors, k=5):
    """
    Exact k-nearest neighbor search using brute force.
    This calculates similarity to ALL vectors and returns the top-k most similar.
    Time complexity: O(n) where n is the number of vectors.
    """
    # Calculate cosine similarity between query and all vectors
    similarities = cosine_similarity([query], vectors)[0]
    # Get indices of top-k most similar vectors (highest similarity first)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    # Return list of (index, similarity) tuples
    return [(idx, similarities[idx]) for idx in top_k_indices]

# Simulate approximate search with controlled error to demonstrate precision issues
def approximate_search(query, vectors, k=5, error_rate=0.1):
    """
    Simulate approximate search with some error.
    This represents what happens with indexing methods like HNSW, IVF, or LSH.
    They don't check all vectors, leading to potential misses or ranking errors.
    """
    # Get more candidates than needed (simulating broader search in indexing)
    exact_results = exact_search(query, vectors, k=k*2)  # Get more candidates

    # Introduce approximation error by adding noise to similarities
    approximate_results = []
    for idx, sim in exact_results[:k]:
        # Add Gaussian noise to simulate indexing approximation errors
        noise = np.random.normal(0, error_rate)
        # Clamp result to [0,1] range (cosine similarity bounds)
        approx_sim = max(0, min(1, sim + noise))  # Keep in [0,1]
        approximate_results.append((idx, approx_sim))

    # Re-sort after introducing approximation errors
    approximate_results.sort(key=lambda x: x[1], reverse=True)
    # Return top-k approximate results
    return approximate_results[:k]

# Example demonstrating the precision problem
np.random.seed(42)  # For reproducible results
vectors = np.random.rand(100, 10)  # 100 vectors in 10D space
query = np.random.rand(10)  # Random query vector

print("Precision Problem Demonstration:")
print(f"Query vector: {query[:5]}...")  # Show first 5 dimensions

# Perform exact search (ground truth)
exact_results = exact_search(query, vectors, k=5)
print("\nExact Search Results:")
for rank, (idx, sim) in enumerate(exact_results, 1):
    print(f"{rank}. Index: {idx}, Similarity: {sim:.3f}")

# Perform approximate search with small error rate
approximate_results = approximate_search(query, vectors, k=5, error_rate=0.05)
print("\nApproximate Search Results:")
for rank, (idx, sim) in enumerate(approximate_results, 1):
    print(f"{rank}. Index: {idx}, Similarity: {sim:.3f}")

# Calculate precision metrics to quantify the approximation quality
def calculate_precision_at_k(exact, approximate, k):
    """
    Calculate precision@k: fraction of approximate results that are in exact top-k.
    This measures how many of the approximate top-k results are actually correct.
    """
    # Get the set of indices that should be in top-k (ground truth)
    exact_top_k = set(idx for idx, _ in exact[:k])
    # Get the indices returned by approximate search
    approx_top_k = [idx for idx, _ in approximate[:k]]
    # Count how many approximate results are in the exact top-k
    correct = sum(1 for idx in approx_top_k if idx in exact_top_k)
    # Return precision as fraction
    return correct / k

# Calculate and display precision@5
precision_at_5 = calculate_precision_at_k(exact_results, approximate_results, 5)
print(f"\nPrecision@5: {precision_at_5:.3f}")

# Real-world precision characteristics of different indexing methods:
# - HNSW: High precision (>0.95) with good speed - best overall trade-off
# - IVF: Good precision with coarse quantization - scales well to large datasets
# - LSH: Lower precision but very fast - good for high-throughput scenarios
# - Trade-offs depend on use case (real-time search vs. accuracy-critical applications)

# Strategies for mitigating precision problems in production:
# - Use higher-quality indexing parameters (more clusters, larger hash tables)
# - Implement re-ranking: get more candidates from index, then re-rank exactly
# - Hybrid approaches: combine multiple indexing methods for better coverage
# - Regular precision monitoring and tuning based on your specific data distribution