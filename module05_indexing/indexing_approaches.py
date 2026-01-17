# indexing_approaches.py
# Explanation of Indexing Approaches
# Different strategies for organizing and searching high-dimensional vector data.
# Each approach has different trade-offs between speed, accuracy, and memory usage.

# Import required libraries for indexing demonstrations
# numpy for numerical operations and vector handling
import numpy as np
# KMeans for clustering in IVF implementation
from sklearn.cluster import KMeans
# cosine_similarity for measuring vector similarity
from sklearn.metrics.pairwise import cosine_similarity

# Inverted File (IVF) - Partition-based approach
# IVF divides vector space into clusters and searches only relevant clusters
class SimpleIVF:
    # Initialize IVF with number of clusters to create
    def __init__(self, n_clusters=10):
        # Number of clusters (partitions) to divide the space into
        self.n_clusters = n_clusters
        # K-means model for clustering (initialized during build_index)
        self.kmeans = None
        # List of lists: each sublist contains vector indices belonging to that cluster
        self.clusters = [[] for _ in range(n_clusters)]
        # Storage for all vectors in the index
        self.vectors = []

    # Build the IVF index by clustering all vectors
    def build_index(self, vectors):
        """
        Build IVF index by clustering vectors.
        This creates partitions that group similar vectors together.
        """
        # Convert all vectors to numpy arrays for consistent handling
        self.vectors = [np.array(v) for v in vectors]
        # Convert to numpy array for sklearn compatibility
        vectors_array = np.array(vectors)

        # Use K-means clustering to partition the vector space
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        # Fit K-means and get cluster assignments for each vector
        cluster_labels = self.kmeans.fit_predict(vectors_array)

        # Assign each vector to its corresponding cluster
        for idx, label in enumerate(cluster_labels):
            self.clusters[label].append(idx)

    # Search using IVF approach: find nearest clusters, then search within them
    def search(self, query, k=5):
        """
        Search using IVF: query nearest clusters first.
        This avoids searching the entire dataset by focusing on relevant partitions.
        """
        # Convert query to numpy array
        query = np.array(query)

        # Find distances to all cluster centers
        cluster_distances = cosine_similarity([query], self.kmeans.cluster_centers_)[0]
        # Get indices of the 3 nearest clusters (can be tuned)
        nearest_cluster_indices = np.argsort(cluster_distances)[::-1][:3]  # Top 3 clusters

        # Collect candidate vectors from the nearest clusters
        candidates = []
        for cluster_idx in nearest_cluster_indices:
            # For each vector in this cluster, calculate similarity to query
            for vec_idx in self.clusters[cluster_idx]:
                similarity = cosine_similarity([query], [self.vectors[vec_idx]])[0][0]
                candidates.append((vec_idx, similarity))

        # Sort candidates by similarity (highest first) and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

# Locality Sensitive Hashing (LSH) - Hash-based approach
# LSH hashes similar vectors to the same buckets for efficient lookup
import hashlib

class SimpleLSH:
    # Initialize LSH with multiple hash tables for better accuracy
    def __init__(self, num_hash_tables=5, hash_size=8):
        # Number of hash tables (more tables = better accuracy but more memory)
        self.num_hash_tables = num_hash_tables
        # Size of hash values (affects collision probability)
        self.hash_size = hash_size
        # List of hash tables (dictionaries mapping hash values to vector indices)
        self.hash_tables = [{} for _ in range(num_hash_tables)]
        # Storage for all vectors
        self.vectors = []

    # Hash function that converts vector to an integer hash value
    def _hash_vector(self, vector, seed):
        """
        Simple hash function for vector (not cryptographically secure).
        Uses MD5 hash of vector string representation with seed for different tables.
        """
        # Convert vector to string with fixed precision
        vec_str = ','.join(f"{x:.6f}" for x in vector)
        # Create hash with seed to get different hashes for different tables
        hash_obj = hashlib.md5(f"{vec_str}_{seed}".encode())
        # Convert hex hash to integer and take modulo for fixed-size hash
        return int(hash_obj.hexdigest(), 16) % (2 ** self.hash_size)

    # Add vector to LSH index by hashing it into multiple tables
    def add_vector(self, vector):
        """
        Add vector to LSH index.
        Each vector is hashed into multiple hash tables for redundancy.
        """
        # Convert to numpy array
        vector = np.array(vector)
        # Store the vector
        self.vectors.append(vector)
        # Get index of newly added vector
        idx = len(self.vectors) - 1

        # Hash the vector into each hash table
        for table_idx in range(self.num_hash_tables):
            # Generate hash value for this table
            hash_value = self._hash_vector(vector, table_idx)
            # Initialize bucket if it doesn't exist
            if hash_value not in self.hash_tables[table_idx]:
                self.hash_tables[table_idx][hash_value] = []
            # Add vector index to the hash bucket
            self.hash_tables[table_idx][hash_value].append(idx)

    # Search using LSH: find vectors that hash to same buckets as query
    def search(self, query, k=5):
        """
        Search using LSH: find vectors in same hash buckets.
        This is probabilistic - may miss some similar vectors but is very fast.
        """
        # Convert query to numpy array
        query = np.array(query)
        # Set to collect candidate vector indices (eliminates duplicates)
        candidates = set()

        # Query each hash table to find potential similar vectors
        for table_idx in range(self.num_hash_tables):
            # Get hash value for query in this table
            hash_value = self._hash_vector(query, table_idx)
            # If hash bucket exists, add all vectors in it to candidates
            if hash_value in self.hash_tables[table_idx]:
                candidates.update(self.hash_tables[table_idx][hash_value])

        # Calculate actual similarities for all candidate vectors
        results = []
        for idx in candidates:
            similarity = cosine_similarity([query], [self.vectors[idx]])[0][0]
            results.append((idx, similarity))

        # Sort by similarity and return top-k results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

# Example usage demonstrating both IVF and LSH approaches
vectors = [
    [1, 0, 0],      # Vector along x-axis
    [0, 1, 0],      # Vector along y-axis
    [0, 0, 1],      # Vector along z-axis
    [0.5, 0.5, 0],  # Vector in xy-plane
    [0, 0.5, 0.5],  # Vector in yz-plane
    [0.5, 0, 0.5]   # Vector in xz-plane
]

# Demonstrate IVF (Inverted File) indexing
ivf = SimpleIVF(n_clusters=3)  # Use 3 clusters for this small dataset
ivf.build_index(vectors)
query = [0.6, 0.4, 0]  # Query vector similar to [0.5, 0.5, 0]
ivf_results = ivf.search(query, k=3)
print("IVF Results:")
for idx, sim in ivf_results:
    print(f"Index: {idx}, Similarity: {sim:.3f}, Vector: {vectors[idx]}")

# Demonstrate LSH (Locality Sensitive Hashing)
lsh = SimpleLSH(num_hash_tables=3, hash_size=4)  # 3 hash tables, 4-bit hashes
for vec in vectors:
    lsh.add_vector(vec)
lsh_results = lsh.search(query, k=3)
print("\nLSH Results:")
for idx, sim in lsh_results:
    print(f"Index: {idx}, Similarity: {sim:.3f}, Vector: {vectors[idx]}")

# Key differences between the approaches:
# IVF: Fast search by limiting to relevant partitions (clusters), more deterministic
# LSH: Probabilistic approach, may miss some similar items but very fast and memory efficient