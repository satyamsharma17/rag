# Step 5: Indexing for Fast Search

## Overview
Indexing is the process of organizing vector data to enable fast similarity searches. Without proper indexing, finding similar vectors requires comparing the query against every vector in the database, which becomes infeasible at scale.

## The Need for Indexing

### Brute Force Limitations
- **Time Complexity**: O(n) - linear search through all vectors
- **Scalability Issues**: Becomes slow with millions of vectors
- **Real-time Requirements**: Many applications need sub-second responses

### Indexing Benefits
- **Speed**: Reduce search time from linear to sub-linear or logarithmic
- **Scalability**: Handle billions of vectors efficiently
- **Accuracy Trade-offs**: Approximate methods sacrifice some precision for speed

## Indexing Algorithms

### Exact Indexing

#### KD-Tree
```python
# See indexing.py for implementation
from sklearn.neighbors import KDTree

# Build index
vectors = np.random.rand(1000, 2)  # 2D for demonstration
kdtree = KDTree(vectors)

# Query
query = np.array([[0.5, 0.5]])
distances, indices = kdtree.query(query, k=5)
```

**Characteristics:**
- Exact nearest neighbor search
- Works well in low dimensions (≤20D)
- Degrades in high dimensions

### Approximate Indexing

#### HNSW (Hierarchical Navigable Small World)
```python
# See indexing_algorithms.py for simplified implementation
class SimpleHNSW:
    def __init__(self, max_connections=16):
        self.vectors = []
        self.graph = []

    def add_vector(self, vector):
        # Add to multi-layer graph structure
        # Connect to nearest neighbors probabilistically

    def search(self, query, k=5):
        # Navigate graph hierarchy for fast search
```

**Characteristics:**
- Graph-based structure with multiple layers
- Excellent search quality vs speed trade-off
- Used by major vector databases (Pinecone, Weaviate)

#### IVF (Inverted File)
```python
# See indexing_approaches.py
class SimpleIVF:
    def build_index(self, vectors):
        # Cluster vectors into partitions
        kmeans = KMeans(n_clusters=10)
        clusters = kmeans.fit_predict(vectors)

        # Store vectors in cluster buckets

    def search(self, query):
        # Search nearest clusters first
        # Then search within clusters
```

**Characteristics:**
- Partition-based approach
- Fast search with good precision
- Memory efficient

#### LSH (Locality Sensitive Hashing)
```python
class SimpleLSH:
    def add_vector(self, vector):
        # Hash vector into multiple buckets
        for table_idx in range(self.num_hash_tables):
            hash_value = self._hash_vector(vector, table_idx)
            self.hash_tables[table_idx][hash_value].append(idx)

    def search(self, query):
        # Query same hash buckets
        # Return candidates from matching buckets
```

**Characteristics:**
- Hash-based approach
- Very fast but lower precision
- Probabilistic guarantees

## Choosing the Right Index

### Selection Factors
- **Dimensionality**: Low-D (KD-Tree), High-D (HNSW, IVF)
- **Dataset Size**: Small (brute force), Large (approximate methods)
- **Accuracy Requirements**: Exact (KD-Tree), Approximate (HNSW)
- **Query Speed**: Critical (LSH), Balanced (HNSW, IVF)

### Performance Comparison
| Algorithm | Search Speed | Accuracy | Memory Usage | Best For |
|-----------|-------------|----------|--------------|----------|
| Brute Force | Slowest | Perfect | Low | Small datasets |
| KD-Tree | Fast | High | Medium | Low dimensions |
| IVF | Fast | Good | Medium | Balanced performance |
| HNSW | Fast | Excellent | High | High accuracy needs |
| LSH | Fastest | Lower | Low | Speed-critical apps |

## Implementation Considerations

### Index Construction
```python
# Build index with parameters
index = build_index(vectors,
                    algorithm='hnsw',
                    max_connections=16,
                    ef_construction=200)
```

### Search Parameters
```python
# Configure search quality vs speed
results = index.search(query,
                      k=10,
                      ef_search=100)  # Higher = better accuracy, slower
```

### Index Maintenance
- **Updates**: Handle additions/deletions efficiently
- **Rebuilding**: Periodic index optimization
- **Memory Management**: Balance index size with performance

## Code Reference
- `indexing.py`: Basic indexing concepts with KD-Tree
- `indexing_algorithms.py`: HNSW implementation
- `indexing_approaches.py`: IVF and LSH implementations

## Next Steps
With efficient indexing in place, proceed to [Step 6: Search Implementation](../module03_embeddings_search/06_search_implementation.md) to learn how to implement the retrieval component of your RAG system.