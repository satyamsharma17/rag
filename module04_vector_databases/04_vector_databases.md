# Step 4: Vector Databases

## Overview
Vector databases are specialized storage systems designed to efficiently store, index, and query high-dimensional vectors. They are essential for RAG applications that need to perform fast similarity searches across large collections of embeddings.

## Why Vector Databases?

### Limitations of Traditional Databases
- **Dimensionality**: Standard databases struggle with high-dimensional data
- **Similarity Search**: SQL doesn't support vector similarity operations
- **Performance**: Brute-force search becomes infeasible at scale

### Vector Database Advantages
- **Optimized Storage**: Efficient compression and storage of high-dimensional vectors
- **Fast Search**: Specialized indexing algorithms for similarity search
- **Scalability**: Handle millions to billions of vectors
- **Rich Metadata**: Store and filter by associated data

## Core Operations

### CRUD Operations
```python
# See vector_databases.py for complete implementation
class VectorDatabase:
    def add_vectors(self, vectors, metadata=None):
        """Store vectors with optional metadata"""

    def search(self, query_vector, top_k=5):
        """Find most similar vectors"""

    def update_vector(self, vector_id, new_vector):
        """Update existing vector"""

    def delete_vector(self, vector_id):
        """Remove vector from database"""
```

### Similarity Search
```python
# Cosine similarity search
def cosine_similarity_search(query, vectors, k=5):
    similarities = cosine_similarity([query], vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(i, similarities[i]) for i in top_indices]
```

## Popular Vector Databases

### Chroma (Open Source)
```python
# See vector_db_implementations.py
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

# Add documents
collection.add(
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    documents=["Document 1", "Document 2"],
    ids=["doc1", "doc2"]
)

# Search
results = collection.query(
    query_embeddings=[[0.15, 0.25, 0.35]],
    n_results=2
)
```

### Pinecone (Managed)
```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("my-index")

# Upsert vectors
index.upsert([
    ("vec1", [0.1, 0.2, 0.3], {"text": "Document 1"}),
    ("vec2", [0.4, 0.5, 0.6], {"text": "Document 2"})
])

# Query
results = index.query(vector=[0.1, 0.2, 0.3], top_k=3)
```

### Weaviate (GraphQL API)
```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# GraphQL query
result = client.query.get("Document", ["content"]).with_near_vector({
    "vector": [0.1, 0.2, 0.3]
}).do()
```

## Choosing a Vector Database

### Selection Criteria
- **Scale**: Number of vectors and query volume
- **Features**: Filtering, metadata, real-time updates
- **Deployment**: Self-hosted vs managed service
- **Cost**: Open-source vs commercial licensing
- **Ecosystem**: Integration with your tech stack

### Common Choices
- **Development/Prototyping**: Chroma, FAISS
- **Production/Small Scale**: Weaviate, Qdrant
- **Production/Large Scale**: Pinecone, Vespa, Milvus

## Implementation Best Practices

### Data Preparation
```python
# Normalize embeddings
from sklearn.preprocessing import normalize
normalized_embeddings = normalize(embeddings, norm='l2')

# Add to database in batches
batch_size = 1000
for i in range(0, len(embeddings), batch_size):
    batch = embeddings[i:i + batch_size]
    metadata_batch = metadata[i:i + batch_size]
    db.add_batch(batch, metadata_batch)
```

### Performance Optimization
- **Batch Operations**: Add vectors in batches for efficiency
- **Indexing**: Choose appropriate indexing algorithm (covered in next step)
- **Caching**: Cache frequently accessed vectors
- **Monitoring**: Track query latency and throughput

## Code Reference
- `vector_databases.py`: Simple vector database implementation
- `vector_db_implementations.py`: Examples with Chroma, Pinecone, Weaviate

## Next Steps
With your embeddings stored in a vector database, move to [Step 5: Indexing](../module05_indexing/05_indexing.md) to learn how to optimize search performance with specialized indexing algorithms.