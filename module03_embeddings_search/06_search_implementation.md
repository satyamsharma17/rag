# Step 6: Search Implementation

## Overview
The search implementation is the retrieval component of your RAG system. It takes a user query, converts it to an embedding, and finds the most relevant documents from your knowledge base using vector similarity.

## Search Pipeline

### Basic Search Flow
1. **Query Processing**: Clean and prepare the input query
2. **Embedding Generation**: Convert query to vector representation
3. **Similarity Search**: Find most similar vectors in database
4. **Re-ranking**: Optional refinement of initial results
5. **Result Formatting**: Prepare retrieved documents for generation

## Implementing Semantic Search

### Query Embedding
```python
# See semantic_search.py for complete implementation
def embed_query(query, model):
    """Convert text query to embedding"""
    query_embedding = model.encode([query])[0]
    return query_embedding
```

### Vector Similarity Search
```python
# See cosine_similarity.py for implementation
def semantic_search(query_embedding, document_embeddings, top_k=5):
    """Find most similar documents"""
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx],
            'index': idx
        })

    return results
```

### Complete Search Function
```python
def retrieve_documents(query, embedding_model, vector_db, top_k=5):
    """
    Complete retrieval pipeline
    """
    # 1. Embed query
    query_embedding = embed_query(query, embedding_model)

    # 2. Search vector database
    results = vector_db.search(query_embedding, top_k=top_k)

    # 3. Format results
    retrieved_docs = []
    for result in results:
        retrieved_docs.append({
            'content': result['metadata']['text'],
            'similarity': result['similarity'],
            'source': result['metadata'].get('source', 'unknown')
        })

    return retrieved_docs
```

## Search Strategies

### Similarity Metrics

#### Cosine Similarity
```python
# Measures angle between vectors (direction similarity)
def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
```

#### Euclidean Distance
```python
# Measures straight-line distance between vectors
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)
```

#### Dot Product
```python
# Measures projection of one vector onto another
def dot_product_similarity(vec1, vec2):
    return np.dot(vec1, vec2)
```

### Hybrid Search
Combine semantic search with keyword-based search for better results:

```python
def hybrid_search(query, embedding_model, vector_db, keyword_index, top_k=5):
    # Semantic search
    semantic_results = semantic_search(query, embedding_model, vector_db, top_k=top_k*2)

    # Keyword search
    keyword_results = keyword_search(query, keyword_index, top_k=top_k*2)

    # Combine and re-rank
    combined_results = combine_results(semantic_results, keyword_results)
    final_results = rerank_results(combined_results, query, top_k=top_k)

    return final_results
```

## Advanced Search Techniques

### Query Expansion
```python
def expand_query(query, llm):
    """Use LLM to expand query with synonyms and related terms"""
    prompt = f"Expand this search query with related terms: {query}"
    expanded = llm.generate(prompt)
    return expanded
```

### Re-ranking
```python
def rerank_results(results, query, model):
    """Use cross-encoder for more accurate re-ranking"""
    # Cross-encoders provide better relevance scoring
    pairs = [(query, result['content']) for result in results]
    scores = model.predict(pairs)

    # Re-sort by cross-encoder scores
    for i, score in enumerate(scores):
        results[i]['rerank_score'] = score

    results.sort(key=lambda x: x['rerank_score'], reverse=True)
    return results
```

### Filtering and Metadata
```python
def filtered_search(query, vector_db, filters, top_k=5):
    """Search with metadata filters"""
    # Apply filters before or after vector search
    results = vector_db.search_with_filters(query, filters, top_k=top_k)
    return results
```

## Performance Optimization

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed_query(query, model_name):
    """Cache query embeddings to avoid recomputation"""
    model = load_model(model_name)
    return model.encode([query])[0]
```

### Batch Processing
```python
def batch_search(queries, vector_db, batch_size=10):
    """Process multiple queries efficiently"""
    all_results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batch_embeddings = embed_queries_batch(batch)
        batch_results = vector_db.batch_search(batch_embeddings)
        all_results.extend(batch_results)

    return all_results
```

## Evaluation and Tuning

### Search Quality Metrics
```python
def evaluate_search_quality(queries, ground_truth, search_function):
    """Evaluate search performance"""
    precision_scores = []
    recall_scores = []

    for query, relevant_docs in zip(queries, ground_truth):
        results = search_function(query)
        retrieved_docs = [r['content'] for r in results]

        # Calculate precision@K and recall@K
        precision = calculate_precision(retrieved_docs, relevant_docs, k=5)
        recall = calculate_recall(retrieved_docs, relevant_docs, k=5)

        precision_scores.append(precision)
        recall_scores.append(recall)

    return {
        'avg_precision': np.mean(precision_scores),
        'avg_recall': np.mean(recall_scores)
    }
```

### Parameter Tuning
- **top_k**: Number of documents to retrieve
- **similarity_threshold**: Minimum similarity score
- **reranking_threshold**: When to apply re-ranking
- **embedding_model**: Different models for different domains

## Code Reference
- `semantic_search.py`: Semantic search implementation
- `cosine_similarity.py`: Similarity metrics and BM25 integration
- `vector_keyword_search.py`: Hybrid search concepts

## Next Steps
With retrieval working, move to [Step 7: LLM Integration](../module07_llm_prompting/07_llm_integration.md) to learn how to generate responses using retrieved documents.