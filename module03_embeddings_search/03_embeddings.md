# Step 3: Creating Embeddings

## Overview
Embeddings are dense vector representations that capture the semantic meaning of text. They transform textual data into numerical vectors that can be processed by machine learning algorithms, enabling semantic search and similarity comparisons.

## Understanding Embeddings

### What are Embeddings?
Embeddings are low-dimensional, dense vectors that represent text in a continuous vector space. Similar meanings are represented by similar vectors, enabling mathematical operations on semantic concepts.

### Types of Embeddings

#### Word Embeddings
- Represent individual words as vectors
- Examples: Word2Vec, GloVe
- Capture word-level semantics

#### Sentence Embeddings
- Represent entire sentences or short passages
- Examples: Sentence Transformers, Universal Sentence Encoder
- Better for document-level understanding

#### Document Embeddings
- Represent longer texts or entire documents
- Often created by averaging sentence embeddings
- Used for document similarity and clustering

## Popular Embedding Models

### Sentence Transformers
```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = ["Hello world", "Hi there", "Greetings"]
embeddings = model.encode(texts)

print(f"Embeddings shape: {embeddings.shape}")  # (3, 384)
```

### Key Models
- **all-MiniLM-L6-v2**: Fast, good quality, 384 dimensions
- **all-mpnet-base-v2**: Higher quality, 768 dimensions
- **text-embedding-ada-002**: OpenAI's embedding model

## Implementation

### Basic Embedding Pipeline
```python
# See embeddings.py for complete implementation
def create_embeddings(chunks, model):
    """
    Convert text chunks to embeddings
    """
    embeddings = model.encode(chunks)
    return embeddings

def save_embeddings(embeddings, metadata, filepath):
    """
    Save embeddings with associated metadata
    """
    data = {
        'embeddings': embeddings,
        'metadata': metadata
    }
    np.save(filepath, data)
```

### Batch Processing
```python
# Process large datasets in batches
batch_size = 100
all_embeddings = []

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    batch_embeddings = model.encode(batch)
    all_embeddings.extend(batch_embeddings)
```

## Quality Considerations

### Choosing the Right Model
- **Speed vs Quality**: Faster models for development, better models for production
- **Dimensionality**: Higher dimensions capture more nuance but require more storage
- **Domain Adaptation**: Fine-tune embeddings for specific domains if needed

### Evaluation Metrics
- **Semantic Similarity**: How well embeddings capture meaning
- **Clustering Quality**: How well similar texts group together
- **Downstream Performance**: Impact on retrieval accuracy

## Storage and Management

### Embedding Storage Options
- **NumPy arrays**: Simple, fast for small datasets
- **Parquet files**: Efficient columnar storage
- **Vector databases**: Specialized storage with indexing (covered in next steps)

### Metadata Association
```python
# Store embeddings with source information
embedding_data = {
    'vectors': embeddings,
    'texts': original_chunks,
    'sources': source_documents,
    'chunk_ids': list(range(len(chunks)))
}
```

## Code Reference
- `embeddings.py`: Complete embedding generation and similarity examples
- `vectors.py`: Understanding vector operations and mathematics

## Next Steps
Now that you have vector representations of your text, proceed to [Step 4: Vector Databases](../module04_vector_databases/04_vector_databases.md) to learn how to efficiently store and query these embeddings.