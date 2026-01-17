# Step 2: Text Chunking

## Overview
Text chunking is the process of breaking down large documents into smaller, manageable pieces. This is crucial for RAG systems because LLMs have token limits and work better with focused, relevant content.

## Why Chunking Matters

### LLM Limitations
- **Token Limits**: Most LLMs have maximum context windows (e.g., 4096-8192 tokens)
- **Processing Efficiency**: Smaller chunks process faster and more accurately
- **Relevance**: Focused chunks provide more precise information

### Chunking Strategies

#### Fixed-Size Chunking
- Simple approach using character or token counts
- May break sentences or concepts
- Easy to implement but less semantic

#### Sentence-Based Chunking
- Respects linguistic boundaries
- Maintains semantic coherence
- Better for natural language understanding

#### Semantic Chunking
- Groups content by topic or meaning
- Most sophisticated approach
- Requires advanced NLP techniques

## Implementation

### Basic Character-Based Chunking
```python
# See chunking.py for complete implementation
def custom_chunking(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundaries
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size - 100:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks
```

### Advanced Chunking with Libraries

#### LangChain Text Splitters
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_text(document)
```

#### spaCy Linguistic Chunking
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Sentence-based chunks
sentences = [sent.text for sent in doc.sents]

# Noun phrase chunks
noun_phrases = [chunk.text for chunk in doc.noun_chunks]
```

## Best Practices

### Chunk Size Considerations
- **Too Small**: May lose context and semantic meaning
- **Too Large**: May exceed token limits or dilute relevance
- **Overlap**: Include overlap between chunks to maintain continuity

### Quality Metrics
- **Semantic Coherence**: Chunks should contain complete thoughts
- **Information Density**: Balance detail with conciseness
- **Retrieval Performance**: Test chunking impact on search quality

## Code Reference
- `chunking.py`: Complete chunking implementations with LangChain and spaCy

## Next Steps
With your documents properly chunked, move to [Step 3: Embeddings](../module03_embeddings_search/03_embeddings.md) to learn how to convert text chunks into vector representations.