# Step 7: LLM Integration

## Overview
LLM integration is the generation component of your RAG system. It takes the retrieved documents and uses them to generate contextually relevant, accurate responses. This step focuses on effectively prompting LLMs with retrieved context.

## Prompt Engineering for RAG

### Basic RAG Prompt Structure
```python
def create_rag_prompt(query, retrieved_documents):
    """Create a prompt that includes retrieved context"""

    context = "\n\n".join([doc['content'] for doc in retrieved_documents])

    prompt = f"""
Use the following context to answer the question. If the context doesn't contain enough information to fully answer the question, say so.

Context:
{context}

Question: {query}

Answer:"""

    return prompt
```

### Advanced Prompt Techniques

#### Few-Shot Prompting
```python
# See prompt_engineering.py for implementation
def few_shot_rag_prompt(query, retrieved_docs, examples):
    """Include examples of good Q&A pairs"""

    prompt = "Here are some examples of how to answer questions using context:\n\n"

    for example in examples:
        prompt += f"Context: {example['context']}\n"
        prompt += f"Question: {example['question']}\n"
        prompt += f"Answer: {example['answer']}\n\n"

    prompt += f"Now, using this context:\n{retrieved_docs}\n"
    prompt += f"Answer this question: {query}"

    return prompt
```

#### Chain-of-Thought Prompting
```python
def cot_rag_prompt(query, retrieved_docs):
    """Encourage step-by-step reasoning"""

    context = "\n".join([doc['content'] for doc in retrieved_docs])

    prompt = f"""
Context: {context}

Question: {query}

Think step by step:
1. What is the question asking?
2. What relevant information is in the context?
3. How does the information relate to the question?
4. What is the most accurate answer?

Final Answer:"""

    return prompt
```

## Response Generation

### Basic Generation
```python
# See llm.py for implementation
def generate_response(query, retrieved_docs, llm_model, max_tokens=500):
    """Generate response using retrieved context"""

    prompt = create_rag_prompt(query, retrieved_docs)

    response = llm_model.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,  # Lower temperature for factual responses
        stop_sequences=["\n\n", "Question:"]
    )

    return response
```

### Streaming Responses
```python
def generate_streaming_response(query, retrieved_docs, llm_model):
    """Generate response with streaming for better UX"""

    prompt = create_rag_prompt(query, retrieved_docs)

    response_stream = llm_model.generate_stream(
        prompt,
        max_tokens=500,
        temperature=0.1
    )

    for chunk in response_stream:
        yield chunk
```

## Context Management

### Context Window Optimization
```python
def optimize_context(retrieved_docs, max_tokens=4000):
    """Fit retrieved documents into LLM context window"""

    # Sort by relevance
    sorted_docs = sorted(retrieved_docs, key=lambda x: x['similarity'], reverse=True)

    context = ""
    selected_docs = []

    for doc in sorted_docs:
        # Estimate tokens (rough approximation)
        estimated_tokens = len(doc['content'].split()) * 1.3

        if len(context.split()) * 1.3 + estimated_tokens <= max_tokens:
            context += doc['content'] + "\n\n"
            selected_docs.append(doc)
        else:
            break

    return context.strip(), selected_docs
```

### Context Compression
```python
def compress_context(retrieved_docs, llm_model, max_tokens=2000):
    """Use LLM to compress context while preserving key information"""

    full_context = "\n".join([doc['content'] for doc in retrieved_docs])

    compression_prompt = f"""
Summarize the following context, keeping all key facts and information relevant to answering questions:

{full_context}

Summary:"""

    compressed_context = llm_model.generate(
        compression_prompt,
        max_tokens=max_tokens
    )

    return compressed_context
```

## Handling Edge Cases

### Insufficient Context
```python
def handle_insufficient_context(query, retrieved_docs, llm_model):
    """Handle cases where retrieved context is inadequate"""

    # Check if context is sufficient
    context_check_prompt = f"""
Context: {retrieved_docs}
Question: {query}

Does the context contain enough information to fully answer the question?
Answer only YES or NO:"""

    is_sufficient = llm_model.generate(context_check_prompt).strip().upper()

    if "NO" in is_sufficient:
        # Fallback to general knowledge or admit limitation
        fallback_prompt = f"""
I don't have specific information about this in my knowledge base.
Based on general knowledge, answer: {query}

If you're unsure, say so."""
        return llm_model.generate(fallback_prompt)

    # Proceed with normal generation
    return generate_response(query, retrieved_docs, llm_model)
```

### Multiple Perspectives
```python
def generate_multi_perspective_response(query, retrieved_docs, llm_model):
    """Generate response considering multiple viewpoints"""

    perspectives_prompt = f"""
Context: {retrieved_docs}
Question: {query}

Consider this question from multiple perspectives and provide a balanced answer that acknowledges different viewpoints if applicable."""

    return llm_model.generate(perspectives_prompt)
```

## Response Post-processing

### Fact Verification
```python
def verify_facts(response, retrieved_docs, llm_model):
    """Verify that response is supported by context"""

    verification_prompt = f"""
Response: {response}
Context: {retrieved_docs}

Does this response accurately reflect the information in the context?
Identify any claims that are not supported by the context.

Verification:"""

    verification = llm_model.generate(verification_prompt)

    # Flag unsupported claims
    if "not supported" in verification.lower():
        response += "\n\n[Note: Some parts of this answer may not be fully supported by the provided context.]"

    return response
```

### Source Attribution
```python
def add_source_attribution(response, retrieved_docs):
    """Add references to source documents"""

    response += "\n\nSources:"
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get('source', f'Document {i}')
        similarity = doc.get('similarity', 0)
        response += f"\n{i}. {source} (relevance: {similarity:.3f})"

    return response
```

## Performance Optimization

### Caching Responses
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_generate_response(query_hash, context_hash, llm_params):
    """Cache generated responses for identical queries"""
    # Implementation depends on your caching strategy
    pass
```

### Model Selection
```python
def select_model_for_task(query, retrieved_docs):
    """Choose appropriate model based on task complexity"""

    # Simple classification
    if len(retrieved_docs) < 3 or len(query.split()) < 5:
        return "fast_model"  # GPT-2, smaller models

    # Complex reasoning
    else:
        return "powerful_model"  # GPT-4, larger models
```

## Code Reference
- `llm.py`: Basic LLM usage and text generation
- `prompt_engineering.py`: Advanced prompting techniques

## Next Steps
With generation working, proceed to [Step 8: Evaluation and Optimization](../module08_rag/08_evaluation.md) to learn how to measure and improve your RAG system's performance.