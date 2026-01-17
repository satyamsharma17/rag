# Step 8: Evaluation and Optimization

## Overview
Evaluation and optimization are crucial for ensuring your RAG system performs well in production. This involves measuring performance, identifying bottlenecks, and implementing improvements across the entire pipeline.

## Key Metrics

### Retrieval Metrics

#### Precision@K and Recall@K
```python
# See precision_problem.py for implementation
def calculate_precision_recall(retrieved_docs, relevant_docs, k=5):
    """Calculate precision and recall at K"""

    retrieved_at_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)

    # True positives: retrieved docs that are relevant
    true_positives = len([doc for doc in retrieved_at_k if doc in relevant_set])

    precision = true_positives / k if k > 0 else 0
    recall = true_positives / len(relevant_docs) if relevant_docs else 0

    return precision, recall
```

#### Mean Average Precision (mAP)
```python
def mean_average_precision(queries, retrieved_results, ground_truth):
    """Calculate mAP across multiple queries"""

    average_precisions = []

    for query_results, query_truth in zip(retrieved_results, ground_truth):
        precisions = []
        relevant_found = 0
        total_relevant = len(query_truth)

        for i, result in enumerate(query_results):
            if result in query_truth:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precisions.append(precision_at_i)

        if precisions:
            ap = sum(precisions) / total_relevant
            average_precisions.append(ap)

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0
```

### Generation Metrics

#### ROUGE Scores (for summarization/extractive tasks)
```python
from rouge_score import rouge_scorer

def calculate_rouge_scores(predictions, references):
    """Calculate ROUGE-1, ROUGE-2, ROUGE-L scores"""

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }

    for pred, ref in zip(predictions, references):
        rouge_scores = scorer.score(ref, pred)

        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            for metric in ['precision', 'recall', 'fmeasure']:
                scores[rouge_type][metric].append(rouge_scores[rouge_type][metric])

    # Calculate averages
    avg_scores = {}
    for rouge_type in scores:
        avg_scores[rouge_type] = {}
        for metric in scores[rouge_type]:
            avg_scores[rouge_type][metric] = sum(scores[rouge_type][metric]) / len(scores[rouge_type][metric])

    return avg_scores
```

#### BERTScore (for semantic similarity)
```python
from bert_score import score

def calculate_bert_score(predictions, references):
    """Calculate BERTScore for semantic similarity"""

    P, R, F1 = score(predictions, references, lang='en', verbose=False)

    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }
```

## End-to-End Evaluation

### RAG-Specific Metrics
```python
def evaluate_rag_system(test_queries, test_answers, rag_pipeline):
    """Comprehensive RAG evaluation"""

    results = {
        'retrieval': {'precision@5': [], 'recall@5': []},
        'generation': {'rouge1_f1': [], 'bert_f1': []},
        'latency': []
    }

    for query, reference_answer in zip(test_queries, test_answers):
        start_time = time.time()

        # Run RAG pipeline
        retrieved_docs, generated_answer = rag_pipeline(query)

        latency = time.time() - start_time

        # Evaluate retrieval
        precision, recall = calculate_precision_recall(
            [doc['content'] for doc in retrieved_docs],
            [reference_answer],  # This would be ground truth relevant docs
            k=5
        )

        results['retrieval']['precision@5'].append(precision)
        results['retrieval']['recall@5'].append(recall)

        # Evaluate generation
        rouge_scores = calculate_rouge_scores([generated_answer], [reference_answer])
        bert_scores = calculate_bert_score([generated_answer], [reference_answer])

        results['generation']['rouge1_f1'].append(rouge_scores['rouge1']['fmeasure'])
        results['generation']['bert_f1'].append(bert_scores['f1'])

        results['latency'].append(latency)

    # Calculate averages
    final_results = {}
    for category in results:
        final_results[category] = {}
        for metric in results[category]:
            final_results[category][metric] = sum(results[category][metric]) / len(results[category][metric])

    return final_results
```

## Performance Optimization

### Identifying Bottlenecks
```python
import time
import cProfile

def profile_rag_pipeline(query, rag_system):
    """Profile RAG pipeline performance"""

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    result = rag_system.process_query(query)
    end_time = time.time()

    profiler.disable()

    print(f"Total time: {end_time - start_time:.3f} seconds")

    # Print profiling results
    import pstats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)

    return result
```

### Optimization Strategies

#### Retrieval Optimization
```python
# Hybrid search implementation
def optimize_retrieval(query, vector_db, keyword_index, alpha=0.5):
    """Combine vector and keyword search"""

    # Vector search
    vector_results = vector_db.search(query, top_k=20)

    # Keyword search
    keyword_results = keyword_index.search(query, top_k=20)

    # Combine with weighting
    combined_scores = {}
    for result in vector_results + keyword_results:
        doc_id = result['id']
        score = combined_scores.get(doc_id, 0)
        if result in vector_results:
            score += alpha * result['similarity']
        else:
            score += (1 - alpha) * result['score']
        combined_scores[doc_id] = score

    # Return top results
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:10]
```

#### Generation Optimization
```python
# Response caching
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_generation(query_hash, context_hash):
    """Cache generated responses"""

    # Check cache
    cache_key = f"{query_hash}_{context_hash}"
    if cache_key in response_cache:
        return response_cache[cache_key]

    # Generate and cache
    response = generate_response(query, context)
    response_cache[cache_key] = response

    return response
```

#### Context Optimization
```python
def optimize_context_window(retrieved_docs, query, llm_model, max_tokens=4000):
    """Optimize context for better performance"""

    # Method 1: Relevance-based selection
    sorted_docs = sorted(retrieved_docs, key=lambda x: x['similarity'], reverse=True)

    # Method 2: Query-focused compression
    compression_prompt = f"""
    Given the query: {query}
    Compress the following documents to the most relevant information:

    {chr(10).join([doc['content'] for doc in sorted_docs[:5]])}

    Compressed context:"""

    compressed_context = llm_model.generate(compression_prompt, max_tokens=max_tokens//2)

    return compressed_context
```

## A/B Testing Framework
```python
def ab_test_rag_systems(system_a, system_b, test_queries, test_answers):
    """Compare two RAG system variants"""

    results_a = evaluate_rag_system(test_queries, test_answers, system_a)
    results_b = evaluate_rag_system(test_queries, test_answers, system_b)

    # Statistical significance testing
    from scipy import stats

    metrics_to_compare = ['retrieval.precision@5', 'generation.rouge1_f1', 'latency']

    comparison_results = {}
    for metric in metrics_to_compare:
        category, submetric = metric.split('.')

        scores_a = results_a[category][submetric]
        scores_b = results_b[category][submetric]

        # T-test for statistical significance
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        comparison_results[metric] = {
            'system_a_avg': sum(scores_a) / len(scores_a),
            'system_b_avg': sum(scores_b) / len(scores_b),
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    return comparison_results
```

## Monitoring and Maintenance

### Production Monitoring
```python
def setup_monitoring(rag_system):
    """Set up monitoring for production RAG system"""

    # Latency tracking
    # Error rate monitoring
    # Quality metric tracking
    # Resource usage monitoring

    # Example: Log key metrics
    def monitored_process_query(query):
        start_time = time.time()

        try:
            result = rag_system.process_query(query)
            latency = time.time() - start_time

            # Log successful request
            logger.info(f"Query processed successfully. Latency: {latency:.3f}s")

            return result

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Query failed. Latency: {latency:.3f}s. Error: {str(e)}")
            raise

    return monitored_process_query
```

### Continuous Improvement
```python
def continuous_evaluation(rag_system, evaluation_dataset):
    """Continuous evaluation with rolling metrics"""

    while True:
        # Sample queries from evaluation dataset
        sample_queries = random.sample(evaluation_dataset, batch_size)

        # Evaluate current performance
        current_metrics = evaluate_rag_system(sample_queries)

        # Compare with baseline
        if current_metrics['retrieval']['precision@5'] < baseline_precision - threshold:
            # Trigger retraining or parameter tuning
            trigger_model_update()

        time.sleep(evaluation_interval)
```

## Code Reference
- `precision_problem.py`: Understanding precision trade-offs in approximate search

## Final Steps
Congratulations! You've built a complete RAG system. Continue monitoring performance, gathering user feedback, and iterating on improvements. Consider advanced topics like multi-modal RAG, agent-based architectures, or domain-specific fine-tuning for further enhancements.