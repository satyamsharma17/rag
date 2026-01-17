# semantic_search.py
# Explanation of Semantic Search
# Semantic search goes beyond keyword matching to understand the meaning of words and context.
# It uses embeddings to represent text semantically, allowing for more accurate retrieval.

# Simple example using sentence transformers for semantic similarity

# Try to import required libraries for semantic search functionality
try:
    # Import SentenceTransformer for generating text embeddings
    from sentence_transformers import SentenceTransformer
    # Import cosine_similarity for measuring similarity between embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    # Import numpy for numerical operations and array handling
    import numpy as np

    # Load a pre-trained sentence transformer model for embedding generation
    # 'all-MiniLM-L6-v2' is a lightweight model that converts text to 384-dimensional vectors
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define the main semantic search function
    def semantic_search(query, documents, model, top_k=3):
        """
        Perform semantic search using embeddings.
        This function converts both query and documents to embeddings,
        then finds the most semantically similar documents to the query.
        """
        # Encode the query text into a single embedding vector
        # model.encode([query]) returns a list, so we take the first (and only) element [0]
        query_embedding = model.encode([query])[0]

        # Encode all documents into embedding vectors
        # This creates a matrix where each row is a document's embedding
        doc_embeddings = model.encode(documents)

        # Calculate cosine similarity between query embedding and all document embeddings
        # cosine_similarity returns a matrix, so we take the first row [0] for our single query
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # Get the indices of the top-k most similar documents
        # np.argsort sorts in ascending order, [::-1] reverses to descending, [:top_k] takes top k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Create results list with document text and similarity scores
        # Pair each top document with its similarity score
        results = [(documents[i], similarities[i]) for i in top_indices]

        # Return the top-k most semantically similar documents with their scores
        return results

    # Example documents to demonstrate semantic search
    # These documents contain both relevant and irrelevant content
    documents = [
        "The cat is sleeping on the couch",        # Directly related to query
        "A feline rests on furniture",            # Semantically similar (feline = cat)
        "Dogs love to play fetch",                # Related to pets but different animal
        "The weather is beautiful today",         # Unrelated topic
        "Pets need regular veterinary care"       # Related to pets in general
    ]

    # Example query that should match semantically similar documents
    query = "My cat is resting"

    # Perform semantic search and get top 3 results
    results = semantic_search(query, documents, model)

    # Display the search results
    print(f"Semantic search results for '{query}':")
    # Print each result with its similarity score formatted to 3 decimal places
    for doc, score in results:
        print(f"Score: {score:.3f} - {doc}")

# Handle case where required libraries are not installed
except ImportError:
    # Print helpful error message with installation instructions
    print("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Additional explanation of semantic search concepts
print("Semantic search uses embeddings to understand meaning rather than exact keywords.")
print("Example: Query 'feline resting' would match 'cat sleeping' due to semantic similarity.")