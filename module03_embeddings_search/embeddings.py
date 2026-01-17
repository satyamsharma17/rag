# embeddings.py
# Explanation of Embeddings
# Embeddings are dense vector representations of text, images, or other data.
# They capture semantic meaning and relationships between items.
# Common types: Word embeddings, sentence embeddings, image embeddings.

try:  # Try block to handle missing dependencies gracefully
    from sentence_transformers import SentenceTransformer  # Import sentence transformer for text embeddings
    import numpy as np  # Import NumPy for array operations

    # Load a sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load pre-trained sentence transformer model

    def get_embeddings(texts):  # Define function to generate embeddings for texts
        """
        Generate embeddings for a list of texts.
        """
        return model.encode(texts)  # Use the model to encode texts into embeddings

    # Example texts
    texts = [  # Define sample texts for embedding generation
        "The cat sat on the mat",  # First text sample
        "A feline rested on the rug",  # Second text sample (similar meaning to first)
        "The dog played in the park",  # Third text sample (different topic)
        "Machine learning is fascinating"  # Fourth text sample (different domain)
    ]

    # Generate embeddings
    embeddings = get_embeddings(texts)  # Generate embeddings for all texts

    print("Embeddings shape:", embeddings.shape)  # Print shape of embeddings array (n_texts, embedding_dim)
    print("First embedding (first 10 values):", embeddings[0][:10])  # Print first 10 values of first embedding

    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity  # Import cosine similarity function
    similarities = cosine_similarity(embeddings)  # Calculate pairwise similarities between all embeddings

    print("\nSimilarity matrix:")  # Print header for similarity matrix
    for i, text in enumerate(texts):  # Iterate through texts with indices
        print(f"{i}: {text}")  # Print index and corresponding text
    print(similarities)  # Print the full similarity matrix

    # Embedding models:
    # - Computer Vision: ResNet, VGG, Vision Transformer (ViT)
    # - NLP: Word2Vec, GloVe, BERT, Sentence Transformers
    # - Audio: Wav2Vec, HuBERT

    # Sentence Transformers are specifically designed for sentence-level embeddings
    # all-MiniLM-L6-v2 is a popular model for semantic similarity tasks

except ImportError:  # Handle case where sentence-transformers is not installed
    print("sentence-transformers not installed. Install with: pip install sentence-transformers")  # Print installation instruction
    print("Embeddings are vector representations that capture semantic meaning.")  # Explain what embeddings are
    print("Example: 'king' - 'man' + 'woman' ≈ 'queen' in word embeddings.")  # Give famous word embedding example