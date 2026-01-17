# dot_product.py
# Explanation of Dot Product
# The dot product (scalar product) is a mathematical operation between two vectors.
# It results in a scalar value and measures the similarity in direction.
# Formula: A · B = Σ(a_i * b_i) = |A| * |B| * cos(θ)

import numpy as np  # Import NumPy for vector operations

def dot_product_manual(vec1, vec2):  # Define function to calculate dot product manually
    """
    Calculate dot product manually.
    """
    if len(vec1) != len(vec2):  # Check if vectors have the same length
        raise ValueError("Vectors must have the same length")  # Raise error if lengths don't match
    return sum(a * b for a, b in zip(vec1, vec2))  # Sum the products of corresponding elements

# Example vectors
vec1 = np.array([1, 2, 3])  # Create first example vector
vec2 = np.array([4, 5, 6])  # Create second example vector
vec3 = np.array([-1, 2, -3])  # Create third vector (somewhat perpendicular to vec1)

print("Vector 1:", vec1)  # Print vec1 components
print("Vector 2:", vec2)  # Print vec2 components
print("Vector 3:", vec3)  # Print vec3 components

# Manual calculation
manual_dot12 = dot_product_manual(vec1, vec2)  # Calculate dot product of vec1 and vec2 manually
manual_dot13 = dot_product_manual(vec1, vec3)  # Calculate dot product of vec1 and vec3 manually

print("\nManual Dot Products:")  # Print section header
print(f"vec1 · vec2 = {manual_dot12}")  # Print manual dot product result
print(f"vec1 · vec3 = {manual_dot13}")  # Print manual dot product result

# Using NumPy
numpy_dot12 = np.dot(vec1, vec2)  # Calculate dot product using NumPy's dot function
numpy_dot13 = np.dot(vec1, vec3)  # Calculate dot product using NumPy's dot function

print("\nNumPy Dot Products:")  # Print section header
print(f"vec1 · vec2 = {numpy_dot12}")  # Print NumPy dot product result
print(f"vec1 · vec3 = {numpy_dot13}")  # Print NumPy dot product result

# Relationship with cosine similarity
def cosine_similarity_from_dot(vec1, vec2):  # Define function to calculate cosine similarity using dot product
    dot = np.dot(vec1, vec2)  # Calculate dot product
    norm1 = np.linalg.norm(vec1)  # Calculate magnitude of vec1
    norm2 = np.linalg.norm(vec2)  # Calculate magnitude of vec2
    return dot / (norm1 * norm2)  # Return cosine similarity: dot product divided by product of magnitudes

cos_sim12 = cosine_similarity_from_dot(vec1, vec2)  # Calculate cosine similarity between vec1 and vec2
cos_sim13 = cosine_similarity_from_dot(vec1, vec3)  # Calculate cosine similarity between vec1 and vec3

print("Cosine Similarities:")  # Print section header
print(f"cos(θ) for vec1, vec2: {cos_sim12:.3f}")  # Print cosine similarity with 3 decimal places
print(f"cos(θ) for vec1, vec3: {cos_sim13:.3f}")  # Print cosine similarity with 3 decimal places

# Applications in ML:
# - Neural network forward pass (weighted sum)
# - Cosine similarity for text/document comparison
# - Projection of vectors onto axes
# - Matrix multiplication (dot product of rows/columns)