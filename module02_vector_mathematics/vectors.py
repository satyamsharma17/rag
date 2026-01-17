# vectors.py
# Explanation of Vectors
# Vectors are mathematical objects with magnitude and direction.
# In machine learning, they represent data points in multi-dimensional space.
# Example: [0.00567, 0.00678, 0.79876] - a 3D vector

import numpy as np  # Import NumPy for vector operations

# Creating vectors
vector1 = np.array([0.00567, 0.00678, 0.79876])  # Create a 3D vector as a NumPy array
vector2 = np.array([0.1, 0.2, 0.3])  # Create another 3D vector
vector3 = np.array([1, 2, 3, 4, 5])  # Create a 5D vector (different dimension)

print("Vector 1:", vector1)  # Print vector1 to show its components
print("Vector 2:", vector2)  # Print vector2 to show its components
print("Vector 3:", vector3)  # Print vector3 to show its components (5 dimensions)

# Vector operations
print("\nVector Operations:")  # Print section header

# Addition
result_add = vector1 + vector2  # Add corresponding components of the two vectors
print("Addition:", result_add)  # Print the result of vector addition

# Subtraction
result_sub = vector2 - vector1  # Subtract corresponding components of vector1 from vector2
print("Subtraction:", result_sub)  # Print the result of vector subtraction

# Scalar multiplication
scalar = 2.5  # Define a scalar value
result_scalar = scalar * vector1  # Multiply each component of vector1 by the scalar
print("Scalar multiplication:", result_scalar)  # Print the result of scalar multiplication

# Magnitude (norm)
magnitude1 = np.linalg.norm(vector1)  # Calculate the Euclidean norm (magnitude) of vector1
magnitude2 = np.linalg.norm(vector2)  # Calculate the Euclidean norm (magnitude) of vector2
print("Magnitude of vector1:", magnitude1)  # Print magnitude of vector1
print("Magnitude of vector2:", magnitude2)  # Print magnitude of vector2

# Unit vector (normalization)
unit_vector1 = vector1 / magnitude1  # Divide vector1 by its magnitude to get unit vector
print("Unit vector of vector1:", unit_vector1)  # Print the unit vector (length = 1)
print("Magnitude of unit vector:", np.linalg.norm(unit_vector1))  # Verify unit vector has magnitude 1

# Vector similarity concepts:
# - Euclidean distance: physical distance between points
# - Cosine similarity: angle between vectors (direction similarity)
# - Dot product: projection of one vector onto another

# In RAG and NLP:
# - Document embeddings are high-dimensional vectors
# - Similarity search finds vectors close in vector space
# - Clustering groups similar vectors together