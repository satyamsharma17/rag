# numpy.py
# Explanation of NumPy
# NumPy is a fundamental package for scientific computing in Python.
# It provides support for large, multi-dimensional arrays and matrices,
# along with mathematical functions to operate on these arrays.

import numpy as np  # Import NumPy with the standard alias 'np'

# Creating arrays
print("Creating Arrays:")  # Print section header
array1d = np.array([1, 2, 3, 4, 5])  # Create a 1-dimensional array from a Python list
array2d = np.array([[1, 2, 3], [4, 5, 6]])  # Create a 2-dimensional array (matrix) from nested lists
zeros_array = np.zeros((3, 3))  # Create a 3x3 array filled with zeros
ones_array = np.ones((2, 4))  # Create a 2x4 array filled with ones
random_array = np.random.rand(3, 2)  # Create a 3x2 array with random values between 0 and 1

print("1D array:", array1d)  # Print the 1D array
print("2D array:\n", array2d)  # Print the 2D array with newline for formatting
print("Zeros array:\n", zeros_array)  # Print the zeros array
print("Ones array:\n", ones_array)  # Print the ones array
print("Random array:\n", random_array)  # Print the random array

# Array operations
print("\nArray Operations:")  # Print section header
a = np.array([1, 2, 3])  # Create first array for operations
b = np.array([4, 5, 6])  # Create second array for operations

print("Addition:", a + b)  # Element-wise addition of arrays
print("Multiplication:", a * b)  # Element-wise multiplication of arrays
print("Dot product:", np.dot(a, b))  # Dot product (scalar product) of the two arrays

# Matrix operations
matrix1 = np.array([[1, 2], [3, 4]])  # Create first 2x2 matrix
matrix2 = np.array([[5, 6], [7, 8]])  # Create second 2x2 matrix

print("\nMatrix Operations:")  # Print section header
print("Matrix 1:\n", matrix1)  # Print first matrix
print("Matrix 2:\n", matrix2)  # Print second matrix
print("Matrix multiplication:\n", np.dot(matrix1, matrix2))  # Matrix multiplication (not element-wise)

# Statistical operations
data = np.random.randn(1000)  # Generate 1000 random numbers from standard normal distribution
print("\nStatistical Operations:")  # Print section header
print("Mean:", np.mean(data))  # Calculate and print the mean (average) of the data
print("Standard deviation:", np.std(data))  # Calculate and print the standard deviation
print("Min:", np.min(data))  # Find and print the minimum value
print("Max:", np.max(data))  # Find and print the maximum value

# Indexing and slicing
print("\nIndexing and Slicing:")  # Print section header
arr = np.arange(10)  # Create array [0, 1, 2, ..., 9] using arange function
print("Original array:", arr)  # Print the original array
print("First 5 elements:", arr[:5])  # Slice first 5 elements (0 to 4)
print("Elements from index 2 to 7:", arr[2:8])  # Slice elements from index 2 to 7 (inclusive start, exclusive end)
print("Every other element:", arr[::2])  # Slice every other element (step of 2)

# Reshaping
print("\nReshaping:")  # Print section header
arr_1d = np.arange(12)  # Create 1D array with 12 elements [0, 1, 2, ..., 11]
arr_2d = arr_1d.reshape(3, 4)  # Reshape the 1D array into a 3x4 2D array
print("1D array:", arr_1d)  # Print the original 1D array
print("Reshaped to 2D:\n", arr_2d)  # Print the reshaped 2D array

# Key NumPy concepts:
# - ndarray: N-dimensional array object
# - Vectorized operations: Element-wise operations without loops
# - Broadcasting: Automatic expansion of arrays for operations
# - Universal functions (ufuncs): Fast element-wise operations
# - Linear algebra, FFT, random number generation