# scikit_learn.py
# Explanation of Scikit-learn
# Scikit-learn is an open-source machine learning library for Python.
# It provides simple and efficient tools for data mining and data analysis.

from sklearn.datasets import make_classification  # Import function to generate synthetic classification datasets
from sklearn.model_selection import train_test_split  # Import function to split data into train/test sets
from sklearn.linear_model import LogisticRegression  # Import logistic regression classifier
from sklearn.metrics import accuracy_score, classification_report  # Import metrics for model evaluation
from sklearn.preprocessing import StandardScaler  # Import standard scaler for feature normalization
from sklearn.pipeline import Pipeline  # Import pipeline for chaining preprocessing and model steps

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)  # Generate 1000 samples with 20 features, 10 informative, for reproducible results

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into 80% training and 20% testing sets

# Create a pipeline with scaling and logistic regression
pipeline = Pipeline([  # Create a pipeline that chains preprocessing and model steps
    ('scaler', StandardScaler()),  # First step: standardize features by removing mean and scaling to unit variance
    ('classifier', LogisticRegression(random_state=42))  # Second step: logistic regression classifier
])

# Train the model
pipeline.fit(X_train, y_train)  # Fit the pipeline (scaler + classifier) on training data

# Make predictions
y_pred = pipeline.predict(X_test)  # Use the trained pipeline to predict labels for test data

# Evaluate
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy: fraction of correct predictions
print(f"Accuracy: {accuracy:.3f}")  # Print accuracy with 3 decimal places

print("\nClassification Report:")  # Print header for detailed classification metrics
print(classification_report(y_test, y_pred))  # Print precision, recall, f1-score for each class

# Example of cross-validation
from sklearn.model_selection import cross_val_score  # Import cross-validation function

scores = cross_val_score(pipeline, X, y, cv=5)  # Perform 5-fold cross-validation on the full dataset
print(f"\nCross-validation scores: {scores}")  # Print the 5 accuracy scores from cross-validation
print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")  # Print mean and 95% confidence interval

# Key components of scikit-learn:
# - Datasets: make_classification, load_iris, etc.
# - Preprocessing: StandardScaler, MinMaxScaler, etc.
# - Model selection: train_test_split, cross_val_score, GridSearchCV
# - Supervised learning: LinearRegression, RandomForestClassifier, SVM
# - Unsupervised learning: KMeans, PCA
# - Metrics: accuracy_score, confusion_matrix, etc.