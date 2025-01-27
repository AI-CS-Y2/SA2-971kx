import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from a CSV file
data = pd.read_csv('chest_x-ray.csv') 

# Preprocess the data:
# Convert the 'sex' column (which is categorical) into numerical values
# 'M' becomes 1, 'F' becomes 0
data['sex'] = data['sex'].map({'M': 1, 'F': 0})

# Define the features (X) and the target variable (y)
# We drop non-relevant columns like 'dcm_path', 'annotation_path', and 'StudyInstanceUID'
X = data.drop(columns=['Pneumothorax', 'dcm_path', 'annotation_path', 'StudyInstanceUID'])
y = data['Pneumothorax']

# Split the dataset into training and testing sets
# We use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features by scaling them to have zero mean and unit variance
# This helps to improve model performance by bringing all features to the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest model with 100 trees and a fixed random state for reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)  # Train the model on the scaled training data

# Make predictions using the trained model on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model’s performance by calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy of the model
# We multiply by 100 to convert the accuracy to a percentage format
print(f'Accuracy: {accuracy * 100:.2f}%')
