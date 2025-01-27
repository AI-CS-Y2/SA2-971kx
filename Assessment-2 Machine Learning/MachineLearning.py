import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('chest_x-ray.csv')

# Convert the 'sex' column from categorical (M/F) to numeric (1/0)
# Mapping 'M' to 1 and 'F' to 0
data['sex'] = data['sex'].map({'M': 1, 'F': 0})

# Set up the features (X) and the target variable (y)
# Dropping columns that aren't useful for the model (like file paths and IDs)
X = data.drop(columns=['Pneumothorax', 'dcm_path', 'annotation_path', 'StudyInstanceUID'])
y = data['Pneumothorax']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features by scaling them to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest Classifier model
# Using a fixed random seed for reproducibility
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Use the trained model to make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
# Printing out accuracy and a detailed classification report
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Generate a confusion matrix to see how well the model performed on each class
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using a heatmap for better visualization
fig, ax = plt.subplots()
ax.matshow(conf_matrix, cmap='Blues')  # Use a color map for better clarity

# Annotate the confusion matrix with the values in each cell (to make it easier to interpret)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='red')

# Add titles and labels to the plot for context
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
