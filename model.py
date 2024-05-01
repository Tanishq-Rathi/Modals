import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
data = pd.read_csv('job_applications.csv')

# Define features and target variable
X = data.drop('recommended_for_job', axis=1)
y = data['recommended_for_job']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = ['current_salary']
numeric_transformer = StandardScaler()

categorical_features = ['latest_degree', 'specialization', 'currently_employed', 'location']
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Feature scaling and one-hot encoding
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Define a function to preprocess custom inputs
def preprocess_inputs(inputs):
    # Create a DataFrame with the custom inputs
    custom_data = pd.DataFrame(inputs, index=[0])
    # Preprocess the custom inputs using the preprocessor defined earlier
    custom_data_scaled = preprocessor.transform(custom_data)
    return custom_data_scaled

# Load the saved model from the pickle file
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Define custom inputs (replace these with your own inputs)
custom_inputs = {
    'current_salary': [50000],
    'latest_degree': ['Bachelor'],
    'specialization': ['Computer Science'],
    'currently_employed': [0],
    'location': ['New York']
}

# Preprocess the custom inputs
custom_inputs_scaled = preprocess_inputs(custom_inputs)

# Make predictions using the loaded model
predictions = loaded_model.predict(custom_inputs_scaled)

# Print the predictions
print("Predictions:", predictions)
