import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Load the dataset
data = pd.read_csv('dataset_med.csv')

# Print column names for debugging
print("Dataset Columns:", data.columns.tolist())

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Step 1: Data Preparation
# Define feature columns and target
features = ['age', 'gender', 'country', 'cancer_stage', 'family_history', 
            'smoking_status', 'bmi', 'cholesterol_level', 'hypertension', 
            'asthma', 'cirrhosis', 'other_cancer', 'treatment_type']
target = 'survived'

# Verify all features exist in the dataset
missing_features = [feat for feat in features if feat not in data.columns]
if missing_features:
    raise KeyError(f"Features not found in dataset: {missing_features}")

# Split features into numerical and categorical
numerical_features = ['age', 'bmi', 'cholesterol_level']
categorical_features = ['gender', 'country', 'cancer_stage', 'family_history', 
                        'smoking_status', 'hypertension', 'asthma', 'cirrhosis', 
                        'other_cancer', 'treatment_type']

# Verify numerical and categorical features
missing_numerical = [feat for feat in numerical_features if feat not in data.columns]
missing_categorical = [feat for feat in categorical_features if feat not in data.columns]
if missing_numerical or missing_categorical:
    raise KeyError(f"Missing numerical features: {missing_numerical}, Missing categorical features: {missing_categorical}")

# Step 2: Preprocessing
# Separate numerical and categorical data
X_numerical = data[numerical_features]
X_categorical = data[categorical_features]

# Scale numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)
X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=numerical_features)

# One-hot encode categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X_categorical)
categorical_encoded_columns = encoder.get_feature_names_out(categorical_features)
X_categorical_encoded = pd.DataFrame(X_categorical_encoded, columns=categorical_encoded_columns)

# Combine numerical and categorical features
X = pd.concat([X_numerical_scaled, X_categorical_encoded], axis=1)
y = data[target]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Save the Model, Scaler, and Encoder
joblib.dump(model, 'lung_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

print("Model, scaler, and encoder saved successfully.")