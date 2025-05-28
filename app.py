from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model, scaler, and encoder
model = joblib.load('lung_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Define feature lists
numerical_features = ['age', 'bmi', 'cholesterol_level']
categorical_features = ['gender', 'country', 'cancer_stage', 'family_history', 
                        'smoking_status', 'hypertension', 'asthma', 'cirrhosis', 
                        'other_cancer', 'treatment_type']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from form
        input_data = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'country': request.form['country'],
            'cancer_stage': request.form['cancer_stage'],
            'family_history': request.form['family_history'],
            'smoking_status': request.form['smoking_status'],
            'bmi': float(request.form['bmi']),
            'cholesterol_level': float(request.form['cholesterol_level']),
            'hypertension': request.form['hypertension'],
            'asthma': request.form['asthma'],
            'cirrhosis': request.form['cirrhosis'],
            'other_cancer': request.form['other_cancer'],
            'treatment_type': request.form['treatment_type']
        }

        # Create DataFrame for numerical and categorical features
        numerical_data = pd.DataFrame([[input_data[feat] for feat in numerical_features]], 
                                     columns=numerical_features)
        categorical_data = pd.DataFrame([[input_data[feat] for feat in categorical_features]], 
                                       columns=categorical_features)

        # Preprocess numerical features
        numerical_scaled = scaler.transform(numerical_data)
        numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_features)

        # Preprocess categorical features
        categorical_encoded = encoder.transform(categorical_data)
        categorical_encoded_df = pd.DataFrame(categorical_encoded, 
                                            columns=encoder.get_feature_names_out(categorical_features))

        # Combine features
        input_processed = pd.concat([numerical_scaled_df, categorical_encoded_df], axis=1)

        # Make prediction
        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)[0][1]

        # Interpret result
        result = 'Survived' if prediction == 1 else 'Not Survived'
        confidence = f"{prediction_proba:.2%}" if prediction == 1 else f"{1 - prediction_proba:.2%}"

        return render_template('result.html', result=result, confidence=confidence)
    
    except Exception as e:
        return render_template('result.html', result=f"Error: {str(e)}", confidence="N/A")

if __name__ == '__main__':
    app.run(debug=True)