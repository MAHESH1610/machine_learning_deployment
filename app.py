import pandas as pd
from flask import Flask, request, jsonify
import joblib
import numpy as np

# --- Preprocessing functions (copied from the notebook) ---
# Function to convert 'Annual-Percentage' range to a numerical value (midpoint)
def annual_percentage_to_num(percentage_str):
    if isinstance(percentage_str, str):
        if '-' in percentage_str:
            lower, upper = map(int, percentage_str.replace('%', '').split('-'))
            return (lower + upper) / 2
        elif 'Above' in percentage_str:
            return int(percentage_str.replace('Above ', '').replace('%', '')) + 5 # Assuming a midpoint above the lower bound
        elif 'Below' in percentage_str:
            return int(percentage_str.replace('Below ', '').replace('%', '')) - 5 # Assuming a midpoint below the upper bound
        else:
            try:
                return float(percentage_str.replace('%', ''))
            except ValueError:
                return None
    return percentage_str

# Function to convert 'Income' range to a numerical value (midpoint in Lakhs)
def income_to_num(income_str):
    if isinstance(income_str, str):
        income_str = income_str.replace('L', '').replace(' ', '').replace('Upto', '0-')
        if '-' in income_str:
            lower_str, upper_str = income_str.split('-')
            lower = float(lower_str) if lower_str else 0.0
            upper = float(upper_str) if upper_str else lower + 1.0 # Handle cases like '5-' as '5-6' for midpoint
            return (lower + upper) / 2
        else:
            try:
                return float(income_str)
            except ValueError:
                return None
    return income_str

# --- Load the trained model and other necessary artifacts ---
model = joblib.load('random_forest_model.joblib')

# These are the columns expected by the model after preprocessing
# We get them from the X.columns in the notebook's kernel state.
model_features = [
    'Annual-Percentage-Numeric',
    'Income-Numeric',
    'Education Qualification_Postgraduate',
    'Education Qualification_Undergraduate',
    'Gender_Male',
    'Community_Minority',
    'Community_OBC',
    'Community_SC/ST',
    'Religion_Hindu',
    'Religion_Muslim',
    'Religion_Others',
    'Exservice-men_Yes',
    'Disability_Yes',
    'Sports_Yes',
    'India_Out'
]

# Median income for imputation (from kernel state)
median_income = 0.75

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data])

        # Apply numerical conversions
        input_df['Annual-Percentage-Numeric'] = input_df['Annual-Percentage'].apply(annual_percentage_to_num)
        input_df['Income-Numeric'] = input_df['Income'].apply(income_to_num)

        # Drop original columns (Name was already dropped earlier in notebook)
        input_df = input_df.drop(columns=['Annual-Percentage', 'Income'])

        # Handle categorical features using one-hot encoding
        # Ensure all expected dummy columns are present, fill with 0 if not
        categorical_cols = ['Education Qualification', 'Gender', 'Community', 'Religion', 'Exservice-men', 'Disability', 'Sports', 'India']

        # Create dummy variables for all categories, then select only the ones the model was trained on
        temp_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # Add missing columns (that would be 0 for this input) and ensure order
        processed_input = pd.DataFrame(columns=model_features)
        for feature in model_features:
            if feature in temp_df.columns:
                processed_input[feature] = temp_df[feature]
            else:
                processed_input[feature] = 0 # Default to 0 if the dummy column is not present

        # Impute missing 'Income-Numeric' if any (shouldn't be if income_to_num works, but for safety)
        processed_input['Income-Numeric'].fillna(median_income, inplace=True)

        # Ensure boolean columns are converted to appropriate types for the model (e.g., int/float)
        for col in processed_input.columns:
            if processed_input[col].dtype == 'bool':
                processed_input[col] = processed_input[col].astype(int)

        # Make prediction
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)

        return jsonify({
            'prediction': int(prediction[0]),
            'probability_class_0': probability[0][0],
            'probability_class_1': probability[0][1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from outside the container
    app.run(host='0.0.0.0', port=5000)
