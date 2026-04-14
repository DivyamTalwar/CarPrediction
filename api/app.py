"""
Flask API for Car Price Prediction.
Serves an HTML form and a /predict endpoint.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

app = Flask(__name__)

# Load model at startup
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_model.pkl")
model = None


def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")


# Known brands (from training data)
BRANDS = [
    'Audi', 'BMW', 'Chevrolet', 'Ford', 'Honda', 'Hyundai',
    'Mahindra', 'Maruti', 'Mercedes-Benz', 'Nissan', 'Renault',
    'Skoda', 'Tata', 'Toyota', 'Volkswagen'
]


@app.route('/')
def home():
    """Render the prediction form."""
    return render_template('index.html', brands=BRANDS, prediction=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from form or JSON."""
    try:
        if model is None:
            return render_template('index.html', brands=BRANDS, prediction=None,
                                   error="Model not loaded. Please train the model first.")

        # Check if JSON request (API mode)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Extract and validate fields
        brand = str(data.get('brand', 'Maruti'))
        present_price = float(data.get('present_price', 0))
        kms_driven = int(float(data.get('kms_driven', 0)))
        fuel_type = str(data.get('fuel_type', 'Petrol'))
        seller_type = str(data.get('seller_type', 'Dealer'))
        transmission = str(data.get('transmission', 'Manual'))
        owner = int(data.get('owner', 0))
        car_age = int(data.get('car_age', 1))

        # Validate
        if present_price <= 0:
            raise ValueError("Present price must be greater than 0")
        if kms_driven < 0:
            raise ValueError("Kilometers driven cannot be negative")
        if car_age < 0:
            raise ValueError("Car age cannot be negative")

        # Create DataFrame matching training features
        input_data = pd.DataFrame([{
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Owner': owner,
            'Car_Age': car_age,
            'Brand': brand,
            'Fuel_Type': fuel_type,
            'Seller_Type': seller_type,
            'Transmission': transmission,
        }])

        # Predict (model outputs log-transformed price)
        log_prediction = model.predict(input_data)[0]
        prediction = float(np.expm1(log_prediction))  # Inverse log transform
        prediction = max(0.1, round(prediction, 2))  # Floor at 0.1 lakh

        if request.is_json:
            return jsonify({
                'prediction': prediction,
                'unit': 'lakhs INR',
                'input': data,
            })

        return render_template('index.html', brands=BRANDS,
                               prediction=prediction, error=None,
                               form_data=data)

    except ValueError as e:
        error_msg = str(e)
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        return render_template('index.html', brands=BRANDS,
                               prediction=None, error=error_msg)
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        if request.is_json:
            return jsonify({'error': error_msg}), 500
        return render_template('index.html', brands=BRANDS,
                               prediction=None, error=error_msg)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
    })


# Auto-load model on import (needed for Vercel serverless)
load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
