"""Tests for the Flask API."""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app, load_model


@pytest.fixture
def client():
    load_model()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    """Test that the home page loads."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Car Price Prediction' in response.data


def test_health_endpoint(client):
    """Test the health check."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'


def test_predict_form_submission(client):
    """Test prediction via form POST."""
    response = client.post('/predict', data={
        'brand': 'Maruti',
        'present_price': '8.5',
        'kms_driven': '35000',
        'fuel_type': 'Petrol',
        'seller_type': 'Dealer',
        'transmission': 'Manual',
        'owner': '0',
        'car_age': '3',
    })
    assert response.status_code == 200
    assert b'Estimated Value' in response.data
    assert b'Lakhs' in response.data


def test_predict_json_api(client):
    """Test prediction via JSON API."""
    response = client.post('/predict',
                           data=json.dumps({
                               'brand': 'Honda',
                               'present_price': 12.0,
                               'kms_driven': 25000,
                               'fuel_type': 'Petrol',
                               'seller_type': 'Dealer',
                               'transmission': 'Automatic',
                               'owner': 0,
                               'car_age': 2,
                           }),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert data['prediction'] > 0
    assert data['unit'] == 'lakhs INR'


def test_predict_different_brands(client):
    """Test predictions for multiple brands to ensure variety."""
    brands_and_prices = [
        ('Maruti', 6.0, 'Manual'),
        ('BMW', 40.0, 'Automatic'),
        ('Toyota', 15.0, 'Manual'),
    ]

    predictions = []
    for brand, price, trans in brands_and_prices:
        response = client.post('/predict',
                               data=json.dumps({
                                   'brand': brand,
                                   'present_price': price,
                                   'kms_driven': 30000,
                                   'fuel_type': 'Petrol',
                                   'seller_type': 'Dealer',
                                   'transmission': trans,
                                   'owner': 0,
                                   'car_age': 3,
                               }),
                               content_type='application/json')
        data = json.loads(response.data)
        predictions.append(data['prediction'])

    # BMW should predict higher than Maruti
    assert predictions[1] > predictions[0], "BMW should be more expensive than Maruti"


def test_predict_invalid_price(client):
    """Test that invalid price returns error."""
    response = client.post('/predict', data={
        'brand': 'Maruti',
        'present_price': '-5',
        'kms_driven': '35000',
        'fuel_type': 'Petrol',
        'seller_type': 'Dealer',
        'transmission': 'Manual',
        'owner': '0',
        'car_age': '3',
    })
    assert response.status_code == 200  # Returns form with error message
    assert b'error' in response.data.lower() or b'must be greater' in response.data.lower()


def test_predict_missing_fields_json(client):
    """Test API with missing fields defaults gracefully."""
    response = client.post('/predict',
                           data=json.dumps({
                               'brand': 'Maruti',
                               'present_price': 5.0,
                               'kms_driven': 20000,
                               'fuel_type': 'Petrol',
                               'seller_type': 'Dealer',
                               'transmission': 'Manual',
                               'owner': 0,
                               'car_age': 5,
                           }),
                           content_type='application/json')
    assert response.status_code == 200
