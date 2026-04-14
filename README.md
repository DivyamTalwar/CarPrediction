# Car Price Prediction Using Machine Learning on Cloud

A machine learning system that predicts used car selling prices based on vehicle attributes, served via a Flask web application and ready for cloud deployment.

## Project Overview

This capstone project builds and compares multiple regression models to predict used car prices from features such as brand, mileage, fuel type, and age. The best-performing model (XGBoost) is deployed behind a Flask API with both a web form and JSON endpoint, containerized with Docker, and prepared for Azure App Service deployment.

## Tech Stack

- **Language:** Python 3
- **Data & ML:** pandas, NumPy, scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn
- **Web Framework:** Flask
- **Containerization:** Docker
- **Cloud:** Azure App Service

## Dataset

CarDekho-style dataset with approximately 8,000 rows and the following features:

| Feature | Description |
|---|---|
| Brand | Car manufacturer |
| Present_Price | Current ex-showroom price (lakhs) |
| Kms_Driven | Distance driven (km) |
| Fuel_Type | Petrol / Diesel / CNG |
| Seller_Type | Dealer / Individual |
| Transmission | Manual / Automatic |
| Owner | Number of previous owners |
| Car_Age | Age of the vehicle (years) |

## Project Structure

```
CAR PRIDICTION/
├── api/
│   ├── app.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
├── data/
│   ├── raw/
│   │   └── car_data.csv
│   └── processed/
│       └── cleaned_data.csv
├── models/
│   ├── best_model.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   │   └── model_comparison.csv
│   └── screenshots/
├── src/
│   ├── __init__.py
│   ├── create_dataset.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   └── train.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.0535 | 1.9765 | 0.8596 |
| Random Forest | 0.4436 | 0.8036 | 0.9768 |
| **XGBoost (Best)** | **0.3833** | **0.7083** | **0.9820** |

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess the data
python src/data_preprocessing.py

# Train models
python src/train.py

# Evaluate models
python src/evaluate.py

# Start the web application
python api/app.py
# Open http://localhost:8080
```

## How to Test

```bash
python -m pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web form for interactive predictions |
| POST | `/predict` | Submit prediction (form data or JSON) |
| GET | `/health` | Health check |

### Example JSON Request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"Present_Price": 5.59, "Kms_Driven": 27000, "Fuel_Type": "Petrol", "Seller_Type": "Dealer", "Transmission": "Manual", "Owner": 0, "Car_Age": 5}'
```

## Docker

```bash
# Build the image
docker build -t car-price .

# Run the container
docker run -p 8080:8080 car-price
```

## Cloud Deployment

The application is configured for deployment on Azure App Service. The Dockerfile exposes port 8080 and is compatible with Azure's container deployment workflow. To deploy, push the Docker image to Azure Container Registry and configure an App Service instance to pull from it.

## License

This project was developed as an academic capstone project.
