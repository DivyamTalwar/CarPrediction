# CarPrediction

Used-car price prediction project built with Python, scikit-learn, XGBoost, and Flask. The repository includes the end-to-end ML workflow, an interactive prediction UI, automated tests, generated visualizations, and deployment-ready packaging.

## Stack

- Python 3.9+
- pandas, NumPy, scikit-learn, XGBoost
- matplotlib, seaborn
- Flask
- pytest, Playwright
- Docker, Vercel config

## Repository Layout

```text
CarPrediction/
├── api/
│   ├── __init__.py
│   ├── app.py
│   ├── static/style.css
│   └── templates/index.html
├── data/
│   ├── raw/car_data.csv
│   └── processed/cleaned_data.csv
├── models/
│   ├── best_model.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
├── outputs/
│   ├── Car_Price_Prediction_Final_Report.pdf
│   ├── figures/
│   ├── metrics/model_comparison.csv
│   └── screenshots/
├── src/
│   ├── __init__.py
│   ├── create_dataset.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── generate_report.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_e2e_playwright.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── Dockerfile
├── PLAN.md
├── REPORT.md
├── requirements.txt
└── vercel.json
```

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.0535 | 1.9765 | 0.8596 |
| Random Forest | 0.4436 | 0.8036 | 0.9768 |
| XGBoost | 0.3833 | 0.7083 | 0.9820 |

## Local Setup

```bash
python3 -m pip install -r requirements.txt
```

If you want to run the Playwright browser test locally, install Chromium once:

```bash
python3 -m playwright install chromium
```

## Run the Pipeline

```bash
python3 src/data_preprocessing.py
python3 src/train.py
python3 src/evaluate.py
python3 src/generate_report.py
```

## Run the Web App

```bash
python3 api/app.py
```

The Flask app defaults to `http://localhost:8080`.

## Run Tests

```bash
python3 -m pytest tests/test_preprocessing.py tests/test_model.py tests/test_api.py tests/test_e2e_playwright.py -q
```

The Playwright suite starts the Flask app automatically unless `BASE_URL` is set. If browser binaries are not installed, that test skips cleanly instead of failing the whole suite.

## API Endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Interactive prediction form |
| `POST` | `/predict` | Price prediction via form or JSON |
| `GET` | `/health` | Health check |

### Example JSON Request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "present_price": 32.0,
    "kms_driven": 60000,
    "fuel_type": "Diesel",
    "seller_type": "Individual",
    "transmission": "Automatic",
    "owner": 1,
    "car_age": 5
  }'
```

## Docker

```bash
docker build -t carprediction .
docker run -p 8080:8080 carprediction
```

## Deployment Notes

- `Dockerfile` packages the Flask application for container deployment.
- `vercel.json` maps the Flask entrypoint for Vercel-style Python hosting.
- The app loads `models/best_model.pkl` at startup, so model artifacts must remain present in the deployed bundle.
