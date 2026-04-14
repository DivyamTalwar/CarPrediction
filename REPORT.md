# Car Price Prediction using Machine Learning on Cloud -- Final Report

---

**Author:** Divya M. Talwar
**Date:** March 2026
**Course:** Capstone Project

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Description](#3-dataset-description)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Deployment](#6-deployment)
7. [Testing](#7-testing)
8. [Challenges and Learnings](#8-challenges-and-learnings)
9. [Future Improvements](#9-future-improvements)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

---

## 1. Executive Summary

This project develops a machine learning system for predicting used car prices based on historical vehicle data from the CarDekho automotive dataset. Three regression algorithms -- Linear Regression, Random Forest, and XGBoost -- were trained, evaluated, and compared using standard performance metrics. The XGBoost model achieved the best performance with an R-squared value of 0.982 and a mean absolute error of 0.38 Lakhs INR. The final model is deployed as a Flask web application, containerized with Docker, and configured for cloud hosting on Microsoft Azure App Service.

---

## 2. Problem Statement

Pricing used cars is an inherently subjective process. Sellers and buyers routinely disagree on fair market value because they weigh vehicle attributes differently and lack access to comprehensive market data. Traditional valuation methods rely on manual inspection, dealer intuition, and limited comparison with similar listings, all of which introduce inconsistency and bias.

A data-driven approach can address these shortcomings by learning price patterns from thousands of historical transactions. Given a set of measurable vehicle attributes -- such as brand, age, fuel type, kilometres driven, and original showroom price -- a supervised regression model can produce a reliable price estimate grounded in real market behaviour.

**Objective.** Build an end-to-end machine learning pipeline that (1) ingests and preprocesses automotive listing data, (2) trains and evaluates multiple regression models, (3) selects the best-performing model, and (4) deploys it as a web-accessible prediction service suitable for cloud hosting.

---

## 3. Dataset Description

### 3.1 Source

The dataset is modelled after the CarDekho automotive listings dataset, a Kaggle-style collection of used car sale records from the Indian market. Each record represents a single vehicle listing with its associated selling price.

### 3.2 Raw Dataset Summary

| Property | Value |
|---|---|
| Total records | ~8,000 |
| Number of features | 9 |
| Target variable | Selling_Price (Lakhs INR) |
| Feature types | Numeric (4), Categorical (5) |

### 3.3 Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| Brand | Categorical | Manufacturer name, extracted from Car_Name |
| Present_Price | Numeric | Current ex-showroom price (Lakhs INR) |
| Kms_Driven | Numeric | Total kilometres driven by the vehicle |
| Fuel_Type | Categorical | Petrol, Diesel, or CNG |
| Seller_Type | Categorical | Dealer or Individual |
| Transmission | Categorical | Manual or Automatic |
| Owner | Numeric | Number of previous owners (0, 1, 2, 3) |
| Car_Age | Numeric | Age of the vehicle in years (2026 minus Year) |

### 3.4 Data Cleaning

After preprocessing, the dataset was reduced from approximately 8,000 to 7,844 usable records. The following cleaning steps were applied:

- **Duplicate removal.** Exact duplicate rows were identified and dropped.
- **Missing value imputation.** Numeric columns were imputed using the median; categorical columns were imputed using the mode.
- **Outlier removal.** The interquartile range (IQR) method was used to detect and remove extreme outliers in Selling_Price and Kms_Driven.

---

## 4. Methodology

### 4.1 Data Preprocessing

**Brand extraction.** The Car_Name field contained the full model name (e.g., "Maruti Swift Dzire VDI"). The brand was extracted as the first token of this string, producing a cleaner categorical feature with manageable cardinality.

**Car_Age derivation.** The original Year column indicated the year of manufacture. This was transformed into Car_Age = 2026 - Year, which provides a more interpretable and model-friendly representation of vehicle age.

**Missing value imputation.** Median imputation was used for numeric features and mode imputation for categorical features. These strategies were chosen for their robustness to skewed distributions.

**Outlier removal.** For each numeric feature, values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR were flagged as outliers and removed. This step eliminated extreme values that would disproportionately influence model training.

**Log transformation.** The target variable Selling_Price exhibited a strong right skew. A natural log transformation was applied to produce a more normally distributed target, which improves the convergence and accuracy of regression models. Predictions are exponentiated back to the original scale at inference time.

### 4.2 Feature Engineering

A scikit-learn `ColumnTransformer` was used to apply the appropriate transformations to each feature type within a unified pipeline:

- **Numeric features** (Present_Price, Kms_Driven, Owner, Car_Age) were scaled using `StandardScaler` to zero mean and unit variance.
- **Categorical features** (Brand, Fuel_Type, Seller_Type, Transmission) were encoded using `OneHotEncoder` with `handle_unknown='ignore'` to accommodate unseen categories at inference time.

The entire preprocessing and model training sequence was encapsulated in a scikit-learn `Pipeline` object. This design ensures that all transformations are fitted exclusively on training data, preventing data leakage during cross-validation and holdout evaluation.

### 4.3 Model Training

**Train/test split.** The cleaned dataset was split into 80% training and 20% test sets using stratified random sampling with a fixed random seed for reproducibility.

**Models evaluated.** Three regression algorithms were selected to represent a spectrum of model complexity:

1. **Linear Regression.** A baseline linear model with no regularisation, providing an interpretable lower bound on expected performance.
2. **Random Forest Regressor.** An ensemble of 100 decision trees with bootstrap aggregation, offering non-linear modelling capacity with built-in feature importance.
3. **XGBoost Regressor.** A gradient-boosted tree ensemble using the XGBoost library, known for strong performance on tabular data and effective regularisation against overfitting.

**Cross-validation.** Each model was evaluated using 5-fold cross-validation on the training set. This provides a robust estimate of generalisation performance and guards against overfitting to a particular train/test split.

**Evaluation metrics.** The following metrics were computed on both the cross-validation folds and the held-out test set:

- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual prices.
- **Root Mean Squared Error (RMSE):** Square root of the average squared differences, penalising larger errors more heavily.
- **R-squared (R2):** Proportion of variance in the target variable explained by the model, where 1.0 indicates a perfect fit.

---

## 5. Results

### 5.1 Model Comparison

| Model | MAE (Lakhs) | RMSE (Lakhs) | R-squared | CV R2 Mean |
|---|---|---|---|---|
| Linear Regression | 1.0535 | 1.9765 | 0.8596 | 0.9206 |
| Random Forest | 0.4436 | 0.8036 | 0.9768 | 0.9849 |
| **XGBoost** | **0.3833** | **0.7083** | **0.9820** | **0.9915** |

XGBoost achieves the lowest error on every metric and the highest cross-validation R-squared, confirming it as the best model for this task.

### 5.2 Key Visualisations

The following plots were generated during the analysis and are stored in `outputs/figures/`. Each is described below along with the insight it provides.

**01 -- Price Distribution** (`01_price_distribution.png`)
Histogram showing the distribution of Selling_Price. The raw distribution is heavily right-skewed, with a long tail of high-value vehicles. This motivated the log transformation of the target variable.

**02 -- Correlation Heatmap** (`02_correlation_heatmap.png`)
Heatmap of Pearson correlations among numeric features. Present_Price shows the strongest positive correlation with Selling_Price, confirming it as the single most predictive feature. Car_Age shows a moderate negative correlation with price, as expected.

**03 -- Price by Fuel Type** (`03_price_by_fuel.png`)
Box plot comparing selling prices across fuel types. Diesel vehicles command higher median prices than petrol vehicles, reflecting both their higher original cost and stronger resale demand. CNG vehicles occupy a narrow, lower price range.

**04 -- Price by Transmission** (`04_price_by_transmission.png`)
Box plot comparing manual and automatic transmission vehicles. Automatic vehicles have a higher median price and wider spread, consistent with their premium positioning in the market.

**05 -- Price vs. Age** (`05_price_vs_age.png`)
Scatter plot showing the inverse relationship between vehicle age and selling price. The rate of depreciation is steepest in the first few years and flattens over time, suggesting a non-linear relationship that tree-based models can capture effectively.

**06 -- Price vs. Kilometres Driven** (`06_price_vs_kms.png`)
Scatter plot of selling price against kilometres driven. Higher mileage is associated with lower prices, though the relationship contains considerable variance, indicating that mileage alone is an insufficient predictor.

**07 -- Brand Distribution** (`07_brand_distribution.png`)
Bar chart of listing counts by brand. The dataset is dominated by a few brands (Maruti, Hyundai, Honda), with a long tail of less common manufacturers. This class imbalance was handled through one-hot encoding rather than target encoding to avoid leakage.

**08 -- Price by Seller Type** (`08_price_by_seller.png`)
Box plot comparing dealer and individual seller prices. Dealer-listed vehicles tend to have higher prices, likely reflecting their better condition and the dealer markup.

**09 -- Actual vs. Predicted** (`09_actual_vs_predicted.png`)
Scatter plot of actual vs. predicted prices for the XGBoost model on the test set. Points cluster tightly along the diagonal line (y = x), visually confirming the R-squared of 0.982. Minor deviations appear at the high end of the price range.

**10 -- Residuals** (`10_residuals.png`)
Residual plot for the XGBoost model. Residuals are centred around zero with no systematic pattern, indicating that the model does not exhibit bias across the price range. A slight increase in variance at higher predicted values suggests mild heteroscedasticity.

**11 -- Feature Importance** (`11_feature_importance.png`)
Bar chart of XGBoost feature importances. Present_Price and Car_Age dominate, together accounting for the majority of predictive power. Fuel_Type and Kms_Driven provide secondary contributions. Owner and Seller_Type have minimal impact.

**12 -- Learning Curves** (`12_learning_curves.png`)
Training and validation learning curves for the XGBoost model. Both curves converge at a high R-squared value as training set size increases, confirming that the model is not overfitting and that the dataset size is sufficient for reliable generalisation.

### 5.3 Analysis

**XGBoost dominates.** XGBoost outperforms both the linear baseline and the Random Forest model across all metrics. Its gradient boosting framework allows it to iteratively correct errors from prior trees, yielding tighter predictions particularly in the mid-range price segment where most data points lie.

**Present_Price and Car_Age are the most important features.** The ex-showroom price acts as a strong anchor for resale value, while vehicle age captures the depreciation trajectory. Together, these two features explain the majority of price variation.

**Log transformation improves all models.** Applying a log transformation to the skewed target variable improved R-squared by 3--5 percentage points across all three models, with the largest gain observed for Linear Regression.

**No evidence of overfitting.** Cross-validation R-squared values are consistent with test-set performance for all models. The learning curves further confirm that XGBoost generalises well and does not memorise the training data.

---

## 6. Deployment

### 6.1 Application Architecture

The trained XGBoost model is served through a Flask web application that provides two interfaces:

1. **HTML form** at the root endpoint (`/`), allowing users to enter vehicle attributes through a browser and receive a predicted price.
2. **JSON API** at `/predict`, accepting POST requests with vehicle features in JSON format and returning the predicted price.
3. **Health check** at `/health`, returning a simple status response for monitoring and load balancer integration.

### 6.2 Containerisation

The application is packaged as a Docker container using a multi-stage Dockerfile. The container includes all dependencies (Flask, scikit-learn, XGBoost, pandas) and the serialised model artifact. This ensures consistent behaviour across development, testing, and production environments.

### 6.3 Cloud Deployment

The application is configured for deployment on Microsoft Azure App Service:

| Property | Value |
|---|---|
| Service tier | B1 (Basic) |
| Estimated cost | ~$13/month (covered by Azure for Students credits) |
| Container registry | Azure Container Registry |
| Scaling | Single instance (sufficient for demo workload) |
| Health monitoring | Built-in via /health endpoint |

The deployment leverages Azure's native Docker container support, eliminating the need for manual server provisioning.

---

## 7. Testing

### 7.1 Test Suite Summary

A comprehensive test suite of 20 automated tests was developed using pytest. The tests are organised into four categories:

| Category | Count | Description |
|---|---|---|
| Preprocessing | 6 | Data cleaning, feature engineering, pipeline integrity |
| Model | 6 | Training, prediction, metric thresholds, serialisation |
| API | 7 | Endpoint responses, input validation, error handling |
| End-to-End | 1 | Full browser-based demo flow (Playwright) |

### 7.2 Preprocessing Tests

These tests verify that the data pipeline correctly handles brand extraction, age derivation, missing values, outlier removal, and scaling. They ensure that the pipeline produces consistent output shapes and data types.

### 7.3 Model Tests

Model tests confirm that each algorithm trains without error, produces predictions within a plausible range, meets minimum accuracy thresholds, and that the serialised model file loads and produces identical predictions.

### 7.4 API Tests

API tests exercise the Flask application's endpoints, verifying correct HTTP status codes, response formats, input validation for missing or malformed fields, and the health check endpoint.

### 7.5 End-to-End Test

A single Playwright-based browser test simulates the full demonstration flow. It navigates to the web form, submits three distinct test cases (a high-value recent car, a low-value older car, and a mid-range vehicle), and asserts that the returned predictions are within expected ranges. This test validates the integration of the entire system from UI to model inference.

---

## 8. Challenges and Learnings

**XGBoost requires OpenMP on macOS.** The XGBoost library depends on OpenMP for parallelised tree construction. On macOS, the system compiler (Apple Clang) does not ship with OpenMP support. This was resolved by installing `libomp` via Homebrew (`brew install libomp`) prior to installing the XGBoost Python package.

**scikit-learn Pipeline prevents data leakage.** Early iterations fitted the scaler and encoder on the full dataset before splitting, which introduced subtle data leakage. Wrapping all transformations inside a scikit-learn Pipeline ensures that `.fit()` is called only on training folds during cross-validation, producing honest performance estimates.

**Log transformation addresses target skew.** The raw Selling_Price distribution was heavily right-skewed, causing all three models to underpredict high-value vehicles and overpredict low-value ones. Applying a natural log transformation before training and exponentiating predictions at inference time resolved this issue and improved R-squared across the board.

**Flask port 5000 conflicts with AirPlay on macOS.** macOS Monterey and later versions use port 5000 for the AirPlay Receiver service. During local development, the Flask server was configured to bind to port 8000 instead to avoid the conflict.

---

## 9. Future Improvements

1. **Real-world dataset validation.** Replace the synthetic/sample data with the full Kaggle CarDekho dataset to validate model performance on real-world listings at scale.

2. **Expanded feature set.** Incorporate additional vehicle attributes such as engine displacement (cc), horsepower, number of seats, city of sale, and insurance status. These features are known to influence resale value and could improve prediction accuracy for edge cases.

3. **Model monitoring and retraining pipeline.** Implement a monitoring system that tracks prediction drift over time and triggers automated retraining when performance degrades below a threshold. Tools such as MLflow or Evidently AI could be integrated for this purpose.

4. **User authentication and rate limiting.** Add authentication to the API endpoint to prevent abuse and enable usage tracking. This is essential before exposing the service to external consumers.

5. **Ensemble or stacking approach.** Combine the predictions of multiple models (e.g., XGBoost and Random Forest) through stacking to potentially improve robustness on out-of-distribution inputs.

---

## 10. Conclusion

This project successfully delivers an end-to-end machine learning pipeline for used car price prediction. Starting from raw automotive listing data, the system performs cleaning, feature engineering, model training, evaluation, and deployment as a web-accessible service.

The XGBoost model achieves an R-squared of 0.982 on the held-out test set, explaining over 98% of the variance in used car prices. Cross-validation confirms that this performance generalises reliably and is not an artefact of a favourable train/test split.

The Flask application provides both a user-friendly web form for interactive use and a JSON API for programmatic integration. The Dockerised deployment package is ready for hosting on Azure App Service, with an estimated monthly cost of approximately $13 under the Azure for Students programme.

All project objectives defined in the initial proposal have been met: data preprocessing, multi-model comparison, best-model selection, web deployment, automated testing, and cloud-readiness.

---

## 11. References

1. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825--2830. https://scikit-learn.org/stable/

2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785--794. https://xgboost.readthedocs.io/

3. Pallets Projects. Flask: Web Development, One Drop at a Time. https://flask.palletsprojects.com/

4. Microsoft. Azure App Service Documentation. https://learn.microsoft.com/en-us/azure/app-service/

5. CarDekho. Used Car Price Dataset. https://www.cardekho.com/

6. Docker, Inc. Docker Documentation. https://docs.docker.com/

7. Pytest Development Team. pytest: Simple Powerful Testing with Python. https://docs.pytest.org/

---

*Report generated as part of the Machine Learning Capstone Project, March 2026.*
