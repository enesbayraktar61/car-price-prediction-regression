# Car Price Prediction (Regression)

This project focuses on predicting car prices using machine learning regression techniques.  
The goal is to estimate the selling price of a car based on its technical and categorical features.

---

## Project Overview

- **Problem Type:** Regression
- **Goal:** Predict car prices based on vehicle features
- **Final Model:** Random Forest Regressor
- **Deployment:** Streamlit app deployed on Hugging Face Spaces

---

## Dataset

- **Name:** CarPrice_Assignment.csv
- **Type:** Tabular data
- **Target Variable:** `price`
- **Features:** Numerical and categorical vehicle attributes

---

## Project Structure

car_price_prediction/
├── app.py
├── requirements.txt
├── models/
│ ├── car_price_model.joblib
│ └── training_columns.json
├── notebooks/
│ └── car_price_prediction.ipynb
├── data/
│ └── CarPrice_Assignment.csv
└── README.md


---

## Methodology

### Exploratory Data Analysis (EDA)
- Target distribution analysis
- Feature inspection and correlation analysis

### Preprocessing
- Missing value handling
- Feature scaling and encoding using sklearn pipelines

### Modeling
- Baseline: Linear Regression
- Final Model: Random Forest Regressor

### Evaluation
- MAE, RMSE, R²

---

## Deployment

- Streamlit web application
- Deployed on Hugging Face Spaces
- Model pipeline loaded directly in the app

---

## How to Run Locally

```bash
git clone https://github.com/enesbayraktar61/car-price-prediction-regression.git
cd car-price-prediction-regression
pip install -r requirements.txt
streamlit run app.py

Conclusion
This project demonstrates an end-to-end machine learning regression workflow including data preprocessing, model training, evaluation, and deployment.

Future Improvements

Hyperparameter tuning
Feature importance analysis
Extended user input support
