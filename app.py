import os
import json
import joblib
import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("Car Price Prediction")
st.write("Enter car features to predict the estimated selling price.")

# Load model (pipeline includes preprocessing)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "car_price_model.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "..", "models", "training_columns.json")

model = joblib.load(MODEL_PATH)

# Load training columns
with open(COLUMNS_PATH, "r") as f:
    training_columns = json.load(f)

st.subheader("Input Features")

# Collect a minimal set of inputs
symboling = st.selectbox("Symboling", [-3, -2, -1, 0, 1, 2, 3], index=3)
fueltype = st.selectbox("Fuel Type", ["gas", "diesel"])
aspiration = st.selectbox("Aspiration", ["std", "turbo"])
carbody = st.selectbox("Car Body", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])

doornumber = st.selectbox("Door Number", ["two", "four"])
drivewheel = st.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])

enginesize = st.number_input("Engine Size", min_value=60, max_value=400, value=130, step=1)
horsepower = st.number_input("Horsepower", min_value=30, max_value=300, value=100, step=1)
curbweight = st.number_input("Curb Weight", min_value=1500, max_value=4500, value=2500, step=10)
citympg = st.number_input("City MPG", min_value=5, max_value=60, value=25, step=1)
highwaympg = st.number_input("Highway MPG", min_value=5, max_value=70, value=30, step=1)

# Create full input row with all required columns
# Default values are set to 0 (numeric) or "unknown" (categorical) where needed
input_data = {col: 0 for col in training_columns}

# Fill known inputs
input_data.update({
    "symboling": symboling,
    "fueltype": fueltype,
    "aspiration": aspiration,
    "carbody": carbody,
    "doornumber": doornumber,
    "drivewheel": drivewheel,
    "enginesize": enginesize,
    "horsepower": horsepower,
    "curbweight": curbweight,
    "citympg": citympg,
    "highwaympg": highwaympg
})

# Convert to DataFrame with correct column order
input_df = pd.DataFrame([input_data], columns=training_columns)

if st.button("Predict Price"):
    # Predict using the trained pipeline
    pred = model.predict(input_df)[0]
    st.success(f"Estimated Car Price: ${pred:,.2f}")

st.caption("Model: RandomForestRegressor + sklearn Pipeline (preprocessing included).")
