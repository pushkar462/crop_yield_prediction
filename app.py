import streamlit as st
import pandas as pd
import joblib


# Load Model Files

model = joblib.load("models/model.pkl")
feature_columns = joblib.load("models/features.pkl")
model_name = joblib.load("models/model_name.pkl")
metrics = joblib.load("models/metrics.pkl")

st.title("Crop Yield Prediction System")
st.write(f"Using Model: {model_name}")


# Show Model Performance

st.subheader("📊 Model Performance")

st.write(f"MAE  : {metrics['mae']:.2f}")
st.write(f"RMSE : {metrics['rmse']:.2f}")
st.write(f"R²   : {metrics['r2']:.3f}")
st.write(f"Approx Accuracy: {metrics['r2']*100:.2f}%")


# User Input Section

st.subheader("🌱 Enter Crop Details")

area = st.text_input("Area (Country)")
item = st.text_input("Crop Type")
year = st.number_input("Year", min_value=1990, max_value=2030, value=2020)
rainfall = st.number_input("Average Rainfall (mm/year)", value=1000.0)
pesticides = st.number_input("Pesticides Used (tonnes)", value=100.0)
temperature = st.number_input("Average Temperature (°C)", value=25.0)

if st.button("Predict Yield"):

    input_dict = {
        "Area": area,
        "Item": item,
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temperature
    }

    input_df = pd.DataFrame([input_dict])

    # Feature Engineering
    input_df["rain_temp_ratio"] = (
        input_df["average_rain_fall_mm_per_year"] /
        (input_df["avg_temp"] + 1)
    )

    # One-hot Encoding
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.success(f"🌾 Predicted Yield: {prediction:.2f} hg/ha")