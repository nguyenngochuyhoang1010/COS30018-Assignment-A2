import streamlit as st
import numpy as np
import joblib
from utils.dataloader import load_traffic_data, scale_data

st.title("Traffic Flow Prediction System (No TensorFlow)")

model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
df = load_traffic_data("data/Scats Data October 2006.csv")
df, _ = scale_data(df)

recent_data = df['volume'].values[-4:]
input_data = np.array([recent_data])

if st.button("Predict next volume"):
    pred = model.predict(input_data)[0]
    actual_pred = scaler.inverse_transform([[pred]])[0][0]
    st.success(f"Predicted traffic volume: {actual_pred:.2f} vehicles")