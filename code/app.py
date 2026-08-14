"""
app.py

This is a SIMPLE Streamlit web app for our Mental Health & Stress Level
Predictor. It shows a form where the user enters their daily routine
details, and when they click "Predict", it loads our trained model and
shows the predicted Stress Level (Low / Medium / High).

To run this app:
    streamlit run app.py

Make sure you have already run train_model.py once, so that the files
stress_model.pkl, scaler.pkl, diet_encoder.pkl, target_encoder.pkl exist.
"""

import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------
# Load the saved model and preprocessing objects (created by train_model.py)
# -----------------------------------------------------------------
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")
diet_encoder = joblib.load("diet_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -----------------------------------------------------------------
# Page title
# -----------------------------------------------------------------
st.title("Mental Health & Stress Level Predictor")
st.write("Fill in your daily routine details below, and we will predict your stress level.")

# -----------------------------------------------------------------
# Simple input form
# -----------------------------------------------------------------
sleep_hours = st.slider("How many hours do you sleep per day?", 0.0, 12.0, 7.0)
study_work_hours = st.slider("How many hours do you study/work per day?", 0.0, 16.0, 6.0)
screen_time = st.slider("How many hours do you spend on screens per day?", 0.0, 16.0, 5.0)
physical_activity = st.slider("How many hours of physical activity per week?", 0.0, 14.0, 2.0)
pressure_score = st.slider("Rate your academic/work pressure (1 = very low, 10 = very high)", 1, 10, 5)
diet_quality = st.selectbox("How would you rate your diet quality?", ["Poor", "Average", "Good"])

# -----------------------------------------------------------------
# When the user clicks the Predict button
# -----------------------------------------------------------------
if st.button("Predict My Stress Level"):

    # Step 1: Convert the diet quality text into the same numeric code
    # that was used during training (using the SAME encoder object)
    diet_encoded = diet_encoder.transform([diet_quality])[0]

    # Step 2: Put all the inputs into a table with the SAME column order
    # that was used when training the model
    input_data = pd.DataFrame([{
        "Sleep_Hours": sleep_hours,
        "Study_Work_Hours": study_work_hours,
        "Screen_Time": screen_time,
        "Physical_Activity": physical_activity,
        "Pressure_Score": pressure_score,
        "Diet_Quality_Encoded": diet_encoded,
    }])

    # Step 3: Scale the input the same way the training data was scaled
    input_scaled = scaler.transform(input_data)

    # Step 4: Ask the model to predict (this returns an encoded number, e.g. 0, 1, or 2)
    prediction_encoded = model.predict(input_scaled)[0]

    # Step 5: Convert the number back into a readable label ("Low"/"Medium"/"High")
    prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

    # Step 6: Show the result to the user
    st.subheader(f"Predicted Stress Level: {prediction_label}")

    if prediction_label == "Low":
        st.success("Your lifestyle habits look balanced. Keep it up!")
    elif prediction_label == "Medium":
        st.warning("You have a moderate stress level. Try improving sleep or reducing screen time.")
    else:
        st.error("Your stress level appears high. Consider more sleep, less screen time, "
                  "regular exercise, and better diet. If this continues, please talk to someone you trust.")
