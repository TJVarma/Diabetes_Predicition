import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

MODEL_PATH = "logistic_regression_model.pkl"  # Save location for the model

st.title("DIABETES PREDICTION")

# Sidebar for user inputs
with st.sidebar:
    st.sidebar.title("USER INPUTS")
    Pregnancies = st.number_input("ENTER PREGNANCIES:", step=1)
    Glucose = st.number_input("ENTER GLUCOSE:", step=1)
    BloodPressure = st.number_input("ENTER BP:", step=1)
    SkinThickness = st.number_input("Enter skin Thickness:", step=1)
    Insulin = st.number_input("ENTER INSULIN:", step=1)
    BMI = st.number_input("ENTER BMI:")
    DPF = st.number_input("ENTER DPF:")
    Age = st.number_input("ENTER AGE:", step=1)

    new_input = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]]
    submit_button = st.button("SUBMIT")

if submit_button:
    if Age == 0:
        st.error("PLEASE CHECK THE AGE AND CLICK SUBMIT")
    else:
        data = pd.read_csv("diabetes.csv")
        st.info("Dataset loaded successfully...")
        st.success("Processing the dataset...")

        data = data.dropna()  # Drop NaN values
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Splitting the dataset into training & testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Training the model with 25 epochs
        logistic_regressor = LogisticRegression(max_iter=1000)  # Ensure convergence
        for epoch in range(25):
            logistic_regressor.fit(x_train, y_train)

        # Predict on test data
        test_predictions = logistic_regressor.predict(x_test)

        # Predict on new user input
        user_prediction = logistic_regressor.predict(new_input)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions)

        st.write(f"**Model Accuracy:** {accuracy:.4f}")
        st.write(f"**Model Precision:** {precision:.4f}")

        # Displaying new user input result
        st.header("NEW USER INPUT RESULTS:")
        if user_prediction[0] == 0:
            st.success("NO POSSIBILITY OF DIABETES")
        else:
            st.error("POSSIBILITY OF DIABETES")
