import streamlit as st
import pandas as pd
import joblib # For model loading
import numpy as np

# Load the trained machine learning model
model = joblib.load('rf_model.pkl')
processor = joblib.load("processor.pkl")

# Create a Streamlit web app
st.title('HIV Prediction System')

# Input form for user data
st.write("Please enter the following information:")
age = st.slider("Age", 0, 100, 30)
std = st.radio("STD", ["Yes", "No"])
marital_status = st.selectbox("Marital Status", ["UNMARRIED", "Married", "Divorced", "Widowed", "Cohabiting"])
education = st.selectbox("Educational Background", ["College Degree", "Senior High School", "Junior High School", "Illiteracy", "Primary School"])
aids_education = st.radio("AIDS Education", ["Yes", "No"])
sex_partner_places = st.selectbox("Places of Seeking Sex Partners", ["Bar", "None", "Park", "Internet", "Public Bath", "Other"])
sexual_orientation = st.selectbox("Sexual Orientation", ["Heterosexual", "Homosexual", "Bisexual"])
drug_taking = st.radio("Drug Taking", ["Yes", "No"])
hiv_test_past_year = st.radio("HIV Test in the Past Year", ["Yes", "No"])

# Make predictions
if st.button("Predict HIV Status"):
   
    input_data = pd.DataFrame({
        'Age': [age],
        'Marital Staus': [marital_status],
        'STD': [std],
        'Educational Background': [education],
        'HIV TEST IN PAST YEAR': [hiv_test_past_year],
        'AIDS education': [aids_education],
        'Places of seeking sex partners': [sex_partner_places],
        'SEXUAL ORIENTATION': [sexual_orientation],
        'Drug- taking': [drug_taking]
    }) 
    transformed_data = processor.transform(input_data)
    prediction = model.predict(transformed_data)

    if prediction[0] == 0:
        st.success("The Predicted HIV Status for the above profile is Negative")
    else:
        st.error("The Predicted HIV Status for the above profile is Positive")
    # st.write(f"Predicted HIV Status: {'Positive' if prediction[0] == 1 else 'Negative'}")

# You can add more features to display the model's confidence or other information.

