import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the saved models
reg_model = joblib.load('regression_model.pkl')
clf_model = joblib.load('classification_model.pkl')

# Streamlit app setup
st.title("Youth Smoking and Drug Experimentation Analysis")

# Age group and socioeconomic mappings
age_group_mapping = {
    '10-14': 1, '15-19': 2, '20-24': 3, '25-29': 4,
    '30-39': 5, '40-49': 6, '50-59': 7, '60-69': 8,
    '70-79': 9, '80+': 10
}
socioeconomic_status_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

# Input fields for user predictions
age_group_input = st.selectbox("Age Group", list(age_group_mapping.keys()))
gender_input = st.selectbox("Gender", ['Male', 'Female'])
socioeconomic_status_input = st.selectbox("Socioeconomic Status", ['Low', 'Medium', 'High'])
peer_influence_input = st.slider("Peer Influence", 1, 5)
mental_health_input = st.slider("Mental Health Score", 1, 10)
community_support_input = st.slider("Community Support Score", 1, 10)

# Prepare input data with all required features
input_data = pd.DataFrame({
    'Year': [2023],  # default or recent year, adjust as needed
    'Age_Group': [age_group_mapping[age_group_input]],
    'Gender': [1 if gender_input == 'Female' else 0],
    'Socioeconomic_Status': [socioeconomic_status_mapping[socioeconomic_status_input]],
    'Peer_Influence': [peer_influence_input],
    'School_Programs': [3],  # default value; adjust as needed
    'Family_Background': [2],  # default value; adjust as needed
    'Mental_Health': [mental_health_input],
    'Access_to_Counseling': [3],  # default value; adjust as needed
    'Parental_Supervision': [3],  # default value; adjust as needed
    'Substance_Education': [3],  # default value; adjust as needed
    'Community_Support': [community_support_input],
    'Media_Influence': [2]  # default value; adjust as needed
})

# Display predictions
st.write("Smoking Prevalence Prediction:", reg_model.predict(input_data)[0])
st.write("Drug Experimentation Risk:", 'High-Risk' if clf_model.predict(input_data)[0] else 'Low-Risk')



