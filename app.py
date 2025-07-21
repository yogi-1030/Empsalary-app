import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("salary_model.pkl")
le_education = joblib.load("le_education.pkl")
le_workclass = joblib.load("le_workclass.pkl")
le_occupation = joblib.load("le_occupation.pkl")
le_gender = joblib.load("le_gender.pkl")
le_income = joblib.load("le_income.pkl")

st.title("Employee Salary Predictor")

age = st.slider("Age", 18, 65, 30)
education = st.selectbox("Education", le_education.classes_)
workclass = st.selectbox("Workclass", le_workclass.classes_)
occupation = st.selectbox("Occupation", le_occupation.classes_)
gender = st.selectbox("Gender", le_gender.classes_)
experience = st.slider("Experience (in years)", 1, 15, 5)

if st.button("Predict Salary"):
    input_data = [[
        age,
        le_education.transform([education])[0],
        le_workclass.transform([workclass])[0],
        le_occupation.transform([occupation])[0],
        le_gender.transform([gender])[0],
        experience
    ]]

    prediction = model.predict(input_data)[0]
    predicted_salary = le_income.inverse_transform([prediction])[0]

    st.success(f"Predicted Salary: {predicted_salary}")


