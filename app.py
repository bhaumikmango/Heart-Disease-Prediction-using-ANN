import streamlit as st
import numpy as np
from keras.models import load_model
import joblib

# Load the saved model and scaler
classifier = load_model('heart_disease_model1.h5')
sc = joblib.load('scaler.pkl')

# Streamlit app
st.title("Heart Disease Prediction")

st.sidebar.header("Input Features")
# Create input fields for features
def user_input_features():
    age = st.sidebar.slider('Age', 9, 77, 54)
    sex = st.sidebar.selectbox('Sex (1=Male, 0=Female)', [1, 0])
    cp = st.sidebar.slider('Chest Pain Type (1-4)', 1, 4, 3)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 131)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar (>120 mg/dl, 1=Yes, 0=No)', [1, 0])
    restecg = st.sidebar.slider('Resting ECG Results (0-2)', 0, 2, 1)
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [1, 0])
    oldpeak = st.sidebar.slider('ST Depression Induced', 0.0, 6.2, 1.0)
    slope = st.sidebar.slider('Slope of the Peak (1-3)', 1, 3, 2)
    ca = st.sidebar.slider('Number of Major Vessels (0-3)', 0, 3, 0)
    thal = st.sidebar.slider('Thalassemia (3=Normal, 6=Fixed, 7=Reversible)', 3, 7, 3)
    
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    return features

input_data = user_input_features()

# Standardize the input data
input_data_scaled = sc.transform(input_data)

# Make predictions
prediction = classifier.predict(input_data_scaled)[0][0]

# Display results
st.subheader("Prediction Result")
if prediction > 0.5:
    st.write(f"High chance of heart disease. (Prediction: {format(prediction)})")
else:
    st.write(f"Low chance of heart disease. (Prediction: {format(prediction)})")

st.write("""
### Note:
This is a demonstration app and not intended for clinical use. Please consult a doctor.
""")