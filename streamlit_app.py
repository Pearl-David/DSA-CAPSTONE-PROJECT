import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load image
image = Image.open('student banner.jpeg')  # optional image file
st.image(image, caption='Student Grade Prediction System', use_column_width=True)

st.title("🎓 Student Final Grade Predictor")

# Load trained model
model = joblib.load('student_grade_predictor.pkl')

# Get number of features
num_features = model.n_features_in_

st.markdown("Enter the input features below:")

inputs = []
for i in range(num_features):
    value = st.number_input(f"Input {i+1}", step=1.0)
    inputs.append(value)

if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"🎯 Predicted Final Grade: {prediction:.2f}")
