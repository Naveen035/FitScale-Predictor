import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('Heightweight.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up the background and page configuration
st.set_page_config(page_title="Height to Weight Predictor", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #6dd5ed, #2193b0);
        color: white;
        text-align: center;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: #f44336;
        color: white;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("Height to Weight Predictor")

# Input section
st.header("Enter your height in feet and inches:")
feet = st.number_input("Feet:", min_value=1, max_value=8, value=5, step=1)
inches = st.number_input("Inches:", min_value=0, max_value=11, value=0, step=1)

# Calculate height in decimal form
height = feet + (inches / 10)

# Make predictions when the button is clicked
if st.button("Predict Weight"):
    # Reshape and predict
    height_array = np.array([[height]])
    predicted_weight = model.predict(height_array)
    st.success(f"Predicted Weight: {predicted_weight[0][0]:.2f} kg")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Streamlit")
