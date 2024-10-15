import streamlit as st
import numpy as np
import pandas as pd
import joblib  # For loading the model
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = joblib.load("final_model.pkl")

# Define the categories for OneHotEncoder (these should match what you trained your model on)
categories = [
    ['SpiceJet', 'IndiGo', 'AirIndia', 'Go_First', 'Air_India', 'Vistara'],  # Airline
    ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata'],     # Source City
    ['Morning', 'Early_Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'],  # Departure Time
    ['Morning', 'Early_Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'],  # Arrival Time
    ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata'],     # Destination City
    ['Business', 'Economy']  # Flight Class
]

# Initialize the OneHotEncoder
encoder = OneHotEncoder(categories=categories, handle_unknown='ignore')

# Streamlit app layout
st.title("Airline Price Prediction App")
st.write("""
### Predict the price of airline tickets based on input features.
Fill out the form below and click on "Predict" to get the price prediction.
""")

# User input fields
airline = st.selectbox('Select Airline', categories[0])
source_city = st.selectbox('Source City', categories[1])
departure_time = st.selectbox('Departure Time', categories[2])
arrival_time = st.selectbox('Arrival Time', categories[3])
destination_city = st.selectbox('Destination City', categories[4])
flight_class = st.selectbox('Flight Class', categories[5])
stops = st.number_input("Enter the number of stops", min_value=0, max_value=3, step=1)
duration = st.number_input("Enter the duration (hours)", min_value=0.0, step=0.1)
days_left = st.number_input("Enter the days left until the flight", min_value=0, step=1)

# When the user clicks 'Predict'
if st.button("Predict"):
    # Prepare the input data
    categorical_features = [[airline, source_city, departure_time, arrival_time, destination_city, flight_class]]
    numerical_features = np.array([[stops, duration, days_left]])

    # Encode categorical features
    encoder.fit(categorical_features)
    encoded_categorical = encoder.transform(categorical_features).toarray()

    # Concatenate categorical and numerical features
    input_features = np.hstack([encoded_categorical, numerical_features])

    # Ensure input is 2D array
    input_features = input_features.reshape(1, -1)

    # Display input features for debugging
    st.write("Input Features (Shape):", input_features.shape)

    # Predict using the loaded model
    try:
        prediction = model.predict(input_features)
        st.success(f"Predicted Flight Price: {prediction[0]:.2f}INR")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Optional: Show model performance metrics
if st.checkbox("Show Model Performance"):
    st.write("You could display the model's performance here if test data is available.")
