import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Function to load and preprocess data
def data():
    df = pd.read_csv("user_behavior_dataset.csv")
    df["Device Model"] = df["Device Model"].map({"Xiaomi Mi 11": 1, "iPhone 12": 2, "Google Pixel 5": 3, "OnePlus 9": 4, "Samsung Galaxy S21": 5})
    df["Operating System"] = df["Operating System"].map({"Android": 1, "iOS": 2})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 2})
    df = df.drop(["User ID"], axis=1)
    return df

# Function to make predictions
def add_prediction(input_data):
    with open("model.pkl", "rb") as pickle_in:
        classifier = pickle.load(pickle_in)
        
    with open("scaler.pkl", "rb") as scaler_in:
        scaler = pickle.load(scaler_in)
        
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = classifier.predict(input_scaled)
    
    st.subheader("Prediction")
    st.write("The user class is:") 
    st.write(prediction[0])

# Function to create sidebar inputs
def add_sidebar():
    st.sidebar.header("Consumer Attributes")
    df = data()
    
    slider_label = [
        ("Device Model: Xiaomi Mi 11=1, iPhone 12=2, Google Pixel 5=3, OnePlus 9=4, Samsung Galaxy S21=5", "Device Model"),
        ("Operating System: Android=1, iOS=2", "Operating System"),
        ("App Usage Time (min/day)", "App Usage Time (min/day)"),
        ("Screen On Time (hours/day)", "Screen On Time (hours/day)"),
        ("Battery Drain (mAh/day)", "Battery Drain (mAh/day)"),
        ("Number of Apps Installed", "Number of Apps Installed"),
        ("Data Usage (MB/day)", "Data Usage (MB/day)"),
        ("Age", "Age"),
        ("Gender: Male=1, Female=2", "Gender")
    ]
    
    input_dict = {}
    for label, key in slider_label:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0,
            max_value=int(df[key].max())
        )
    return input_dict

# Main function to run the app
def main():
    st.set_page_config(
        page_title="Mobile Consumer Behaviour App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    input_data = add_sidebar()
    
    with st.container():
        st.title("Mobile Consumer Behaviour App")
        st.write("This app is designed for the prediction of customer behavior based on the provided attributes.")
    
    if st.button("Predict"):
        add_prediction(input_data)  
    
if __name__ == "__main__":
    main()
