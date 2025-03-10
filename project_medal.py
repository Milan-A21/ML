

import os
import joblib
import numpy as np
import streamlit as st
import pandas as pd

# Load dataset
DATASET_PATH = r"C:\Users\ASUS\Downloads\olympics_dataset.csv\olympics_dataset.csv"
if os.path.exists(DATASET_PATH):
    try:
        a = pd.read_csv(DATASET_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
else:
    st.error("Dataset file not found.")
    st.stop()

# Function to load the saved model and label encoders
def load_model():
    try:
        model_path = r"C:\Users\ASUS\OneDrive\Desktop\_python\modules\Project\xgboost_best_model.pkl"
        label_paths = {
            "Event": r"C:\Users\ASUS\OneDrive\Desktop\_python\modules\Project\l1",
            "Sex": r"C:\Users\ASUS\OneDrive\Desktop\_python\modules\Project\l3",
            "Team": r"C:\Users\ASUS\OneDrive\Desktop\_python\modules\Project\l4",
            "NOC": r"C:\Users\ASUS\OneDrive\Desktop\_python\modules\Project\l5",
            "Sport": r"C:\Users\ASUS\OneDrive\Desktop\_python\modules\Project\l7",
        }
        model = joblib.load(model_path)
        label_encoders = {key: joblib.load(path) for key, path in label_paths.items()}
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading model or label encoders: {e}")
        st.stop()

# Load the model and label encoders
model, label_encoders = load_model()

# Streamlit app
st.title("Olympic Medal Prediction")
st.write("Predict outcomes based on player details and performance metrics.")

# Add an image
image_path = r"C:\Users\ASUS\Downloads\Untitled.ipynb - JupyterLab cls_files\Beijing-Olympic-medals-013122-Getty-FTR.jpg"
if os.path.exists(image_path):
    st.image(image_path, caption="Welcome to the Olympic Medal Predictor!", use_container_width=True)


# Input fields for user data in the specified order
try:
    required_columns = ["player_id", "Sex", "Team", "NOC", "Sport", "Event"]
    missing_columns = [col for col in required_columns if col not in a.columns]

    if missing_columns:
        st.error(f"Missing columns in dataset: {missing_columns}")
        st.stop()

    input_data = {
        "player_id": st.number_input("Player ID", min_value=0, value=0, step=1),
        "Sex": st.selectbox("Sex", sorted(a["Sex"].dropna().unique())),
        "Team": st.selectbox("Team", sorted(a["Team"].dropna().unique())),
        "NOC": st.selectbox("NOC", sorted(a["NOC"].dropna().unique())),
        "Sport": st.selectbox("Sport", sorted(a["Sport"].dropna().unique())),
        "Event": st.selectbox("Event", sorted(a["Event"].dropna().unique())),
    }
except KeyError as e:
    st.error(f"Error accessing dataset columns: {e}")
    st.stop()

# Function to preprocess input data
def preprocess_input(input_data, label_encoders, expected_features):
    processed_data = []
    try:
        for feature in expected_features:
            if feature in label_encoders:
                processed_data.append(label_encoders[feature].transform([input_data[feature]])[0])
            else:
                processed_data.append(input_data[feature])  # Assume numerical features are directly used
    except Exception as e:
        st.error(f"Error during input preprocessing: {e}")
        st.stop()
    return np.array(processed_data).reshape(1, -1)

# Mapping prediction values to corresponding outcomes
outcome_mapping = {
    0: "Bronze",
    1: "Gold",
    2: "No Medal",
    3: "Silver"
}

# Prediction logic
if st.button("Olympic Medal Prediction"):
    try:
        # Expected features based on model training in the correct order
        expected_features = ["player_id", "Sex", "Team", "NOC", "Sport", "Event"]

        # Preprocess the input data
        processed_data = preprocess_input(input_data, label_encoders, expected_features)

        # Ensure processed data matches model input shape
        if processed_data.shape[1] != model.n_features_in_:
            st.error("Mismatch between processed data and model input features.")
            st.stop()

        # Make a prediction
        prediction = model.predict(processed_data)

        # Map the prediction to the corresponding outcome
        predicted_outcome = outcome_mapping.get(prediction[0], "Unknown Outcome")

        # Display the result
        st.success(f"The predicted outcome for the Olympic medal is: {predicted_outcome}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
