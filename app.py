import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifacts():
    model = pickle.load(open('disease_model.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    symptoms_list = pickle.load(open('symptoms_list.pkl', 'rb'))
    return model, le, symptoms_list


model, le, symptoms_list = load_artifacts()

st.title("🩺 Disease Prediction System")
st.write("Select your symptoms below, and the model will predict the most likely disease.")

st.markdown("### Select your symptoms")
selected_symptoms = st.multiselect(
    label="Choose symptoms:",
    options=sorted(symptoms_list),
    placeholder="Type to search symptoms...",
    label_visibility="collapsed"
)

if st.button("Predict Disease 🔍", type="primary"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # build the input vector: 1 for selected symptoms, 0 otherwise
        input_vector = pd.DataFrame(
            [[1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]],
            columns=symptoms_list
        )

        prediction = model.predict(input_vector)[0]
        disease_name = le.inverse_transform([prediction])[0]

        # confidence from the model's predicted probabilities
        probabilities = model.predict_proba(input_vector)[0]
        confidence = max(probabilities) * 100

        st.success(f"### Predicted Disease: {disease_name}")
        st.write(f"**Confidence:** {confidence:.1f}%")

        st.caption(
            "⚠️ This is a demo ML project trained on a small synthetic dataset. "
            "It is not a substitute for professional medical advice."
        )