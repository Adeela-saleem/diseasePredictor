# 🩺 Disease Prediction System

A machine learning web app that predicts the most likely disease based on user-selected symptoms.

🔗 **Live Demo:** [Add your Streamlit Cloud link here]

---

## Overview

This project takes a set of symptoms as input and predicts the most probable disease using a trained classification model. Unlike content-based recommendation systems (like my Movie/Song Recommender), this project uses **supervised machine learning** — the model is trained on labeled data and learns patterns between symptoms and diseases.

## How It Works

1. **Data Preprocessing** — Raw dataset had symptoms spread across 17 columns (`Symptom_1` to `Symptom_17`) per patient record. This was converted into a binary feature matrix (131 unique symptoms, each as a 0/1 column).
2. **Label Encoding** — Disease names were encoded into numeric classes using `LabelEncoder`.
3. **Train/Test Split** — Data split 80/20 for training and evaluation.
4. **Model Training** — Compared Random Forest, Decision Tree, and Logistic Regression classifiers.
5. **Evaluation** — Used accuracy score and classification report (precision, recall, F1-score).
6. **Deployment** — Trained model, label encoder, and symptom list saved as `.pkl` files and served through a Streamlit web app.

## Tech Stack

- **Python**
- **pandas, numpy** — data handling
- **scikit-learn** — model training (Random Forest Classifier) and evaluation
- **Streamlit** — web app interface

## Dataset

[Disease Symptom Description Dataset (Kaggle)](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
- 4,920 records across 41 diseases (120 records per disease)
- 131 unique symptoms

## ⚠️ Known Limitation — Model Accuracy

The model achieves **100% accuracy** on the test set. This is **not necessarily a sign of a great model** — it's a sign worth understanding:

- The dataset is synthetic and highly structured, with many duplicate symptom combinations repeated across records for the same disease.
- Because of this, the train/test split often contains nearly identical symptom patterns in both sets, making the prediction task easier than it would be on noisy, real-world clinical data.
- In a production healthcare setting, this dataset and model would need real, diverse patient data, additional validation, and clinical oversight before being used for actual diagnosis support.

This project is intended as a **portfolio/learning project** demonstrating an end-to-end ML pipeline (data cleaning → feature engineering → model training → evaluation → deployment), not as a medical diagnostic tool.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

This is a demo ML project for educational purposes only. It is **not a substitute for professional medical advice, diagnosis, or treatment.**
