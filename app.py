import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import numpy as np

# Streamlit page setup
st.set_page_config(page_title="RA Patients' Lung Cancer Risk Prediction", layout="wide")
st.title(" RA Patients' Lung Cancer Risk Prediction")
st.write("Upload dataset → train multiple ML algorithms → compare accuracies → predict for single patient with automatic threshold suggestion.")

# Upload dataset
uploaded_file = st.file_uploader("ra_lung_cancer_dataset_cleaned.csv", type=["xlsx", "csv"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Detect lung_cancer column regardless of case
    target_col = next((col for col in df.columns if col.lower() == "lung_cancer"), None)

    if target_col is None:
        st.error(" Target column 'lung_cancer' not found in dataset!")
        st.stop()

    # Drop unnecessary columns
    drop_cols = ["patient_id", "diagnosis_year"]
    drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    # Encode categorical columns
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "bool"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Show class distribution
    st.subheader("Class Distribution (Before Balancing)")
    st.write(y.value_counts())

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    st.subheader(" Class Distribution (After SMOTE Balancing)")
    st.write(y_res.value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    # Accuracy results
    st.subheader("Model Accuracy Comparison")
    results_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
    st.write(results_df)

    # Plot bar chart
    fig, ax = plt.subplots()
    ax.barh(results_df["Model"], results_df["Accuracy"], color=["#ff9999","#66b3ff","#99ff99","#ffcc99"])
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)

    # Best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    st.success(f"Best Model: {best_model_name} (Accuracy: {results[best_model_name]:.2f})")

    #  Find Best Threshold for Recall 
    probs = best_model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    recall_scores = [recall_score(y_test, (probs >= t).astype(int)) for t in thresholds]

    best_threshold = thresholds[np.argmax(recall_scores)]
    best_recall = max(recall_scores)

    st.info(f"Suggested Best Threshold: {best_threshold:.2f} (Recall: {best_recall:.2f})")

    # --- Single-patient prediction ---
    st.subheader(" Single Patient Prediction")
    input_data = {}
    for col in X.columns:
        if col in label_encoders:
            options = label_encoders[col].classes_
            value = st.selectbox(f"{col}", options)
            input_data[col] = label_encoders[col].transform([value])[0]
        else:
            value = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
            input_data[col] = value

    # Threshold slider (default = suggested threshold)
    threshold = st.slider("Prediction Threshold", 0.1, 0.9, float(best_threshold))

    if st.button("Predict"):
        patient_df = pd.DataFrame([input_data])
        prob = best_model.predict_proba(patient_df)[0][1]

        pred = 1 if prob >= threshold else 0

        if pred == 1:
            st.error(f"Prediction: Lung Cancer (Probability: {prob:.2f}, Threshold: {threshold:.2f})")
        else:
            st.success(f"Prediction: No Lung Cancer (Probability: {1-prob:.2f}, Threshold: {threshold:.2f})")