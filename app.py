
import json
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "spam_model.joblib")

with open(MODEL_DIR / "evaluation.json", "r") as f:
    results = json.load(f)

st.set_page_config(
    page_title="Spam Email & SMS Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Spam Email & SMS Detection")
st.write(
    "Machine learning project using TF-IDF "
    "and Multinomial Naive Bayes."
)

predict_tab, evaluation_tab = st.tabs(
    ["Spam Detection", "Model Evaluation"]
)

with predict_tab:
    st.subheader("Check a message")

    message_type = st.selectbox(
        "Select message type",
        ["Email", "SMS"]
    )

    message = st.text_area(
        "Enter your email or SMS message",
        height=180,
        placeholder="Paste your message here..."
    )

   
if st.button("Detect Spam", type="primary"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        # Calculate spam probability
        spam_index = list(
            model.named_steps["classifier"].classes_
        ).index("spam")

        spam_probability = model.predict_proba([message])[0][spam_index]

        # Use threshold selected during model training
        threshold = results.get("spam_threshold", 0.50)

        prediction = (
            "spam"
            if spam_probability >= threshold
            else "not_spam"
        )

        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: NOT SPAM")

        st.write(f"**Spam score:** {spam_probability:.1%}")
        st.caption(f"Spam decision threshold: {threshold:.2f}")

        st.caption(
            f"Input type selected: {message_type}. "
            "The same trained model processes both types."
        )


with evaluation_tab:
    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "Test Accuracy",
        f"{results['accuracy'] * 100:.2f}%"
    )
    col2.metric(
        "Test Samples",
        results["test_samples"]
    )

    st.subheader("Classification Report")

    report = results["classification_report"]
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.subheader("Confusion Matrix")
    st.caption("Rows = actual class; columns = predicted class.")

    matrix = pd.DataFrame(
        results["confusion_matrix"],
        index=results["class_labels"],
        columns=results["class_labels"]
    )

    st.dataframe(matrix, use_container_width=True)

    st.subheader("Confusion Matrix Heatmap")
    st.write(
        "Actual Not Spam vs predicted Not Spam/Spam, "
        "and actual Spam vs predicted Not Spam/Spam."
    )
    st.bar_chart(matrix)

st.divider()
st.caption(
    "ITL701 Data Science Lab | "
    "TF-IDF + Multinomial Naive Bayes"
)
