import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Fraud Detector", layout="wide")

# Title
st.title("💳 Credit Card Fraud Detection App")
st.markdown("Detect fraudulent transactions using a trained Random Forest model.")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# File upload
uploaded_file = st.file_uploader("Upload transaction CSV file (same format as dataset)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    if 'Class' in df.columns:
        X = df.drop("Class", axis=1)
        y = df["Class"]
        y_pred = model.predict(X)

        st.subheader("📊 Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        frauds = sum(y_pred)
        st.info(f"⚠️ Total frauds detected: {frauds}")
        # 🔍 Try Live Prediction
        st.subheader("🔎 Predict a Single Transaction")
        # 🔎 Single Transaction Search & Prediction

        transaction_num = st.text_input("Enter transaction row number (0 to {}):".format(len(df)-1))

        if st.button("Predict Transaction"):
            if transaction_num.isdigit():
                idx = int(transaction_num)
            if 0 <= idx < len(df):
                sample_data = df.drop('Class', axis=1).iloc[idx] if 'Class' in df.columns else df.iloc[idx]
                st.write("Transaction details:")
                st.dataframe(pd.DataFrame(sample_data).transpose())

            # Predict
                pred = model.predict([sample_data])[0]
                if pred == 1:
                    st.error("❌ Fraudulent Transaction")
                else:
                    st.success("✅ Legit Transaction")
            else:
                st.warning("⚠️ Please enter a number between 0 and {}".format(len(df)-1))
        else:
            st.warning("⚠️ Invalid input. Please enter a number.")

    else:
        st.warning("CSV must contain a 'Class' column for evaluation.")
else:
    st.info("Upload a sample transaction CSV to begin.")
