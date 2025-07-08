# 💳 Credit Card Fraud Detector

A machine learning web application to identify fraudulent credit card transactions in real-time. Built for CodeFin 2025.

## 🔍 Features
- Uses a trained Random Forest classifier
- Upload transaction CSV and detect frauds
- Shows classification metrics and confusion matrix
- Live single-transaction prediction
- Deployed on Streamlit Cloud

## 🛠️ Built With
- Python
- scikit-learn
- pandas, matplotlib, seaborn
- Streamlit

## 📂 Dataset
The original dataset used for training is from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Due to GitHub’s 100MB file size limit, it is not included in this repo.

## About Repository 
“Model was trained using **train_model.py** and saved as **fraud_model.pkl**. The app uses the saved model directly for predictions.”

## 🚀 Live Demo
👉 [https://credit-card-fraud-detector-yqjhbwzzgvzmzth7hr3ati.streamlit.app/](#)

## 📂 How to Run Locally
```bash
git clone https://github.com/Naksh2006/credit-card-fraud-detector
cd credit-card-fraud-detector
pip install -r requirements.txt
streamlit run app.py
