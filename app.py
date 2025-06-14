
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Spot the Scam", layout="wide")

# Load model and encoder
model = joblib.load("fraud_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("🚨 Spot the Scam – Job Fraud Detection")

st.write("""
Upload a job listing CSV and let our machine learning model tell you which jobs might be scams.
""")

# File upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show preview
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # Basic preprocessing (optional: expand this for full logic)
    text_cols = ['title', 'description']
    df[text_cols] = df[text_cols].fillna('missing')

    cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('missing')
        else:
            df[col] = 'missing'  # Add missing columns if not present

    encoded_cat = encoder.transform(df[cat_cols])

    # Combine with text features (this is dummy logic – expand with your vectorizer)
    # final_features = np.hstack([encoded_cat.toarray(), tfidf_vectors])
    # For now, assume encoded_cat is enough
    preds = model.predict(encoded_cat)
    probs = model.predict_proba(encoded_cat)[:, 1]

    df['Prediction'] = preds
    df['Fraud Probability'] = probs

    st.subheader("🧾 Results")
    st.dataframe(df[['title', 'location', 'Prediction', 'Fraud Probability']])

    st.subheader("📈 Fraud Probability Distribution")
    st.bar_chart(df['Fraud Probability'])

    fraud_count = df['Prediction'].value_counts()
    st.subheader("📊 Real vs Fake Job Count")
    st.write(fraud_count)

    st.subheader("🔍 Top 10 Most Suspicious Jobs")
    top_frauds = df.sort_values(by='Fraud Probability', ascending=False).head(10)
    st.table(top_frauds[['title', 'location', 'Fraud Probability']])
else:
    st.info("Please upload a CSV file to begin.")
