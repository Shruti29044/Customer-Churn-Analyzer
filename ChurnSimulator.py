
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# Load Data
st.title("Customer Churn Simulator")
uploaded_file = st.file_uploader("Upload your customer dataset (CSV)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Simple preprocessing: Assume churn column exists
    if 'Churn' not in df.columns:
        st.error("Dataset must contain 'Churn' column (1=churn, 0=no churn)")
    else:
        # Fill missing values for simplicity
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # Select features automatically (excluding 'Churn')
        features = df.drop(columns=['Churn']).select_dtypes(include=[np.number]).columns.tolist()
        st.write("Using features:", features)

        X = df[features]
        y = df['Churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluation
        st.subheader("Model Evaluation")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        st.pyplot(plt)
        
        # Churn Probability Simulation
        st.subheader("Churn Simulation")
        uplift = st.slider("Retention Campaign Effect (reduction in churn probability %):", 0, 50, 10)
        churn_probs = model.predict_proba(X_test)[:, 1]
        adjusted_probs = churn_probs * (1 - uplift / 100)
        adjusted_churn = (adjusted_probs >= 0.5).astype(int)
        reduced_churn_rate = np.mean(adjusted_churn)
        st.write(f"Projected Churn Rate after campaign: {reduced_churn_rate*100:.2f}%")
        
        # Financial Impact
        revenue_per_customer = st.number_input("Average revenue per customer:", min_value=1.0, value=500.0)
        baseline_revenue_loss = np.sum(y_test) * revenue_per_customer
        adjusted_revenue_loss = np.sum(adjusted_churn) * revenue_per_customer
        st.write(f"Current Revenue Loss: ${baseline_revenue_loss:,.2f}")
        st.write(f"Projected Revenue Loss after retention campaign: ${adjusted_revenue_loss:,.2f}")
        
        # Export Option
        if st.button("Export Adjusted Predictions"):
            output = X_test.copy()
            output['ActualChurn'] = y_test
            output['PredictedChurn'] = y_pred
            output['AdjustedChurn'] = adjusted_churn
            output.to_csv("adjusted_churn_predictions.csv", index=False)
            st.success("Exported adjusted_churn_predictions.csv")
