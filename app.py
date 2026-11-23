import streamlit as st
st.markdown("<h1 style='text-align: center;'>💳 Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load model and dataset
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv("creditcard.csv")


st.write("Machine Learning based system for detecting fraudulent credit card transactions.")

st.divider()

# Dashboard metrics
total_transactions = len(df)
fraud_transactions = int(df["Class"].sum())
normal_transactions = total_transactions - fraud_transactions

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", f"{total_transactions:,}")

with col2:
    st.metric("Normal Transactions", f"{normal_transactions:,}")

with col3:
    st.metric("Fraud Transactions", f"{fraud_transactions:,}")

st.divider()
st.info("🤖 Model: Random Forest Classifier | Class 0 = Normal | Class 1 = Fraud")

st.subheader("🔍 Transaction Analyzer")

transaction_index = st.number_input(
    "Enter Transaction Number",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

if st.button("Analyze Transaction"):
    
    input_data = df.drop("Class", axis=1).iloc[[transaction_index]]
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    actual_class = df.iloc[transaction_index]["Class"]

    if prediction == 1:
        st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
    else:
        st.success("✅ TRANSACTION APPEARS NORMAL")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        if actual_class == 1:
            st.metric("Actual Class", "Fraud")
        else:
            st.metric("Actual Class", "Normal")

    st.subheader("Transaction Details")
    st.dataframe(input_data, use_container_width=True)

st.divider()

st.subheader("📊 Class Distribution")

class_counts = df["Class"].value_counts()
chart_data = pd.DataFrame({
    "Transaction Type": ["Normal", "Fraud"],
    "Count": [
        int(class_counts.get(0, 0)),
        int(class_counts.get(1, 0))
    ]
})

st.bar_chart(chart_data.set_index("Transaction Type"))

st.caption(
    "Class 0 = Normal Transaction | Class 1 = Fraudulent Transaction"
)