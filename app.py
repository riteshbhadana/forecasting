import streamlit as st
import numpy as np
import pandas as pd
import os

from dotenv import load_dotenv
from groq import Groq

from src.preprocessing import create_time_series
from src.arima import load_model, forecast
from src.optimization import optimize_usage

# ------------------------
# Load environment
# ------------------------
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------------
# LLM helper
# ------------------------
def ask_llm(summary, question):
    prompt = f"""
You are a data analytics assistant.

Dataset summary:
{summary}

User question:
{question}

Explain clearly in simple language.
"""

    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )

    return chat.choices[0].message.content


# ------------------------
# Streamlit UI
# ------------------------
st.title("📦 Demand Forecasting & Optimization Dashboard")

uploaded_file = st.file_uploader("Upload orders CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "order_purchase_timestamp" not in df.columns:
        st.error("CSV must contain 'order_purchase_timestamp' column")
        st.stop()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    ts = create_time_series(df)

    # ------------------------
    # Historical chart
    # ------------------------
    st.subheader("Historical Demand")
    st.line_chart(ts.tail(60))

    # ------------------------
    # Forecast
    # ------------------------
    model = load_model()
    pred = forecast(model, 24).round()

    optimized, before, after, savings = optimize_usage(pred)

    forecast_df = pd.DataFrame({
        "Forecast": pred,
        "Optimized": optimized
    })

    st.subheader("24-Step Forecast vs Optimized Usage")
    st.line_chart(forecast_df)

    # ------------------------
    # Metrics
    # ------------------------
    st.subheader("Cost Comparison")

    col1, col2, col3 = st.columns(3)
    col1.metric("Cost Before", f"{before:.2f}")
    col2.metric("Cost After", f"{after:.2f}")
    col3.metric("Savings", f"{savings:.2f}")

    st.success("Optimization complete ✅")

    # ------------------------
    # LLM summary
    # ------------------------
    summary = f"""
Average demand: {ts.mean()}
Max demand: {ts.max()}
Forecast next 24: {pred.tolist()}
Savings after optimization: {savings}
"""

    st.subheader("🤖 Ask AI about your data")

    question = st.text_input("Ask a question about the dataset")

    if question:
        with st.spinner("AI thinking..."):
            answer = ask_llm(summary, question)
        st.write(answer)

else:
    st.info("Upload a CSV file to begin forecasting")
