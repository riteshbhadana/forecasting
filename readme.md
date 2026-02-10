# 📦 Demand Forecasting & Optimization Dashboard

An end-to-end machine learning system that forecasts logistics demand, optimizes usage cost, and provides an AI-powered analytics assistant through an interactive dashboard.

This project demonstrates time-series forecasting, optimization, and LLM integration in a real-world pipeline.

---

## 🚀 Features

* 📊 Time-series demand forecasting (ARIMA + LSTM)
* ⚡ Cost-aware optimization engine
* 📈 Interactive Streamlit dashboard
* 📁 Upload custom datasets
* 🤖 AI assistant to explain trends (Groq LLM)
* 🔒 Secure API key management with `.env`
* 🧠 Real logistics dataset pipeline

---

## 🧠 System Architecture

Dataset → Preprocessing → Forecasting → Optimization → Dashboard → AI Assistant

The system predicts future demand and simulates cost savings under peak pricing conditions.

---

## 📂 Project Structure

energy-forecasting/
│
├── data/
├── src/
│   preprocessing.py
│   arima.py
│   lstm.py
│   optimization.py
│
├── models/
├── app.py
├── main.py
├── requirements.txt
└── README.md

---

## 📦 Dataset

This project uses the **Olist Brazilian E-commerce dataset**, a public logistics dataset widely used in research.

Dataset source:
[https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## ⚙ Installation

Clone repository:

git clone [https://github.com/riteshbhadana/forcasting.git](https://github.com/your-username/energy-forecasting.git)
cd energy-forecasting

Create virtual environment:

python -m venv venv
venv\Scripts\activate  (Windows)

Install dependencies:

pip install -r requirements.txt

---

## 🔐 Environment Setup

Create `.env` file in project root:

GROQ_API_KEY=your_api_key_here

Do not commit `.env` to GitHub.

---

## ▶ Run Training

python main.py

This trains ARIMA + LSTM and saves models.

---

## ▶ Run Dashboard

streamlit run app.py

Upload CSV → forecast → optimize → ask AI questions.

---

## 💡 Example Questions for AI Assistant

* “Is demand increasing?”
* “Explain the forecast trend”
* “What does optimization change?”
* “Summarize recent demand”

---

## 🧪 Technologies Used

* Python
* Pandas / NumPy
* PyTorch (LSTM)
* Statsmodels (ARIMA)
* Streamlit
* Groq LLM API
* SciPy Optimization

---

## 📈 Skills Demonstrated

* Time-series forecasting
* Deep learning modeling
* Optimization algorithms
* ML system design
* API integration
* Interactive dashboards
* LLM analytics
* Secure environment handling

---

## 💼 Resume Summary

Built an end-to-end demand forecasting and optimization system using ARIMA and LSTM models. Developed an interactive dashboard with AI-assisted analytics and cost optimization simulation using real logistics data.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Ritesh Bhadana
BTech CSE (AI)
Machine Learning & AI Engineer
