from src.preprocessing import load_orders, create_time_series
from src.arima import train_arima, save_model, forecast
from src.lstm import train_lstm
from src.optimization import optimize_usage
import os

os.makedirs("models", exist_ok=True)

# ------------------------
# Load + preprocess data
# ------------------------
df = load_orders()
ts = create_time_series(df)

# ------------------------
# Train ARIMA
# ------------------------
arima_model = train_arima(ts)
save_model(arima_model)

# ------------------------
# Train LSTM
# ------------------------
train_lstm(ts)

# ------------------------
# Forecast (next 24 steps)
# ------------------------
pred = forecast(arima_model, 24)

# ------------------------
# Optimize usage
# ------------------------
optimized, before, after, savings = optimize_usage(pred)

print("\nForecast:", pred.values)
print("\nOptimized:", optimized)
print("\nCost before:", before)
print("Cost after:", after)
print("Savings:", savings)

print("\nARIMA + LSTM + Optimization complete!")
