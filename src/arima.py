import pickle
from statsmodels.tsa.arima.model import ARIMA


def train_arima(series):
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit


def save_model(model, path="models/arima.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path="models/arima.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def forecast(model, steps=7):
    return model.forecast(steps)
