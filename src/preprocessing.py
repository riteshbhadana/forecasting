import pandas as pd


def load_orders(path="data/olist_orders_dataset.csv"):
    df = pd.read_csv(path)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    return df


def create_time_series(df, freq="D"):
    df = df.set_index("order_purchase_timestamp")
    ts = df.resample(freq).size()
    ts = ts.rename("demand")

    return ts.asfreq(freq, fill_value=0)


if __name__ == "__main__":
    df = load_orders()
    ts = create_time_series(df)

    print(ts.head())
    print("\nTime-series ready for forecasting!")
