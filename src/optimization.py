import numpy as np


def optimize_usage(forecast, peak_price=2.0, off_peak_price=1.0):
    """
    Optimize energy usage based on pricing

    Peak hours cost more
    Off-peak hours cheaper

    We simulate shifting some demand away from peak hours
    """

    forecast = np.array(forecast)

    prices = np.array([
        peak_price if i % 24 in range(8, 20) else off_peak_price
        for i in range(len(forecast))
    ])

    # Simple strategy: reduce peak demand by 10%
    optimized = forecast.copy()

    for i in range(len(optimized)):
        if i % 24 in range(8, 20):  # peak hour
            optimized[i] *= 0.9

    cost_before = np.sum(forecast * prices)
    cost_after = np.sum(optimized * prices)

    savings = cost_before - cost_after

    return optimized, cost_before, cost_after, savings
