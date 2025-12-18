\
import numpy as np

def mean_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    mu = np.mean(history, axis=0)
    return np.tile(mu.reshape(1, -1), (horizon, 1)).astype(np.float32)
