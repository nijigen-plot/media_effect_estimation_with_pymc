from typing import Tuple

import numpy as np
from sklearn.preprocessing import MaxAbsScaler


class MetricScaler:
    def max_abs_scaler(self, data: np.ndarray) -> Tuple[np.ndarray, MaxAbsScaler]:
        scaler = MaxAbsScaler()
        scaler.fit(data.reshape(-1, 1))
        scaled_data = scaler.transform(data.reshape(-1, 1)).flatten()
        return scaled_data, scaler
