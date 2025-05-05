# %%
import numpy as np
import pytensor.tensor as pt
from dotenv import load_dotenv

load_dotenv()


class MediaEffectModel:
    def geometric_adstock(self, x, alpha: float = 0.0, l_max: int = 12):
        cycles = [
            pt.concatenate(
                [pt.zeros(i), x[: x.shape[0] - i]]
            )
            for i in range(l_max)
        ]
        x_cycle = pt.stack(cycles)
        w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
        return pt.dot(w, x_cycle)

    def logistic_saturation(self, x: np.ndarray, lam:float):
        return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))

    def media_coefficient(self, x: np.ndarray, beta:float):
        return x * beta
