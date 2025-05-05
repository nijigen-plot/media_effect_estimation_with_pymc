import dataclasses
from typing import Any, Dict, Tuple

import arviz as az
import numpy as np
import pymc as pm


@dataclasses.dataclass(frozen=True)
class EstimateModelInput:
    fixed_parameters: Dict[str, float]
    y: np.ndarray
    x: np.ndarray
    t: np.ndarray


@dataclasses.dataclass(frozen=False)
class EstimateModelOutput:
    prior_predictive: az.data.inference_data.InferenceData
    posterior_predictive: az.data.inference_data.InferenceData
    model: pm.model.core.Model
    trace: az.data.inference_data.InferenceData
