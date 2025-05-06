import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pymc as pm
from tqdm import tqdm

from util.media_effect_model import MediaEffectModel
from util.metric_scaler import MetricScaler
from util.models import EstimateModelInput, EstimateModelOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MEM = MediaEffectModel()
MS = MetricScaler()

class MCMCMediaEffectEstimate:
    def estimate(self, input: EstimateModelInput):
        fixed_parameters = input.fixed_parameters
        y = input.y
        x = input.x
        t = input.t
        t_scaled = (t - np.min(t)) / (np.max(t) - np.min(t))
        coords = {"date": t}
        with pm.Model(coords=coords) as model:
            x = pm.MutableData(name="x", value=x, dims="date")
            if 'alpha' in fixed_parameters:
                logger.info("alpha key detected. alpha is not estimated.")
                alpha = fixed_parameters['alpha']
            else:
                alpha = pm.Beta(name="alpha", alpha=1, beta=1)
            if 'lam' in fixed_parameters:
                logger.info("lam key detected. lam is not estimated.")
                lam = fixed_parameters['lam']
            else:
                lam = pm.Gamma(name="lam", alpha=2, beta=1)
            if 'x_coefficient' in fixed_parameters:
                logger.info("x_coefficient key detected. x_coefficient is not estimated.")
                x_coefficient = fixed_parameters['x_coefficient']
            else:
                x_coefficient = pm.HalfNormal(name="x_coefficient", sigma=0.1)
            x_adstock = pm.Deterministic(
                name="x_adstock",
                var=MEM.geometric_adstock(
                    x=x,
                    alpha=alpha
                ), dims="date"
            )
            x_saturation = pm.Deterministic(
                name="x_saturation",
                var=MEM.logistic_saturation(
                    x=x_adstock,
                    lam=lam
                ), dims="date"
            )
            x_effect = pm.Deterministic(
                name="x_effect",
                var=MEM.media_coefficient(
                    x=x_saturation,
                    beta=x_coefficient,
                ), dims="date"
            )

            intercept = pm.HalfNormal(name="intercept", sigma=0.03)
            k = pm.HalfNormal(name="k", sigma=5)
            intercept_and_trend = pm.Deterministic(name="intercept_and_trend", var=intercept + pm.math.exp(-k * (t_scaled - 0.08)), dims="date")
            sigma = pm.HalfNormal(name="sigma", sigma=0.1)
            nu = pm.Gamma(name="nu", alpha=5, beta=2)
            mu = pm.Deterministic(
                name="mu",
                var=intercept_and_trend + x_effect, dims="date"
            )

            # 尤度関数
            pm.StudentT(
                name="likelihood",
                nu=nu,
                mu=mu,
                sigma=sigma,
                observed=y,
                dims="date"
            )

            prior_predictive = pm.sample_prior_predictive()
        with model:
            trace = pm.sample(
                tune=500,
                draws=1000,
                chains=4
            )
            model_posterior_predictive = pm.sample_posterior_predictive(
                trace=trace
            )
        return EstimateModelOutput(
            prior_predictive=prior_predictive,
            posterior_predictive=model_posterior_predictive,
            model=model,
            trace=trace
        )
