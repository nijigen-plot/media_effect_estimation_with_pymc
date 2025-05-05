# %%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pymc as pm
import arviz as az
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from util.metric_scaler import MetricScaler
from util.mcmc_media_effect_estimate import MCMCMediaEffectEstimate
from util.models import EstimateModelInput, EstimateModelOutput

%load_ext autoreload
%autoreload 2
MS = MetricScaler()
MCME = MCMCMediaEffectEstimate()
# %%
# 必要なデータをロード
y_obs = np.load("../data/y_obs.npy")
other_media_obs = np.load("../data/other_media_obs.npy")
other_media_effect = np.load("../data/other_media_effect.npy")
tl = np.load("../data/tl.npy")

# %%
y = y_obs + other_media_effect
y_scaled, y_scaler = MS.max_abs_scaler(y)
x_scaled, x_scaler = MS.max_abs_scaler(other_media_obs)
estimate_model_input = EstimateModelInput(
    fixed_parameters={},
    y=y_scaled,
    x=x_scaled,
    t=tl,
)
# %%
estimate_model_output = MCME.estimate(estimate_model_input)

# %%
pm.model_to_graphviz(estimate_model_output.model)
# %%
az.summary(
    data=estimate_model_output.trace,
    var_names=["alpha", "lam", "x_coefficient", "intercept", "k", "sigma", "nu"],
)
# %%
axes = az.plot_trace(
    data=estimate_model_output.trace,
    var_names=["alpha", "lam", "x_coefficient", "intercept", "k", "sigma", "nu"],
    compact=True,
    backend_kwargs={
        "figsize": (12, 9),
        "layout": "constrained"
    }
)
fig = axes[0][0].get_figure()
fig.suptitle('Base Model - Trace')
# %%
fig, ax = plt.subplots(figsize=(8, 6))
az.plot_forest(
    data=estimate_model_output.trace,
    var_names=["alpha", "lam", "x_coefficient", "intercept", "k", "sigma", "nu"],
    combined=True,
    ax=ax
)
ax.set(
    title="Base Model: 94.0% HDI",
    xscale="log"
)
# %%
type(estimate_model_output.prior_predictive)
type(estimate_model_output.posterior_predictive)
# %%
type(estimate_model_output.model)
