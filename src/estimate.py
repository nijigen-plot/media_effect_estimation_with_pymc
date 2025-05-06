# %%
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import Dict, List

import pymc as pm
import arviz as az
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from util.metric_scaler import MetricScaler
from util.mcmc_media_effect_estimate import MCMCMediaEffectEstimate
from util.models import EstimateModelInput, EstimateModelOutput

%load_ext autoreload
%autoreload 2
MS = MetricScaler()
MCME = MCMCMediaEffectEstimate()
def estimate_target_parameters(d: Dict) -> List:
    params = ["alpha", "lam", "x_coefficient", "intercept", "k", "sigma", "nu"]
    return [x for x in params if x not in d]
# 必要なデータをロード
y_obs = np.load("../data/y_obs.npy")
other_media_obs = np.load("../data/other_media_obs.npy")
other_media_effect = np.load("../data/other_media_effect.npy")
unobservable_media_obs_impact = np.load("../data/unobservable_media_obs_impact.npy")
tl = np.load("../data/tl.npy")

y = y_obs + other_media_effect
# y = y_obs + other_media_effect + unobservable_media_obs_impact # 観測不可な要因による主要指標への影響を加えたい場合はこちらを使う
y_scaled, y_scaler = MS.max_abs_scaler(y)
x_scaled, x_scaler = MS.max_abs_scaler(other_media_obs)
fixed_parameters = {"lam": 2} # alpha, lam, x_coefficientについて事前に値を決めたい場合ここに入れる
estimated_target_parameters = estimate_target_parameters(fixed_parameters)
estimate_model_input = EstimateModelInput(
    fixed_parameters=fixed_parameters,
    y=y_scaled,
    x=x_scaled,
    t=tl,
)
estimate_model_output = MCME.estimate(estimate_model_input)
# 事前分布によるとりうる値の範囲を確認
palette = "viridis_r"
cmap = plt.get_cmap(palette)
percs = np.linspace(51, 99, 100)
colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))

fig, ax = plt.subplots()

for i, p in enumerate(percs[::-1]):
    upper = np.percentile(
        estimate_model_output.prior_predictive.prior_predictive["likelihood"],
        p,
        axis=1
    )
    lower = np.percentile(
        estimate_model_output.prior_predictive.prior_predictive["likelihood"],
        100 - p,
        axis=1
    )
    color_val = colors[i]
    ax.fill_between(
        x=tl,
        y1=upper.flatten(),
        y2=lower.flatten(),
        color=cmap(color_val),
        alpha=0.1
    )
sns.lineplot(x=tl, y=y_scaled, color="black", label="target(scaled)", ax=ax)
ax.legend()
ax.set(
    title="Base Model - Prior Predictive Samples"
)
# MCMCモデルの可視化
pm.model_to_graphviz(estimate_model_output.model)
# 各パラメータ推定後の分布を確認
print(az.summary(
    data=estimate_model_output.trace,
    var_names=estimated_target_parameters,
))
axes = az.plot_trace(
    data=estimate_model_output.trace,
    var_names=estimated_target_parameters,
    compact=True,
    backend_kwargs={
        "figsize": (12, 9),
        "layout": "constrained"
    }
)
fig = axes[0][0].get_figure()
fig.suptitle('Base Model - Trace')
fig, ax = plt.subplots(figsize=(8, 6))
az.plot_forest(
    data=estimate_model_output.trace,
    var_names=estimated_target_parameters,
    combined=True,
    ax=ax
)
ax.set(
    title="Base Model: 94.0% HDI",
    xscale="log"
)
# 主要指標予測結果の確認
posterior_predictive_likelihood = az.extract(
    data=estimate_model_output.posterior_predictive,
    group='posterior_predictive',
    var_names=["likelihood"],
)
posterior_predictive_likelihood_inv = y_scaler.inverse_transform(
    X=posterior_predictive_likelihood
)
fig, ax = plt.subplots()
for i, p in enumerate(percs[::-1]):
    upper = np.percentile(posterior_predictive_likelihood_inv, p, axis=1)
    lower = np.percentile(posterior_predictive_likelihood_inv, 100 - p, axis=1)
    color_val = colors[i]
    ax.fill_between(
        x=tl,
        y1=upper,
        y2=lower,
        color=cmap(color_val),
        alpha=0.1
    )

sns.lineplot(
    x=tl,
    y=np.median(posterior_predictive_likelihood_inv, axis=1),
    color="C2",
    label="posterior predictive median",
    ax=ax
)
sns.lineplot(
    x=tl,
    y=y,
    color="black",
    label="target",
    ax=ax
)
ax.legend(loc="upper left")
ax.set(title="Base Model - Posterior Predictive Samples")
# 効果量分布の確認
posterior_predictive_x_effect = az.extract(
    data=estimate_model_output.trace,
    group='posterior',
    var_names=["x_effect"]
)
posterior_predictive_x_effect_inv = y_scaler.inverse_transform(
    X=posterior_predictive_x_effect
)
fig, ax = plt.subplots()
for i, p in enumerate(percs[::-1]):
    upper = np.percentile(posterior_predictive_x_effect_inv, p, axis=1)
    lower = np.percentile(posterior_predictive_x_effect_inv, 100 - p, axis=1)
    color_val = colors[i]
    ax.fill_between(
        x=tl,
        y1=upper,
        y2=lower,
        color=cmap(color_val),
        alpha=0.1
    )

sns.lineplot(
    x=tl,
    y=np.median(posterior_predictive_x_effect_inv, axis=1),
    color="C2",
    label="posterior x_effect median",
    ax=ax
)
sns.lineplot(
    x=tl,
    y=other_media_effect,
    color="black",
    label="target",
    ax=ax
)
ax.legend(loc="upper left")
ax.set(title="Base Model - Posterior x Effect Samples")
