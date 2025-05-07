import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from pytensor import function
from scipy.stats import nbinom

load_dotenv()

from util.media_effect_model import MediaEffectModel
from util.metric_scaler import MetricScaler

if __name__ == "__main__":
    MEM = MediaEffectModel()
    MS = MetricScaler()
    T = 300 # サンプルの長さ
    tl = np.arange(1, T+1)
    t = np.linspace(0, 1, T)
    mu_start = 10000; mu_end = 500

    # 減衰具合を調整
    k = 10 # 傾き
    t0 = 0.15 # 勾配が変化する点
    s = 1 / (1 + np.exp(-k * (t - t0))) # ロジスティック関数で0~1の値を生成
    mu = mu_end + (mu_start - mu_end) * (1 - s) # 逆数をとってmuとする
    var = 10000

    # mu, varから負の二項分布のパラメータを計算してサンプルを生成
    p = mu / var
    n = mu**2 / (var - mu)
    # 主要指標の観測値
    y_obs = nbinom.rvs(n, p, random_state=46)
    scaled_y_obs, y_obs_scaler = MS.max_abs_scaler(y_obs)
    # 施策の観測値
    other_media_obs = np.concatenate([
        np.zeros(200),
        nbinom.rvs(30000, 0.5, size=3, random_state=46),
        nbinom.rvs(15000, 0.5, size=1, random_state=46),
        nbinom.rvs(800, 0.5, size=96, random_state=46)
    ])
    scaled_other_media_obs, media_obs_scaler = MS.max_abs_scaler(other_media_obs)
    # 施策による主要指標観測値への影響係数
    alpha = 0.5 # ジオメトリックアドストック関数のパラメータ
    lam = 2.00 # ロジスティックサチュレーション関数のパラメータ
    beta = 0.1 # 主要指標観測値への影響変数
    apply_geom_adstock = function([], MEM.geometric_adstock(scaled_other_media_obs, alpha))()
    apply_logistic_saturation = function([], MEM.logistic_saturation(apply_geom_adstock, lam))()
    apply_beta_media_coefficient = MEM.media_coefficient(apply_logistic_saturation, beta)
    other_media_effect = media_obs_scaler.inverse_transform(apply_beta_media_coefficient.reshape(-1, 1)).flatten()
    # 未知(観測不可)の要因による主要指標観測値への影響
    unobservable_media_obs_impact = np.concatenate([
        np.zeros(210),
        np.array([550,600,550]),
        np.zeros(10),
        nbinom.rvs(250, 0.5, size=15, random_state=46),
        np.zeros(5),
        np.array([1500,700, 500, 300, 200, 100]),
        np.zeros(26),
        nbinom.rvs(500, 0.5, size=25, random_state=46)
    ])
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=(
            "主要指標観測値",
            "主要指標観測値(影響別積み上げ)",
            "主要指標観測値(他要因除き)",
            "施策による主要指標観測値への効果",
            "施策の観測値(主要指標とは値の単位が異なる)"
        )
    )
    fig.add_trace(
        go.Scatter(x=tl, y=y_obs+other_media_effect, stackgroup='one', name="主要指標観測値", mode='lines', line=dict(color='black')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=y_obs, stackgroup='one', name="主要指標観測値(他要因除き)", fill='tonexty'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=other_media_effect, stackgroup='one', name="施策による主要指標観測値への効果", fill='tonexty'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=y_obs, mode='lines', name="主要指標観測値(他要因除き)", line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=other_media_effect, mode='lines', name="施策による主要指標観測値への効果", line=dict(color='orange')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=other_media_obs, mode='lines', name="施策による主要指標観測値への効果", line=dict(color='red')),
        row=5, col=1
    )
    fig.update_layout(height=1200, width=1200, title_text="主要指標観測値")
    fig.write_image('../data/create_data_graph.png')
    fig.show()
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=(
            "主要指標観測値",
            "主要指標観測値(影響別積み上げ)",
            "主要指標観測値(他要因除き)",
            "施策による主要指標観測値への効果",
            "施策の観測値(主要指標とは値の単位が異なる)",
            "観測不可要因による主要指標観測値への影響"
        )
    )
    fig.add_trace(
        go.Scatter(x=tl, y=y_obs+other_media_effect+unobservable_media_obs_impact, stackgroup='one', name="主要指標観測値", mode='lines', line=dict(color='black')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=y_obs, stackgroup='one', name="主要指標観測値(他要因除き)", fill='tonexty'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=other_media_effect, stackgroup='one', name="施策による主要指標観測値への効果", fill='tonexty'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=unobservable_media_obs_impact, stackgroup='one', name="観測不可要因による主要指標観測値への影響", fill='tonexty'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=y_obs, mode='lines', name="主要指標観測値(他要因除き)", line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=other_media_effect, mode='lines', name="施策による主要指標観測値への効果", line=dict(color='orange')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=other_media_obs, mode='lines', name="施策による主要指標観測値への効果", line=dict(color='red')),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=tl, y=unobservable_media_obs_impact, mode='lines', name="観測不可要因による主要指標観測値への影響", line=dict(color='green')),
        row=6, col=1
    )
    fig.update_layout(height=1400, width=1200, title_text="主要指標観測値(観測不可要因含む)")
    fig.write_image('../data/create_data_graph_include_unobservable_media_obs_impact.png')
    fig.show()
    # データを保存
    np.save('../data/tl.npy', tl)
    np.save('../data/y_obs.npy', y_obs)
    np.save('../data/other_media_obs.npy', other_media_obs)
    np.save('../data/other_media_effect.npy', other_media_effect)
    np.save('../data/unobservable_media_obs_impact.npy', unobservable_media_obs_impact)
