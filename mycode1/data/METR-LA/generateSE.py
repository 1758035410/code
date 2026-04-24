import pandas as pd
from statsmodels.tsa.seasonal import STL
import numpy as np
import torch
import pickle
import h5py
with h5py.File('METR-LA.h5', 'r') as f:
    traffic_flow_data = f['df']['block0_values'][:]  # 从 'block0_values' 读取速度数据

traffic = torch.from_numpy(traffic_flow_data)
mean, std = torch.mean(traffic), torch.std(traffic)

flow = (traffic- mean) / std
trend_list = []
seasonal_list = []
stl_results = []
num_nodes = traffic.shape[1]
trend_correlation = np.zeros((num_nodes, num_nodes))
season_correlation = np.zeros((num_nodes, num_nodes))
# 进行STL分解并保存结果
for i in range(num_nodes):
    sensor_flow = traffic[:, i]
    stl = STL(sensor_flow, period=12*24 * 7, seasonal=25).fit()
    stl_results.append((stl.trend, stl.seasonal))
# 进行STL分解并计算相关矩阵
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:  # 排除节点自身与自身的相关性计算
            trend_i, seasonal_i = stl_results[i]
            trend_j, seasonal_j = stl_results[j]

            # 计算趋势相关矩阵（假设使用Pearson相关系数）
            trend_correlation[i, j] = np.corrcoef(trend_i, trend_j)[0, 1]

            # 计算季节相关矩阵（假设使用Pearson相关系数）
            season_correlation[i, j] = np.corrcoef(seasonal_i, seasonal_j)[0, 1]
# 计算相关邻接矩阵
adjacency_matrix = trend_correlation * season_correlation

with open('PEMS04_trend_correlation.pkl', 'wb') as f:
    pickle.dump(trend_correlation, f)
with open('PEMS04_season_correlation.pkl', 'wb') as f:
    pickle.dump(season_correlation, f)

with open('PEMS04_correlation.pkl', 'wb') as f:
    pickle.dump(adjacency_matrix, f)





