import matplotlib.pyplot as plt
import numpy as np
import h5py
with h5py.File('y_pred.h5', 'r') as hf:
    # 获取名为'data'的数据集
    y_pred = hf['data'][:]
with h5py.File('testY.h5', 'r') as hf:
    # 获取名为'data'的数据集
    testY = hf['data'][:]


pred = y_pred[:288,:1,12]
true = testY[:288,:1,12]


# 绘制对比图
# plt.figure(figsize=(10, 6))
# plt.plot(pred, label='Predicted')
# plt.plot(true , label='True')
# plt.legend(fontsize=38)
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.show()