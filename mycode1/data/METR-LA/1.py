import pickle

# 以二进制模式打开 pickle 文件，并使用 latin1 编码
with open('adj_METR-LA.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 检查返回值的数量
num_return_values = len(data)
print(data)
print(data.shape)
print("pickle 文件中返回值的数量为:", num_return_values)