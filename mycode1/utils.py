import pandas as pd
import torch
import pickle
import dgl
import numpy as np
from scipy import sparse as sp
import hashlib
from metrics import MAE_torch
import logging
import os
from torch.autograd import Variable


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# define gradient penalty function
def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = Variable(alpha * x.data + (1 - alpha) * y.data, requires_grad=True).cuda()
    # gradient penalty
    o = f(z)
    g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].cuda()
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
    return gp

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


# def masked_mae(preds, labels, null_val=0.0):
#     mask = (labels != null_val).float()  # 创建掩码，将null_val之外的值标记为1.0，null_val标记为0.0
#     loss = torch.abs(preds - labels)  # 计算绝对误差
#     loss = loss * mask  # 使用掩码过滤损失
#     return torch.mean(loss)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds-labels)/labels
    
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,rmse,mape





def create_time_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return windows

def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def load_data(args):
    # Traffic
    if args.data_type == 'flow':
        df = np.load(args.traffic_file)
        traffic_flow_data = df['data'][:, :, 0]
        traffic = torch.from_numpy(traffic_flow_data)
    else:
        df = pd.read_hdf(args.traffic_file)
        traffic = torch.from_numpy(df.values)
    # train/val/test
    num_step = traffic.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = traffic[: train_steps]
    val = traffic[train_steps: train_steps + val_steps]
    test = traffic[-test_steps:]
    # X, Y
    trainX, trainY = seq2instance(train, args.num_his, args.num_pred)
    valX, valY = seq2instance(val, args.num_his, args.num_pred)
    testX, testY = seq2instance(test, args.num_his, args.num_pred)
    # normalization
    # testY_flat = testY.reshape(-1, 307)
    # np.savetxt('testY.txt', testY_flat)
    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std


    # temporal embedding
    num_days = args.num_days
    minutes_per_day = 24 * 60
    num_time_points = num_days * minutes_per_day
    start_date = pd.Timestamp(args.start_date)
    time_stamps = pd.date_range(start=start_date, periods=num_time_points, freq='5T')
    # 将时间戳转换为DataFrame的index
    df_index = pd.DatetimeIndex(time_stamps)

    # 获取每个时间点的星期几和时间在一天中的分钟数
    dayofweek = torch.reshape(torch.tensor(df_index.dayofweek.values), (-1, 1))
    timeofday = (df_index.hour * 3600 + df_index.minute * 60 + df_index.second) // 300
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    # 时间embedding
    time = torch.cat((dayofweek, timeofday), -1)

    # train/val/test
    train = time[: train_steps]
    val = time[train_steps: train_steps + val_steps]
    test = time[-test_steps:]

    trainTE = seq2instance(train, args.num_his, args.num_pred)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
             mean, std)

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean



class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


def laplacian_positional_encoding(g, pos_enc_dim=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Laplacian
    A = g.adj_external(scipy_fmt='coo').astype(float)
    # A = g.adj_external(ctx=device)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    return lap_pos_enc

def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 4
    node_color_dict = {}
    node_neighbor_dict = {}
    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()
    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}
    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1
    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
    wl_pos_enc = torch.LongTensor(list(node_color_dict.values())).unsqueeze(1)
    return wl_pos_enc 

def load_graph(args):
    dsadj_file = './data/{}/adj_{}.pkl'.format(args.ds, args.ds)
    if args.data_type == 'speed':
        _, _, adj_mx = load_pickle(dsadj_file)
    else:
        adj_mx = load_pickle(dsadj_file)
    src, dst = np.nonzero(adj_mx)
    g = dgl.graph((src, dst))

    if args.se_type == 'node2vec':
        # spatial embedding
        with open("./data/node2vec_"+args.ds+".txt", mode='r') as f:
            lines = f.readlines()
            temp = lines[0].split(' ')
            num_vertex, dims = int(temp[0]), int(temp[1])
            SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
            for line in lines[1:]:
                temp = line.split(' ')
                index = int(temp[0])
                SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    elif args.se_type == 'wl':
        SE = wl_positional_encoding(g)
    elif args.se_type == 'lap':
        SE = laplacian_positional_encoding(g)
    return  SE, g

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logger(root, name=None, debug=True):
    logger = logging.getLogger(name)
    # critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)

    return logger

def masked_mae_loss(preds,labels, mask_value):
    loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)

    return loss



