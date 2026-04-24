import configparser
import numpy as np
import os
from datetime import datetime
import torch
import pandas as pd
import torch.nn as nn
import argparse
# from generator import Generator
from stladp import Generator
from discriminator import Discriminator
from trainer import Trainer
# from train1 import Trainer
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau


#*************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'PEMS08'      # PEMS03 or PEMS04 or PEMS07 or PEMS08 or METR-LA or PEMS-Bay
ADJ_MATRIX = './data/{}/{}.csv'.format(DATASET, DATASET)
#*************************************************************************#

# get configuration
config_file = './config/{}.conf'.format(DATASET)
print('Reading configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='L-GGAN',
                    help='model_type')
parser.add_argument('--adj_file', default=ADJ_MATRIX, type=str)
parser.add_argument('--mode', default=Mode, type=str)
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=12,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--window', type=int, default=5,
                    help='temporal window size for attentions')
parser.add_argument('--train_ratio', type=float, default=0.6,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.2,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')

parser.add_argument('--decay_epoch', type=int, default=60,
                    help='decay epoch')
parser.add_argument('--ds', default='METR-LA',
                    help='dataset name')
parser.add_argument('--se_type', default='lap',
                    help='spatial embedding file')

parser.add_argument('--remark', default='',
                    help='remark')

parser.add_argument('--temporal_steps', type=int, default=288,
                    help='temporal_steps')
parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
parser.add_argument('--num_days', default=config['data']['num_days'], type=int)
parser.add_argument('--start_date', default=config['data']['start_date'], type=str)
parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
parser.add_argument('--real_value', default=True, type=eval, help='use real value for loss calculation')
parser.add_argument('--lr_decay', default=True, type=eval)
parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
parser.add_argument('--debug', default=DEBUG, type=eval)
parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
# test
parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
# log
parser.add_argument('--log_dir', default='./', type=str)
parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
parser.add_argument('--plot', default=config['log']['plot'], type=eval)

parser.add_argument('--weight_decay', type=float, default=0,
                    help='dropout')
parser.add_argument('--lambda_gp', type=float, default=0.01)
parser.add_argument('--max_epoch', type=int, default=100,
                    help='epoch to run')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--lr_decay_rate', default='0.7',type=float)
parser.add_argument('--loss_G_D', type=float, default=0.01,
                    help='ration of loss_G*loss_D')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--data_type', default='flow',
                    help='model_type')





args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_extension = '.h5' if os.path.exists(f'data/{args.ds}/{args.ds}.h5') else '.npz'
args.traffic_file = f'data/{args.ds}/{args.ds}{file_extension}'
args.adj= f'data/{args.ds}/{args.ds}_correlation.pkl'
SE, g = load_graph(args)


cuda = True if torch.cuda.is_available() else False
TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
with open(args.adj, 'rb') as f:
    norm_dis_matrix = pickle.load(f)

norm_dis_matrix=TensorFloat(norm_dis_matrix)

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    # print_model_parameters(model, only_num=False)

    return model


if __name__ == "__main__":


    generator = Generator(args,SE, window_size=args.window, T=args.num_his, N=args.num_nodes)
    generator = generator.to(args.device)
    generator = init_model(generator)

    discriminator = Discriminator(args)
    discriminator = discriminator.to(args.device)
    discriminator = init_model(discriminator)


    # optimizer
    optimizer_G = torch.optim.Adam(params=generator.parameters(),
                                   lr=args.learning_rate,
                                   eps=1.0e-8,
                                   weight_decay=args.weight_decay,
                                   amsgrad=False)

    optimizer_D = torch.optim.Adam(params=discriminator.parameters(),
                                   lr=args.learning_rate * 0.1,
                                   eps=1.0e-8,
                                   weight_decay=args.weight_decay,
                                   amsgrad=False)


    loss_D = torch.nn.BCELoss()
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_G,
                                                          milestones=lr_decay_steps,
                                                          gamma=args.lr_decay_rate)

    lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D,
                                                          milestones=lr_decay_steps,
                                                          gamma=args.lr_decay_rate)



    # config log path
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir, 'log', args.ds, current_time)
    args.log_dir = log_dir


    # model training or testing
    trainer = Trainer(args,
                      generator, discriminator,loss_D,
                      optimizer_G, optimizer_D,
                      lr_scheduler_G, lr_scheduler_D,norm_dis_matrix)

    if args.mode.lower() == 'train':
        trainer.train()
    elif args.mode.lower() == 'test':

        model_state_dict = torch.load('/root/autodl-tmp/pem04/log/PEMS04/best/best_mod1el.pth')['state_dict']
        generator.load_state_dict(model_state_dict)
        print("Load saved model")
        trainer.test(norm_dis_matrix,generator, args, trainer.logger)
    else:
        raise ValueError("Invalid mode")