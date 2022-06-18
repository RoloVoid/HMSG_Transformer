''' 
personal implementation of training procedure for Hierarchical Multi-Scale Gaussian Transformer
dzy 2022.5.31
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,random_split
import math
import pandas as pd
import random
import tqdm
import model
import yaml
from load import StockData,StockDataSet

# choose which dataset to train on
datasetdir = "./dataset/csi500labeled/16"
datefile = "./dataset/csi500date.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 5
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
EPOCHS = 30
N_HEADS = 4
WINDOW_SIZE = 16
GAMA = 0.5 # A hyperparameter for Orthogonal Regularization

if __name__ == "__main__":

    # # split data
    # target = open("./config.yaml",encoding='utf-8')
    # temp = target.read()
    # config = yaml.load(temp,Loader=yaml.FullLoader)

    # print(config['TrainStart'])

    # csi500 = StockData(datasetdir,shuffle=False)
    # csi500DataSet = StockDataSet(
    #     trainstart=config['TrainStart'],
    #     trainend=config['TrainEnd'],
    #     teststart=config['TestStart'],
    #     testend=config['TestEnd'],
    #     valstart=config['ValStart'],
    #     valend=config['ValEnd'],
    #     datefile=datefile,
    #     window_size=WINDOW_SIZE,
    #     data=csi500,
    #     type="train"
    #     )
    
    # print(csi500DataSet.start,csi500DataSet.end)
    # # Because of cleanning, dataset size is changing
    # size = len(csi500DataSet)
    # train = math.ceil(size/1.5) # use about 2/3 of the dataset as training part
    # val = round((size-train)/2)
    # test = size-train-val

    # # # generate dataloader
    # # dataset_train,dataset_val,dataset_test = random_split(csi500DataSet,(train,val,test))
    # # dataloader_train = DataLoader(dataset_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=False,drop_last=True)
    # # dataloader_val = DataLoader(dataset_val,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,drop_last=True)
    # # dataloader_test = DataLoader(dataset_test,batch_size=BATCH_SIZE,)
    

    # a model instance
    mlmodel = model.HMSGTransformer(
        H=512,
        E=1024,
        d_q=40,
        d_k=40,
        d_v=40,
        n_heads=4,
        n_layers=3,
        d_ff=16,
        f_size=5,
        seq_len=20,
        device=device
    )

    # Orthogonal Regularization for Multi-Head Self-Attention Mechanism
    #  weight: [d_v * n_heads, f_size] -> [n_heads, d_v*f_size]
    m = []
    w_vhs = mlmodel.encoder.w_vhs
    fnorm = nn.MSELoss(reduction='mean')
    l_p = 0
    for w_vh in w_vhs:
        A = w_vh.reshape(N_HEADS,-1)
        A = A/torch.norm(A)
        l_p += fnorm(A,torch.ones(A.size())).item()

    # Basic loss fomula: loss = l_c_e + \gamma * l_p
    # Gamma is a hyper_parameter            
    # According to the paper, the model uses BCELoss
    loss_func = nn.BCELoss()
    optim = optim.Adam(mlmodel.parameters(),lr=LEARNING_RATE)

