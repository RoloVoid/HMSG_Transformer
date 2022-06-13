''' 
personal implementation of training procedure for Hierarchical Multi-Scale Gaussian Transformer
dzy 2022.5.31
'''

from tkinter import N
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,random_split
import math
import pandas as pd
import random
import tqdm
import model




# label = "./dataset/example.label.csv"
# dataset = "./dataset/example.dataset.csv"

dataset = "./dataset/000300.XSHG.daily.csv"
label = "./dataset/000300.XSHG.daily.label.csv"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 5
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
EPOCHS = 30
N_HEADS = 4
GAMA = 0.5 # for Orthogonal Regularization

# Dataset class
class StockDataSet(Dataset):
    def __init__(self, data:pd.DataFrame, label:pd.DataFrame):
        self.__load(data,label)
    
    def __getitem__(self,index):
        stockcode = self.tags[index]
        return torch.Tensor(self.data.get_group(stockcode).drop('code',axis=1).values),self.label[stockcode]

    def __len__(self):
        return len(self.tags)

    def __load(self,data:pd.DataFrame,label:pd.DataFrame):
        self.tags = label.columns.values.tolist()[1:] # get rid of 'time' in [0]
        self.label = label
        self.data = data.drop(['time'],axis=1).groupby('code') # which limits the format of the raw_data

# Iterator
def cycle(loader):
    while True:
        for data in loader:
            yield data

if __name__ == "__main__":
    # read data
    data = pd.read_csv(dataset,index_col=0)
    label = pd.read_csv(label,index_col=0)

    # generate train/label/test dataset
    dataset_all = StockDataSet(data,label)
    
    # Because of cleanning, dataset size is changing
    size = len(dataset_all)
    train = math.ceil(size/1.5) # use about 2/3 of the dataset as training part
    val = round((size-train)/2)
    test = size-train-val

    # generate dataloader
    dataset_train,dataset_val,dataset_test = random_split(dataset_all,(train,val,test))
    dataloader_train = DataLoader(dataset_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=False,drop_last=True)
    dataloader_val = DataLoader(dataset_val,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,drop_last=True)
    dataloader_test = DataLoader(dataset_test,batch_size=BATCH_SIZE,)
    

    # a model instance
    mlmodel = model.HMSGTransformer(
        H=512,
        d_q=40,
        d_k=40,
        d_v=40,
        n_heads=4,
        n_layers=3,
        d_ff=16,
        batch_size=BATCH_SIZE,
        d_model=5,
        seq_len=5,
    )

    # Orthogonal Regularization for Multi-Head Self-Attention Mechanism
    #  weight: [d_v * n_heads, d_model] -> [n_heads, d_v*d_model]
    m = []
    w_vhs = mlmodel.encoder.w_vhs
    fnorm = nn.MSELoss(reduction='mean')
    l_p = 0
    for w_vh in range (w_vhs):
        A = w_vh.reshape(N_HEADS,-1)
        A = A/torch.norm(A)
        l_p += fnorm(A,torch.ones(A.size())).item()

    # Basic loss fomula: loss = l_c_e + \gamma * l_p
    # Gamma is a hyper_parameter            
    # According to the paper, the model uses BCELoss
    loss_func = nn.BCELoss()
    optim = optim.Adam(mlmodel.parameters,lr=LEARNING_RATE)