''' 
personal implementation of training procedure for Hierarchical Multi-Scale Gaussian Transformer
dzy 2022.5.31
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import model
import yaml
from load import StockData,StockDataSet

# choose which dataset to train on
datasetdir = "./dataset/csi500labeled/16"
datefile = "./dataset/csi500date.csv"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
BATCH_SIZE = 5
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
EPOCHS = 30
N_HEADS = 4
WINDOW_SIZE = 16
GAMA = 0.5 # A hyperparameter for Orthogonal Regularization

if __name__ == "__main__":

    # split data
    target = open("./config.yaml",encoding='utf-8')
    temp = target.read()
    config = yaml.load(temp,Loader=yaml.FullLoader)

    csi500 = StockData(datasetdir,shuffle=False)

    # Generate Data
    DatasetTest = StockDataSet(
        startdate=config['TestStart'],
        enddate=config['TestEnd'],
        datefile=datefile,
        window_size=WINDOW_SIZE,
        data=csi500[2],
        type="train"
    )

    DatasetVal = StockDataSet(
        startdate=config['ValStart'],
        enddate=config['ValEnd'],
        datefile=datefile,
        window_size=WINDOW_SIZE,
        data=csi500[2],
        type="train"
    )


    DatasetTrain = StockDataSet(
        startdate=config['TrainStart'],
        enddate=config['TrainEnd'],
        datefile=datefile,
        window_size=WINDOW_SIZE,
        data=csi500[2],
        type="train"
        )

    # generate dataloader
    dataloader_train = DataLoader(DatasetTrain,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=False,drop_last=True)
    dataloader_val = DataLoader(DatasetVal,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
    dataloader_test = DataLoader(DatasetTest,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
    

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
        seq_len=16,
        device=device
    )

    # Basic loss fomula: loss = l_c_e + \gamma * l_p
    # Gamma is a hyper_parameter            
    # According to the paper, the model uses BCELoss
    loss_func = nn.BCEWithLogitsLoss()
    optim = optim.Adam(mlmodel.parameters(),lr=LEARNING_RATE)


    for i in range (EPOCHS):
        for step,[data,label] in enumerate(tqdm(dataloader_train)):

            # Orthogonal Regularization for Multi-Head Self-Attention Mechanism
            #  weight: [d_v * n_heads, f_size] -> [n_heads, d_v*f_size]
            m = []
            w_vhs = mlmodel.encoder.w_vhs
            fnorm = nn.MSELoss(reduction='mean')
            l_p = 0
            for w_vh in w_vhs:
                A = w_vh.reshape(N_HEADS,-1)
                A = A/torch.norm(A)
                A = torch.matmul(A,A.T)
                l_p += fnorm(A,torch.ones(A.size())).item()
            if 0 in label: continue

            outputs = mlmodel(data)
            loss = loss_func(outputs,label)+GAMA*l_p
            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % 50 == 0:
                print(outputs,label) 
                print("train time: {}, Loss: {}".format(step, loss.item()))

