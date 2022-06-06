''' 
personal implementation of training procedure for Hierarchical Multi-Scale Gaussian Transformer
dzy 2022.5.31
'''
import pandas as pd
import torch
from torch.utils.data import Dataset
import model
import tqdm


# label = "./dataset/example.label.csv"
# dataset = "./dataset/example.dataset.csv"

dataset = "./dataset/000300.XSHG.daily.csv"
label = "./dataset/000300.XSHG.daily.label.csv"


# Dataset class
class StockDataSet(Dataset):
    def __init__(self, data:pd.DataFrame, label:pd.DataFrame, flag='train'):
        assert flag in ['train','test','valid']
        self.flag = flag
        self.__load(data,label)
    
    def __getitem__(self,index):
        stockcode = self.tags[index]
        return torch.Tensor(self.data.get_group(stockcode).values),self.label[self.date][stockcode]

    def __len__(self):
        return len(self.tags)

    def __load(self,data:pd.DataFrame,label:pd.DataFrame):
        self.tags = label.index.values.tolist()
        self.date = label.columns.values.tolist()[0]
        self.label = label
        self.data = data.groupby('code')

# Iterator
def cycle(loader):
    while True:
        for data in loader:
            yield data

if __name__ == "__main__":
    data = pd.read_csv(dataset)
    label = pd.read_csv(label)

    dataloader = StockDataSet(data,label)

    mlmodel = model.HMSGTransformer(
        H=512,
        d_q=40,
        d_k=40,
        d_v=40,
        n_heads=4,
        n_layers=3,
        d_ff=16,
        batch_size=4,
        d_model=5,
        seq_len=40,
    )