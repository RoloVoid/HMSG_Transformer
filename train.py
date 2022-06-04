''' 
personal implementation of train procedure for Hierarchical Multi-Scale Gaussian Transformer
dzy 2022.5.31
'''
import pandas as pd
import torch
from torch.utils.data import Dataset
from HMSG_transformer import HMSG_Transformer


label = "./dataset/example.label.csv"
dataset = "./dataset/example.dataset.csv"

# Dataset
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


if __name__ == "__main__":
    data = pd.read_csv(dataset)
    label = pd.read_csv(label)

    dataset = StockDataSet(data,label)
    # model = HMSG_Transformer(512)