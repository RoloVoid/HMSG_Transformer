'''
This script is used to split dataset into three part
'''

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import random 

# Data init 
class StockData():
    def __init__(
        self,
        datadir,
        shuffle=True
        ):
        # assert os.path.exists(stockfile), "stockfile maybe not initialize correctly"
        # self.stocklist = pd.read_csv(stockfile).T.values.to_list()

        assert os.path.exists(datadir), "target dataset does not exist"
        datalist = os.listdir(datadir)

        if shuffle: self.stocklist = random.shuffle(datalist)
        else: self.datalist = datalist

        self.datadir = datadir
        self.data = []
        self._read()

    def _read(self):
        for m in self.datalist:
            print(f"reading {m} ......")
            self.data.append(torch.Tensor(pd.read_csv(self.datadir+"/"+m).values))
        print("reading dataset done")
    
    def __len__(self):
        return len(self.data)
    
# Dataset class
class StockDataSet(Dataset):
    def __init__(
        self,
        trainstart,
        trainend,
        teststart,
        testend,
        valstart,
        valend,
        datefile,
        window_size,
        data,
        type="train"
        ):
        super(StockDataSet,self).__init__()

        # date for different dataset
        self.trainstart = trainstart
        self.trainend = trainend
        self.teststart = teststart
        self.testend = testend
        self.valstart = valstart
        self.valend = valend
        self.data = data

        # all the date
        # dateinit is used to mark whether the length of dataset is inited
        self.datefile = datefile
        self._initcheck = False
        self.start = 0
        self.end = 0

        assert type in ["train","test","val"], "Not a suitable dataset type"
        self.window = window_size
        self.type = type

        
        self._dateinit()
    
    def __getitem__(self,index):
        start = self.start+index
        end = start+self.window_size
        return self.data[start,end][:-1],self.data[end-1][-1]
    
    def _dateinit(self):
        assert os.path.exists(self.datefile), "Date file may not initialize successfully"
        date = pd.read_csv(self.datefile)     
        # 0 and 15 is specially for dataset which fre = 15min
        if self.type == "train":
            self.start = date[date['date']==self.trainstart].index[0]
            self.end = date[date['date']==self.trainend].index[0]
        elif type == "test":
            self.start = date[date['date']==self.teststart].index[0]
            self.end = date[date['date']==self.testend].index[0]
        else:
            self.start = date[date['date']==self.valstart].index[0]
            self.end = date[date['date']==self.valend].index[0]
        self._initcheck = True

    def __len__(self):
        assert self._dateinit, "date file may not initialize successfully"
        return self.end-self.start+1
        
