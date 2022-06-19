'''
This script is used to split dataset into three part
'''

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from tqdm import tqdm 

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
        self._data = []
        self._read()

    def _read(self):
        print("start reading dataset......")
        for m in tqdm(self.datalist):
            self._data.append(torch.Tensor(pd.read_csv(self.datadir+"/"+m).values))
        print("reading dataset done")
    
    def __len__(self):
        return len(self._data)

    def __getitem__(self,index):
        return self._data[index]
    
# Dataset class
class StockDataSet(Dataset):
    def __init__(
        self,
        startdate,
        enddate,
        datefile,
        window_size,
        data,
        type="train"
        ):
        super(StockDataSet,self).__init__()

        # date for different dataset
        self.datestart = startdate
        self.dateend = enddate
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
        end = start+self.window
        return self.data[start:end,0:-1],self.data[end-1,-1]

    def gettype(self):
        return self.type
    
    def _dateinit(self):
        assert os.path.exists(self.datefile), "Date file may not initialize successfully"
        date = pd.read_csv(self.datefile)     
        # 0 and 15 is specially for dataset which fre = 15min

        self.start = date[date['date']==self.datestart].index[0]
        self.end = date[date['date']==self.dateend].index[0]

        self._initcheck = True

    def __len__(self):
        assert self._dateinit, "date file may not initialize successfully"
        return self.end-self.start+1
        
