# build dataset loader
from torch.utils.data import Data

'''
这里的参数是：选取的序列长度，选取的股票数量，选取的特征序列
无论如何，特征序列的最后一行一定是收盘价
'''

class StockDataSet(Data.Dataset):
    def __init__(self,stock_input,stock_output):
        super(StockDataSet,self).__init__()
        self.stock_input = stock_input
        self.stock_output = stock_output