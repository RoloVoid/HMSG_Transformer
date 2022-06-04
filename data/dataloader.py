# build dataset loader
from torch.utils.data import Data

'''
这里的参数是：选取的序列长度(时间窗口长度)，选取的股票数量，选取的特征序列
无论如何，特征序列的最后一行一定是收盘价
'''

# class StockDataSet(Data.Dataset):
#     def __init__(self,stock_input,stock_output):
#         super(StockDataSet,self).__init__()
#         self.stock_input = stock_input
#         self.stock_output = stock_output

class TSDataSet(Data.dataset):
    def __init__(self, flag='train'):
        assert flag in ['train','test','valid']
        self.flag = flag
        self.__load_data__()
    
    def __getitem__(self,index):
        pass

    def __len__(self):
        pass

    def __load_data__(self):
        pass