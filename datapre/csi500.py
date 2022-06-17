'''
Script for getting raw csi500 data -- dzy 6.12
'''

import baostock as bs
import pandas as pd
import numpy as np
import os

## system.login() ##
lg = bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# If you get it from https://www.csindex.com.cn/zh-CN/indices/index#/indices/family/detail?indexCode=000905
# Then the filename is this.
filename = "000905cons.xls"
codepath = "../dataset/csi500.csv"

prefix = "../dataset"
if not os.path.exists(prefix): os.mkdir(prefix)

# Get CSI500 components
assert os.path.exists(f"../dataset/{filename}"),f'Please get required csi500 file {filename}'

if not os.path.exists("../dataset/csi500.csv"):
    filename = "../dataset/000905cons.xls"
    data = pd.read_excel(filename,converters={'成分券代码Constituent Code':str})
    # add code prefix
    data['成分券代码Constituent Code'][data['交易所Exchange'] == "上海证券交易所"] =  "sh."+ \
        data['成分券代码Constituent Code'][data['交易所Exchange'] == "上海证券交易所"]

    data['成分券代码Constituent Code'][data['交易所Exchange'] == "深圳证券交易所"] =  "sz."+ \
        data['成分券代码Constituent Code'][data['交易所Exchange'] == "深圳证券交易所"]

    p = data['成分券代码Constituent Code']
    p.to_csv(codepath,index=False)


# Get csi500 raw data
codes = np.array(pd.read_csv(codepath)).T.tolist()[0]
for x in codes:
    filename = f'../dataset/csi500raw/{x}.csv'
    print(x)
    if os.path.exists(filename): continue

    rs = bs.query_history_k_data_plus(x,
        "date,time,open,high,low,close,volume",
        start_date='2018-06-04', end_date='2022-06-10',
        frequency="15")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result.to_csv(filename, index=False)

# logout
bs.logout()