# This script is for Chinese A shares data

import jqdatasdk as jq
import yaml
import os.path

# We can attempt to use old tushare free api part instead if we have no access to jqdatasdk, but there is only data in past three months 
# ToDo: implement tushare methods
# import tushare as ts
# print(ts.get_hist_data('600847', ktype='15',start='2010-01-01',end='2022.06.01') )
# print(ts.get_hs300s())

# generate configs
target = open("./config.yaml",'r',encoding='utf-8')
config = target.read()
data = yaml.load(config,Loader=yaml.FullLoader)

# get target filename
filename = data['stockts']['index']+'.'+data['stockts']['fre']+'.csv'

# get data via jqdatasdk
if not os.path.exists('../dataset/'+filename): 
    jq.auth(data['metadata']['username'],data['metadata']['password'])
    stockcodes = jq.get_index_stocks(data['stockts']['index'], date=None)
    df = jq.get_price(stockcodes, 
                    end_date=data['stockts']['enddate'], 
                    count=data['stockts']['count'], 
                    frequency=data['stockts']['fre'], 
                    fields=data['stockts']['fields'],
                    panel=False)

    # print(df.at[1,'code'])
    # save data to local directory
    df.to_csv('../dataset/'+filename)

    # generate labels
    # for example, if we want to predict the close price trend of 6.4, 
    # then we choose data from 6.3 as dataset and relevant trend of 6.3 and 6.4 as label 
    grouped = df.groupby('code')
    


# # group by stockcodes
# grouped = df.groupby('code')
# # print(grouped.get_group('600763.XSHG'))