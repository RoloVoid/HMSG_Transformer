# This script is for Chinese A shares data

import jqdatasdk as jq
import yaml

# We can use tushare free api part instead if we have no access to jqdatasdk
# import tushare as ts
# print(ts.get_hist_data('600847', ktype='15',start='2010-01-01',end='2022.06.01') )
# print(ts.get_hs300s())

# generate configs
target = open("./config.yaml",'r',encoding='utf-8')
config = target.read()
data = yaml.load(config,Loader=yaml.FullLoader)

# get data via jqdatasdk
jq.auth(data['metadata']['username'],data['metadata']['password'])
stockcodes = jq.get_index_stocks(data['stockts']['index'], date=None)
df = jq.get_price(stockcodes, 
                end_date=data['stockts']['enddate'], 
                count=data['stockts']['count'], 
                frequency=data['stockts']['fre'], 
                fields=data['stockts']['fields'],
                panel=False)

filename = data['stockts']['index']+'.'+data['stockts']['enddate']+'.'+data['stockts']['fre']+'.csv'
df.to_csv(filename)
# print(df.at[1,'code'])

# group by stockcodes
grouped = df.groupby('code')
# print(grouped.get_group('600763.XSHG'))