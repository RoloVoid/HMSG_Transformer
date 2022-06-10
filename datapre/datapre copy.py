# This script is for Chinese A shares data -- dzy 2022.6.2

import jqdatasdk as jq
import yaml
import os.path

# We can attempt to use old tushare free api part instead if we have no access to jqdatasdk, but there is only data in past three months 
# ToDo: implement tushare methods --Dzy
# import tushare as ts
# print(ts.get_hist_data('600847', ktype='15',start='2010-01-01',end='2022.06.01') )
# print(ts.get_hs300s())

# generate configs
target = open("./config2.yaml",'r',encoding='utf-8')
config = target.read()
data = yaml.load(config,Loader=yaml.FullLoader)

# get target filename
datasetname = data['stockts']['index']+'.'+data['stockts']['fre']+'.csv'
labelname = data['stockts']['index']+'.'+data['stockts']['fre']+'.label.csv'
tagname = data['stockts']['index']+'.csv'

# for label use
def apply_label(tg,beta_rise,beta_fall):    
    if tg < beta_fall: return -1
    if tg > beta_rise: return 1
    return 0

# get data via jqdatasdk
if not os.path.exists('../dataset/'+datasetname):
    # get data
    jq.auth(data['metadata']['username'],data['metadata']['password'])
    stockcodes = jq.get_index_stocks(data['stockts']['index'], date=None)
    # this part is specially for daily/minunitely data
    # dataset = jq.get_price(stockcodes, 
    #                 end_date=data['stockts']['enddate'], 
    #                 count=data['stockts']['count'], 
    #                 frequency=data['stockts']['fre'], 
    #                 fields=data['stockts']['fields'],
    #                 panel=False)

    dataset = jq.get_bars(stockcodes,
        count=data['stockts']['count'],
        unit=data['stockts']['fre'],
        fields=data['stockts']['fields']
        )

    # save data to local directory
    dataset.to_csv('../dataset/'+datasetname)
    # group by stockcodes
    # grouped = dataset.groupby('code')
    # print(type(grouped.get_group(stockcodes[0])))


# generate labels    
# for example, if we want to predict the close price trend of 6.4, then step = 2
# then we choose data from 6.2 as dataset and relative trend of 6.2 and 6.4 as label
if not os.path.exists('../dataset/'+labelname):
    # get data
    jq.auth(data['metadata']['username'],data['metadata']['password'])
    stockcodes = jq.get_index_stocks(data['stockts']['index'], date=None)
    lag = jq.get_price(stockcodes,
                        end_date=data['stockts']['tgdate'],
                        count=data['stockts']['count']+1,
                        frequency=data['stockts']['fre'],
                        fields = data['stockts']['tgt'],
                        panel=False)

    lag = lag.pivot(index='time',columns='code',values=data['stockts']['tgt'])
    labels = lag.pct_change(periods=data['Step'])
 
    labels = labels.dropna(axis=1,thresh=data['stockts']['count']-data['Step']+1).\
        dropna().applymap(apply_label,beta_rise=data['Threshold']['beta_rise'],beta_fall=data['Threshold']['beta_fall'])

    # save data to local directory
    labels.to_csv('../dataset/'+labelname)

