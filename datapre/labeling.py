'''
This script is written for labeling raw data
'''
from os import O_RDONLY
import yaml

# generate configs
target = open("./config2.yaml",'r',encoding='utf-8')
config = target.read()
data = yaml.load(config,Loader=yaml.FullLoader)

# get target filename
datasetname = data['stockts']['index']+'.'+data['stockts']['fre']+'.csv'
labelname = data['stockts']['index']+'.'+data['stockts']['fre']+'.label.csv'


stock = open(datasetname,O_RDONLY)