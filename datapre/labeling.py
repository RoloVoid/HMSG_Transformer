'''
This script is written to label raw data
'''
import os
import pandas as pd
import numpy as np

prefix = "../dataset"

# window size 1*16 5*16 10*16 20*16 40*16
windows_size = [16,80,160,320,640]
beta_rise = 0.005
beta_fall = -0.00105
fixed_num = 16

# declare dirs
origin_dir = "csi500raw"
target_pdir = "csi500labeled"

if not os.path.exists(f"{prefix}/{target_pdir}"):
    os.mkdir(f"{prefix}/{target_pdir}")


m = os.listdir(f"{prefix}/{origin_dir}")

for target in m:
    data = pd.read_csv(f"{prefix}/{origin_dir}/{target}")
    for w in windows_size:

        # generate target dir
        datasetdir = prefix+"/"+target_pdir+"/"+str(w)
        if not os.path.exists(datasetdir):
            os.mkdir(datasetdir)

        # generate target group
        labeled = data.copy()

        # normalize shift length is fixed to 15
        labeled['preclose'] = data['close'].shift(15)
        labeled['prevolume'] = data['volume'].shift(15)
        labeled['open'] = (data['open']-labeled['preclose'])/labeled['preclose']
        labeled['high'] = (data['high']-labeled['preclose'])/labeled['preclose']
        labeled['low'] = (data['low']-labeled['preclose'])/labeled['preclose']
        labeled['close'] = (data['close']-labeled['preclose'])/labeled['preclose']
        labeled['volume'] = (data['volume']-labeled['prevolume'])/labeled['prevolume']

        # labeling
        labeled['label'] = 0
        labeled['label'][labeled['close'].map(lambda x: x<=beta_rise and x>=beta_fall)] = 0
        labeled['label'][labeled['close']>beta_rise] = 1
        labeled['label'][labeled['close']<beta_fall] = -1

        labeled = labeled.drop(['preclose','time','date','prevolume'],axis=1)
        labeled[:w-1] = -123321
        labeled.to_csv(datasetdir+"/"+target,index=False,header=None)
