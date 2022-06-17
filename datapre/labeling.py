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

        # normalize
        labeled['preclose'] = data['close'].shift(w)
        labeled['open'] = (data['open']-labeled['preclose'])/labeled['preclose']
        labeled['high'] = (data['high']-labeled['preclose'])/labeled['preclose']
        labeled['low'] = (data['low']-labeled['preclose'])/labeled['preclose']

        # labeling
        labeled['label']=labeled['close'].pct_change(periods=w)
        labeled['label'][labeled['label'].map(lambda x: x<=beta_rise and x>=beta_fall)] = 0
        labeled['label'][labeled['label']>beta_rise] = 1
        labeled['label'][labeled['label']<beta_fall] = -1

        labeled = labeled.drop('preclose',axis=1).dropna()
        labeled.to_csv(datasetdir+"/"+target)
