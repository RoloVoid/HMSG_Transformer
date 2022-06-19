import pandas as pd
import torch
from timeit import default_timer as timer
m = pd.read_csv("csi500date.csv")

print(m[m['date']=="2022-06-10"].index[0])

# n = pd.read_csv("csi500.csv").T.values.tolist()[0]
# print(n)

s = timer()
q = torch.Tensor(pd.read_csv("./csi500labeled/16/sh.600006.csv").values)[0:15,0:-1]
n = timer()
print(q)

m = [1,1,1]

def test(n):
    for i in n:
        print(i)

test(m)