import baostock as bs
import pandas as pd

## system.login()##
lg = bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

## Get CSI500 components
sts = bs.query_zz500_stocks()
print('query_zz500 error_code:'+sts.error_code)
print('query_zz500  error_msg:'+sts.error_msg)

# 打印结果集
zz500_stocks = []
codes = []
while (sts.error_code == '0') & sts.next():
    # 获取一条记录，将记录合并在一起
    zz500_stocks.append(sts.get_row_data())
    codes.append(sts.get_row_data()[1])
result = pd.DataFrame(zz500_stocks, columns=sts.fields)
# 结果集输出到csv文件
result.to_csv("../dataset/csi500.csv", index=False)

for x in codes:
    rs = bs.query_history_k_data_plus(x,
        "date,time,open,high,low,close,volume",
        start_date='2018-06-04', end_date='2022-06-10',
        frequency="15")
    print(x)
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result.to_csv(f'../dataset/csi500/{x}.csv', index=False)

# logout
bs.logout()