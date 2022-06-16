import baostock as bs
import pandas as pd

## system.login() ##
lg = bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# Get hs300 component
rs = bs.query_hs300_stocks()
print('query_hs300 error_code:'+rs.error_code)
print('query_hs300  error_msg:'+rs.error_msg)

hs300_stocks = []
while (rs.error_code == '0') & rs.next():
    hs300_stocks.append(rs.get_row_data())
result = pd.DataFrame(hs300_stocks, columns=rs.fields)
result.to_csv("../dataset/hs300.csv", index=False)

# logout
bs.logout()