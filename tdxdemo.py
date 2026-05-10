
# 使用tqcenter的API函数查看平安银行日线数据示例
from tqcenter import tq

#初始化
tq.initialize(__file__) #所有策略连接通达信客户端都必须调用此函数进行初始化

#获取平安银行日线前复权收盘数据
df = tq.get_market_data(
        field_list = ['Close'],
        stock_list = ["000001.SZ"],
        start_time = '20251219',
        end_time = '20251225',
        dividend_type='front',
        period='1d',
    )
print(df)

