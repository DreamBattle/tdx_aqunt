"""
MACD背离实时监控策略
持续监控三六零(601360.SH) 15分钟级别行情，自动检测背离并交易
完整实现回测策略逻辑：
1. 5分钟趋势过滤
2. 30/60分钟级联确认
3. 移动止损 + 趋势止损
"""

from tqcenter import tq, tqconst
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json


class MACDDivergenceMonitor:
    """
    MACD背离实时监控系统（完整策略版）
    """
    
    def __init__(self, stock_code='601360.SH', period='15m'):
        self.stock_code = stock_code
        self.period = period
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 止损参数
        self.use_fixed_stop = False     # 是否启用固定止损
        self.stop_loss_pct = 0.03      # 固定止损比例 3%
        self.trailing_stop_pct = 0.05  # 移动止损比例 5%
        
        # 趋势过滤参数（5分钟级别）
        self.enable_trend_filter = True  # 是否启用趋势过滤
        self.trend_period = '5m'        # 趋势判断周期
        
        # 多级别确认参数
        self.confirm_period_30m = '30m'  # 30分钟确认周期
        self.confirm_period_60m = '60m'  # 60分钟确认周期
        
        # 持仓状态
        self.position = 0              # 持仓数量
        self.buy_price = 0.0          # 买入价格
        self.buy_amount = 0.0         # 买入金额
        self.highest_price = 0.0      # 持仓期间最高价
        self.lowest_price = 0.0       # 持仓期间最低价
        self.buy_bar_count = 0        # 买入后K线计数（用于延迟止损检查）
        
        # 信号状态
        self.last_signal = 'HOLD'
        self.signal_count = 0
        self.trade_count = 0
        
    def calculate_macd(self, close_prices):
        """计算MACD指标"""
        ema_fast = close_prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close_prices.ewm(span=self.macd_slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.macd_signal, adjust=False).mean()
        macd_hist = (dif - dea) * 2
        
        return pd.DataFrame({
            'DIF': dif,
            'DEA': dea,
            'MACD': macd_hist
        })
    
    def find_local_extrema(self, series, window=5):
        """寻找局部极值点"""
        lows = []
        highs = []
        
        for i in range(window, len(series) - window):
            if all(series.iloc[i] <= series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] <= series.iloc[i+j] for j in range(1, window+1)):
                lows.append(i)
            
            if all(series.iloc[i] >= series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] >= series.iloc[i+j] for j in range(1, window+1)):
                highs.append(i)
        
        return lows, highs
    
    def detect_bullish_divergence(self, prices, macd_hist):
        """检测底背离"""
        price_lows, _ = self.find_local_extrema(prices)
        macd_lows, _ = self.find_local_extrema(macd_hist)
        
        if len(price_lows) < 2 or len(macd_lows) < 2:
            return False
        
        recent_price_lows = price_lows[-2:]
        price_low_1 = prices.iloc[recent_price_lows[0]]
        price_low_2 = prices.iloc[recent_price_lows[1]]
        
        macd_at_price_low_1 = macd_hist.iloc[recent_price_lows[0]]
        macd_at_price_low_2 = macd_hist.iloc[recent_price_lows[1]]
        
        price_lower_low = price_low_2 < price_low_1
        macd_not_lower_low = macd_at_price_low_2 >= macd_at_price_low_1
        
        return price_lower_low and macd_not_lower_low
    
    def detect_bearish_divergence(self, prices, macd_hist):
        """检测顶背离"""
        _, price_highs = self.find_local_extrema(prices)
        _, macd_highs = self.find_local_extrema(macd_hist)
        
        if len(price_highs) < 2 or len(macd_highs) < 2:
            return False
        
        recent_price_highs = price_highs[-2:]
        price_high_1 = prices.iloc[recent_price_highs[0]]
        price_high_2 = prices.iloc[recent_price_highs[1]]
        
        macd_at_price_high_1 = macd_hist.iloc[recent_price_highs[0]]
        macd_at_price_high_2 = macd_hist.iloc[recent_price_highs[1]]
        
        price_higher_high = price_high_2 > price_high_1
        macd_not_higher_high = macd_at_price_high_2 <= macd_at_price_high_1
        
        return price_higher_high and macd_not_higher_high
    
    def is_downward_trend(self) -> tuple:
        """
        判断5分钟级别是否处于下降趋势
        返回True表示下降趋势，此时不应买入
        判断条件：MA55 < MA89 < MA181 < MA420
        """
        if not self.enable_trend_filter:
            return False, {}
        
        data = tq.get_market_data(
            field_list=[],
            stock_list=[self.stock_code],
            period=self.trend_period,
            count=420,
            dividend_type='front'
        )
        
        if not data:
            return False, {}
        
        close_field = None
        for field in ['Close', 'close', 'CLOSE']:
            if field in data:
                close_field = field
                break
        
        if close_field is None:
            return False, {}
        
        close_prices = data[close_field][self.stock_code]
        
        ma55 = close_prices.rolling(55).mean()
        ma89 = close_prices.rolling(89).mean()
        ma181 = close_prices.rolling(181).mean()
        ma420 = close_prices.rolling(420).mean()
        
        condition1 = ma55.iloc[-1] < ma89.iloc[-1]
        condition2 = ma89.iloc[-1] < ma181.iloc[-1]
        condition3 = ma181.iloc[-1] < ma420.iloc[-1]
        
        details = {
            'ma55': round(ma55.iloc[-1], 2) if len(ma55) > 0 else None,
            'ma89': round(ma89.iloc[-1], 2) if len(ma89) > 0 else None,
            'ma181': round(ma181.iloc[-1], 2) if len(ma181) > 0 else None,
            'ma420': round(ma420.iloc[-1], 2) if len(ma420) > 0 else None,
            'condition1': condition1,
            'condition2': condition2,
            'condition3': condition3
        }
        
        is_downward = condition1 and condition2 and condition3
        
        return is_downward, details
    
    def detect_bullish_divergence_at_period(self, period) -> bool:
        """检测指定周期的底背离信号"""
        data = tq.get_market_data(
            field_list=[],
            stock_list=[self.stock_code],
            period=period,
            count=100,
            dividend_type='front'
        )
        
        if not data:
            return False
        
        close_field = None
        for field in ['Close', 'close', 'CLOSE']:
            if field in data:
                close_field = field
                break
        
        if close_field is None:
            return False
        
        close_prices = data[close_field][self.stock_code]
        macd_df = self.calculate_macd(close_prices)
        
        return self.detect_bullish_divergence(close_prices, macd_df['MACD'])
    
    def check_15min_trend_stop(self) -> bool:
        """
        检查15分钟级别趋势是否满足止损条件
        止损条件: MA24 < MA55 < MA89
        """
        data = tq.get_market_data(
            field_list=[],
            stock_list=[self.stock_code],
            period='15m',
            count=100,
            dividend_type='front'
        )
        
        if not data:
            return False
        
        close_field = None
        for field in ['Close', 'close', 'CLOSE']:
            if field in data:
                close_field = field
                break
        
        if close_field is None:
            return False
        
        close_prices = data[close_field][self.stock_code]
        
        ma24 = close_prices.rolling(24).mean()
        ma55 = close_prices.rolling(55).mean()
        ma89 = close_prices.rolling(89).mean()
        
        if len(ma24) > 0 and len(ma55) > 0 and len(ma89) > 0:
            condition = ma24.iloc[-1] < ma55.iloc[-1] < ma89.iloc[-1]
            return condition
        
        return False
    
    def check_stop_loss(self, current_price) -> tuple:
        """检查止损条件"""
        if self.position <= 0:
            return False, '', 0
        
        self.buy_bar_count += 1
        if self.buy_bar_count <= 1:
            return False, '', 0
        
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price:
            self.lowest_price = current_price
        
        # 1. 移动止损（盈利后）
        if current_price > self.buy_price:
            trailing_stop_price = self.highest_price * (1 - self.trailing_stop_pct)
            if current_price <= trailing_stop_price:
                return True, '移动止损', trailing_stop_price
        
        # 2. 固定止损
        if self.use_fixed_stop:
            stop_price_fixed = self.buy_price * (1 - self.stop_loss_pct)
            if current_price <= stop_price_fixed:
                return True, '固定止损', stop_price_fixed
        
        # 3. 趋势止损（亏损3%以上）
        if current_price < self.buy_price * 0.97:
            if self.check_15min_trend_stop():
                return True, '趋势止损', current_price
        
        return False, '', 0
    
    def check_signal(self):
        """检测当前信号（完整策略逻辑）"""
        try:
            data = tq.get_market_data(
                field_list=[],
                stock_list=[self.stock_code],
                period=self.period,
                count=100,
                dividend_type='front'
            )
            
            if not data:
                return None
            
            close_field = None
            for field in ['Close', 'close', 'CLOSE']:
                if field in data:
                    close_field = field
                    break
            
            if close_field is None:
                return None
            
            close_prices = data[close_field][self.stock_code]
            
            if len(close_prices) < 50:
                return None
            
            current_price = close_prices.iloc[-1]
            
            # 有持仓时检查止损
            if self.position > 0:
                should_stop, stop_type, stop_price = self.check_stop_loss(current_price)
                if should_stop:
                    return {
                        'signal': 'SELL',
                        'signal_type': stop_type,
                        'current_price': current_price,
                        'sell_reason': stop_type
                    }
            
            # 无持仓时检测买入信号
            if self.position <= 0:
                macd_df = self.calculate_macd(close_prices)
                bullish_div = self.detect_bullish_divergence(close_prices, macd_df['MACD'])
                bearish_div = self.detect_bearish_divergence(close_prices, macd_df['MACD'])
                
                if bullish_div:
                    # 检查5分钟趋势
                    is_downward, trend_details = self.is_downward_trend()
                    
                    if not is_downward:
                        return {
                            'signal': 'BUY',
                            'signal_type': '底背离',
                            'current_price': current_price,
                            'trend_filter': '通过',
                            'trend_details': trend_details
                        }
                    else:
                        # 5分钟趋势向下，检查30分钟
                        div_30min = self.detect_bullish_divergence_at_period(self.confirm_period_30m)
                        
                        if not div_30min:
                            return {
                                'signal': 'FILTERED_30M',
                                'signal_type': '等待30分钟确认',
                                'current_price': current_price,
                                'trend_filter': '过滤'
                            }
                        else:
                            # 30分钟也检测到，继续检查60分钟
                            div_60min = self.detect_bullish_divergence_at_period(self.confirm_period_60m)
                            
                            if not div_60min:
                                return {
                                    'signal': 'FILTERED_60M',
                                    'signal_type': '等待60分钟确认',
                                    'current_price': current_price,
                                    'trend_filter': '过滤'
                                }
                            else:
                                return {
                                    'signal': 'BUY',
                                    'signal_type': '底背离(60分钟确认)',
                                    'current_price': current_price,
                                    'trend_filter': '60分钟确认',
                                    'trend_details': trend_details
                                }
                
                if bearish_div:
                    return {
                        'signal': 'SELL',
                        'signal_type': '顶背离',
                        'current_price': current_price,
                        'sell_reason': '顶背离'
                    }
            
            return {
                'signal': 'HOLD',
                'signal_type': '持有/等待',
                'current_price': current_price
            }
            
        except Exception as e:
            print(f"检测信号异常: {e}")
            return None
    
    def execute_buy(self, account_id, signal_data):
        """执行买入"""
        current_price = signal_data['current_price']
        signal_type = signal_data['signal_type']
        
        print(f"[{datetime.now()}] 执行买入: {self.stock_code} @ {current_price:.2f}")
        
        asset = tq.query_stock_asset(account_id)
        available_funds = float(asset.get('AvailableFunds', 0))
        
        max_shares = int(available_funds * 0.9 / current_price / 100) * 100
        
        if max_shares >= 100:
            order_result = tq.order_stock(
                account_id=account_id,
                stock_code=self.stock_code,
                order_type=tqconst.STOCK_BUY,
                order_volume=max_shares,
                price_type=tqconst.PRICE_MY,
                price=current_price,
                notify=1
            )
            print(f"买入结果: {order_result}")
            
            # 更新持仓状态
            self.position = max_shares
            self.buy_price = current_price
            self.buy_amount = current_price * max_shares
            self.highest_price = current_price
            self.lowest_price = current_price
            self.buy_bar_count = 0
            
            # 发送预警
            tq.send_warn(
                stock_list=[self.stock_code],
                time_list=[datetime.now().strftime('%Y%m%d%H%M%S')],
                price_list=[str(current_price)],
                close_list=[str(current_price)],
                volum_list=['0'],
                bs_flag_list=['0'],
                reason_list=[f'{signal_type}买入 {max_shares}股']
            )
            self.trade_count += 1
            return True
        else:
            print("资金不足，无法买入")
            return False
    
    def execute_sell(self, account_id, signal_data):
        """执行卖出"""
        current_price = signal_data['current_price']
        sell_reason = signal_data.get('sell_reason', '顶背离')
        
        print(f"[{datetime.now()}] 执行卖出: {self.stock_code} @ {current_price:.2f} ({sell_reason})")
        
        positions = tq.query_stock_positions(account_id)
        sell_volume = 0
        
        for pos in positions:
            if pos['StockCode'] == self.stock_code:
                sell_volume = int(pos.get('UseableVolume', 0))
                break
        
        if sell_volume > 0:
            order_result = tq.order_stock(
                account_id=account_id,
                stock_code=self.stock_code,
                order_type=tqconst.STOCK_SELL,
                order_volume=sell_volume,
                price_type=tqconst.PRICE_MY,
                price=current_price,
                notify=1
            )
            print(f"卖出结果: {order_result}")
            
            # 重置持仓状态
            pnl = (current_price - self.buy_price) * self.position
            print(f"交易盈亏: {pnl:.2f}元 ({((current_price - self.buy_price)/self.buy_price*100):.2f}%)")
            
            self.position = 0
            self.buy_price = 0.0
            self.buy_amount = 0.0
            self.highest_price = 0.0
            self.lowest_price = 0.0
            self.buy_bar_count = 0
            
            # 发送预警
            tq.send_warn(
                stock_list=[self.stock_code],
                time_list=[datetime.now().strftime('%Y%m%d%H%M%S')],
                price_list=[str(current_price)],
                close_list=[str(current_price)],
                volum_list=['0'],
                bs_flag_list=['1'],
                reason_list=[f'{sell_reason}卖出 {sell_volume}股']
            )
            self.trade_count += 1
            return True
        else:
            print("无持仓，无法卖出")
            return False
    
    def run(self, auto_trade=False, check_interval=60):
        """运行实时监控"""
        print(f"\n{'='*60}")
        print(f"MACD背离实时监控系统（完整策略版）")
        print(f"股票: {self.stock_code}")
        print(f"周期: {self.period}")
        print(f"自动交易: {'是' if auto_trade else '否'}")
        print(f"检测间隔: {check_interval}秒")
        print(f"{'='*60}\n")
        
        account_id = -1
        if auto_trade:
            account_id = tq.stock_account()
            if account_id < 0:
                print("获取交易账户失败，将仅监控不交易")
                auto_trade = False
            else:
                print(f"交易账户ID: {account_id}")
        
        print(f"[{datetime.now()}] 开始监控...\n")
        
        try:
            while True:
                signal_data = self.check_signal()
                
                if signal_data:
                    current_time = datetime.now()
                    signal = signal_data['signal']
                    price = signal_data['current_price']
                    signal_type = signal_data['signal_type']
                    
                    # 只在信号变化或有持仓时输出
                    if signal != self.last_signal or self.position > 0:
                        if signal == 'BUY':
                            print(f"[{current_time}] [买入] @ {price:.2f}")
                            print(f"        ┌─ 买入原因:")
                            print(f"        │  • 15分钟级别检测到底背离信号")
                            
                            trend_details = signal_data.get('trend_details')
                            if trend_details:
                                print(f"        │  • 5分钟均线状态: MA55({trend_details['ma55']}) MA89({trend_details['ma89']}) MA181({trend_details['ma181']}) MA420({trend_details['ma420']})")
                                print(f"        │  └─ 均线判断: MA55<MA89={trend_details['condition1']}, MA89<MA181={trend_details['condition2']}, MA181<MA420={trend_details['condition3']}")
                            
                            print(f"        └─ 买入信号类型: {signal_type}")
                            
                            if auto_trade:
                                self.execute_buy(account_id, signal_data)
                            
                            self.signal_count += 1
                            
                        elif signal == 'SELL':
                            sell_reason = signal_data.get('sell_reason', '顶背离')
                            print(f"[{current_time}] [卖出] @ {price:.2f} | {sell_reason}")
                            
                            if auto_trade:
                                self.execute_sell(account_id, signal_data)
                            
                            self.signal_count += 1
                            
                        elif signal.startswith('FILTERED'):
                            print(f"[{current_time}] [过滤] {signal_type}")
                            
                        elif signal == 'HOLD':
                            if self.position > 0:
                                print(f"[{current_time}] [持有] 继续持有 | 当前价: {price:.2f} | 成本: {self.buy_price:.2f}")
                            else:
                                print(f"[{current_time}] [等待] 无买入信号")
                        
                        self.last_signal = signal
                    else:
                        # 每5分钟输出一次状态
                        if int(current_time.timestamp()) % 300 < check_interval:
                            status = f"持仓: {self.position}股" if self.position > 0 else "空仓"
                            print(f"[{current_time}] 监控中... 价格: {price:.2f} | {status} | 累计信号: {self.signal_count}")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n监控已停止")
            print(f"累计产生信号: {self.signal_count}次")
            print(f"累计交易次数: {self.trade_count}次")


def main():
    """主函数"""
    tq.initialize(__file__)
    
    monitor = MACDDivergenceMonitor(
        stock_code='601360.SH',
        period='15m'
    )
    
    monitor.run(
        auto_trade=False,
        check_interval=60
    )
    
    tq.close()


if __name__ == '__main__':
    main()
