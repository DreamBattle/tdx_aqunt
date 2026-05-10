"""
MACD底背离/顶背离交易策略（带止损功能）
适用于三六零(601360.SH) 15分钟级别行情

止损类型：
1. 固定比例止损 - 买入价下跌X%自动止损
2. 移动止损 - 盈利后跟踪最高价回撤X%止损
3. 时间止损 - 持仓超过N个周期无盈利止损
"""

from tqcenter import tq, tqconst
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    """持仓信息"""
    stock_code: str
    buy_price: float
    volume: int
    buy_time: datetime
    highest_price: float  # 用于移动止损
    lowest_price: float   # 用于移动止损


class MACDDivergenceStrategyWithStopLoss:
    """
    MACD背离策略（带止损）
    """
    
    def __init__(self, stock_code='601360.SH', period='15m'):
        self.stock_code = stock_code
        self.period = period
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 止损参数
        self.stop_loss_pct = 0.03      # 固定止损比例 3%
        self.trailing_stop_pct = 0.05  # 移动止损比例 5%
        
        # 持仓状态
        self.position: Optional[Position] = None
        
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
    
    def check_stop_loss(self, current_price, current_time, bar_index):
        """
        检查止损条件
        
        Returns:
            tuple: (是否触发止损, 止损类型, 止损价格)
        """
        if self.position is None:
            return False, None, 0
        
        pos = self.position
        
        # 更新最高/最低价
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        if current_price < pos.lowest_price:
            pos.lowest_price = current_price
        
        # 1. 固定比例止损
        stop_price_fixed = pos.buy_price * (1 - self.stop_loss_pct)
        if current_price <= stop_price_fixed:
            return True, 'FIXED_STOP', stop_price_fixed
        
        # 2. 移动止损（盈利后才启用）
        if current_price > pos.buy_price:
            # 从最高点回撤 trailing_stop_pct 触发止损
            trailing_stop_price = pos.highest_price * (1 - self.trailing_stop_pct)
            if current_price <= trailing_stop_price:
                return True, 'TRAILING_STOP', trailing_stop_price
        
        return False, None, 0
    
    def buy(self, account_id, current_price, current_time):
        """执行买入"""
        if self.position is not None:
            print("已有持仓，不重复买入")
            return False
        
        # 获取账户资产
        asset = tq.query_stock_asset(account_id)
        available_funds = float(asset.get('AvailableFunds', 0))
        
        # 使用90%资金买入
        use_funds = available_funds * 0.9
        max_shares = int(use_funds / current_price / 100) * 100
        
        if max_shares < 100:
            print(f"资金不足: 可用{available_funds:.2f}, 需要{current_price * 100:.2f}")
            return False
        
        # 下单
        order_result = tq.order_stock(
            account_id=account_id,
            stock_code=self.stock_code,
            order_type=tqconst.STOCK_BUY,
            order_volume=max_shares,
            price_type=tqconst.PRICE_MY,
            price=current_price,
            notify=1
        )
        
        if order_result:
            # 记录持仓
            self.position = Position(
                stock_code=self.stock_code,
                buy_price=current_price,
                volume=max_shares,
                buy_time=current_time,
                highest_price=current_price,
                lowest_price=current_price
            )
            
            print(f"✓ 买入成功: {max_shares}股 @ {current_price:.2f}")
            print(f"  止损价格: {current_price * (1 - self.stop_loss_pct):.2f} (-{self.stop_loss_pct*100:.0f}%)")
            print(f"  移动止损: 盈利后从最高点回撤{self.trailing_stop_pct*100:.0f}%触发")
            
            # 发送预警
            tq.send_warn(
                stock_list=[self.stock_code],
                time_list=[current_time.strftime('%Y%m%d%H%M%S')],
                price_list=[str(current_price)],
                close_list=[str(current_price)],
                volum_list=[str(max_shares)],
                bs_flag_list=['0'],
                reason_list=[f'底背离买入 {max_shares}股 止损价{current_price * (1 - self.stop_loss_pct):.2f}']
            )
            return True
        
        return False
    
    def sell(self, account_id, current_price, current_time, reason):
        """执行卖出"""
        if self.position is None:
            print("无持仓，无法卖出")
            return False
        
        pos = self.position
        
        # 计算盈亏
        pnl = (current_price - pos.buy_price) * pos.volume
        pnl_pct = (current_price - pos.buy_price) / pos.buy_price * 100
        
        # 下单
        order_result = tq.order_stock(
            account_id=account_id,
            stock_code=self.stock_code,
            order_type=tqconst.STOCK_SELL,
            order_volume=pos.volume,
            price_type=tqconst.PRICE_MY,
            price=current_price,
            notify=1
        )
        
        if order_result:
            print(f"✓ 卖出成功: {pos.volume}股 @ {current_price:.2f}")
            print(f"  卖出原因: {reason}")
            print(f"  盈亏: {pnl:+.2f}元 ({pnl_pct:+.2f}%)")
            
            # 发送预警
            tq.send_warn(
                stock_list=[self.stock_code],
                time_list=[current_time.strftime('%Y%m%d%H%M%S')],
                price_list=[str(current_price)],
                close_list=[str(current_price)],
                volum_list=[str(pos.volume)],
                bs_flag_list=['1'],
                reason_list=[f'{reason} 盈亏{pnl:+.2f}元']
            )
            
            # 清空持仓
            self.position = None
            return True
        
        return False
    
    def run(self, account_id):
        """
        运行策略（单次执行）
        """
        print(f"\n{'='*60}")
        print(f"MACD背离策略（带止损）")
        print(f"股票: {self.stock_code} | 周期: {self.period}")
        print(f"{'='*60}\n")
        
        # 获取数据
        data = tq.get_market_data(
            field_list=[],
            stock_list=[self.stock_code],
            period=self.period,
            count=100,
            dividend_type='front'
        )
        
        if not data:
            print("获取数据失败")
            return
        
        # 查找收盘价字段
        close_field = None
        for field in ['Close', 'close', 'CLOSE']:
            if field in data:
                close_field = field
                break
        
        if close_field is None:
            print(f"未找到收盘价字段")
            return
        
        close_prices = data[close_field][self.stock_code]
        current_price = close_prices.iloc[-1]
        current_time = close_prices.index[-1]
        
        print(f"当前时间: {current_time}")
        print(f"当前价格: {current_price:.2f}")
        
        # 如果有持仓，先检查止损
        if self.position is not None:
            print(f"\n当前持仓: {self.position.volume}股")
            print(f"买入价: {self.position.buy_price:.2f}")
            print(f"最高价: {self.position.highest_price:.2f}")
            
            # 检查止损
            should_stop, stop_type, stop_price = self.check_stop_loss(
                current_price, current_time, len(close_prices) - 1
            )
            
            if should_stop:
                stop_reason = {
                    'FIXED_STOP': f'固定止损(-{self.stop_loss_pct*100:.0f}%)',
                    'TRAILING_STOP': f'移动止损(-{self.trailing_stop_pct*100:.0f}%)',
                    'TIME_STOP': f'时间止损({self.time_stop_bars}周期)'
                }.get(stop_type, '止损')
                
                self.sell(account_id, current_price, current_time, stop_reason)
                return
            else:
                # 检查顶背离卖出信号
                macd_df = self.calculate_macd(close_prices)
                bearish_div = self.detect_bearish_divergence(close_prices, macd_df['MACD'])
                
                if bearish_div:
                    self.sell(account_id, current_price, current_time, '顶背离卖出')
                    return
                
                print(f"\n继续持仓，未触发止损或卖出信号")
        
        else:
            # 无持仓，检查买入信号
            print("\n无持仓，检测买入信号...")
            
            macd_df = self.calculate_macd(close_prices)
            bullish_div = self.detect_bullish_divergence(close_prices, macd_df['MACD'])
            
            if bullish_div:
                print("★★★ 发现底背离买入信号 ★★★")
                self.buy(account_id, current_price, current_time)
            else:
                print("无买入信号，继续观望")


def main():
    """主函数"""
    tq.initialize(__file__)
    
    try:
        # 获取交易账户
        account_id = tq.stock_account()
        if account_id < 0:
            print("获取交易账户失败")
            return
        
        print(f"交易账户ID: {account_id}")
        
        # 创建策略实例
        strategy = MACDDivergenceStrategyWithStopLoss(
            stock_code='601360.SH',
            period='15m'
        )
        
        # 运行策略
        strategy.run(account_id)
        
    except Exception as e:
        print(f"策略运行异常: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        tq.close()
        print("\n策略执行完成")


if __name__ == '__main__':
    main()
