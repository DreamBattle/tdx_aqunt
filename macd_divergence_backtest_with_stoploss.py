"""
MACD底背离/顶背离策略回测系统（带止损功能）
适用于三六零(601360.SH) 15分钟级别行情

止损类型：
1. 固定比例止损 - 买入价下跌X%自动止损
2. 移动止损 - 盈利后跟踪最高价回撤X%止损
3. 时间止损 - 持仓超过N个周期无盈利止损
"""

from tqcenter import tq, tqconst
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class TradeRecord:
    """交易记录"""
    trade_id: int
    trade_time: datetime
    stock_code: str
    trade_type: str  # 'BUY' or 'SELL'
    price: float
    volume: int
    amount: float
    signal_type: str  # '底背离', '顶背离', '固定止损', '移动止损', '时间止损'
    pnl: float = 0.0  # 盈亏（卖出时计算）


@dataclass
class BacktestResult:
    """回测结果"""
    stock_code: str
    period: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    # 止损统计
    stop_loss_count: int = 0
    stop_loss_types: Dict[str, int] = field(default_factory=dict)
    # 趋势过滤统计
    filtered_signals_count: int = 0
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


class MACDDivergenceBacktestWithStopLoss:
    """
    MACD背离策略回测引擎（带止损）
    """
    
    def __init__(self, stock_code='601360.SH', period='15m', initial_capital=100000):
        self.stock_code = stock_code
        self.period = period
        self.initial_capital = initial_capital
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
        self.trend_ma_periods = [89, 181, 420]  # 判断趋势的均线周期
        
        # 回测状态
        self.current_capital = initial_capital
        self.position = 0              # 持仓数量
        self.position_value = 0.0
        self.trades: List[TradeRecord] = []
        self.equity_curve_data = []
        self.trade_id = 0
        
        # 持仓信息
        self.buy_price = 0.0
        self.buy_amount = 0.0
        self.highest_price = 0.0       # 持仓期间最高价
        self.lowest_price = 0.0        # 持仓期间最低价
        self.buy_bar_count = 0         # 买入后K线计数（用于延迟止损检查）
        
        # 止损统计
        self.stop_loss_count = 0
        self.stop_loss_types = {'FIXED': 0, 'TRAILING': 0, 'TREND': 0}
        
        # 趋势过滤统计
        self.filtered_signals_count = 0  # 被趋势过滤的信号数量
        
        # 多级别确认参数
        self.confirm_period_30m = '30m'  # 30分钟确认周期
        self.confirm_period_60m = '60m'  # 60分钟确认周期
        
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
            return False, {}
        
        recent_price_lows = price_lows[-2:]
        price_low_1 = prices.iloc[recent_price_lows[0]]
        price_low_2 = prices.iloc[recent_price_lows[1]]
        
        macd_at_price_low_1 = macd_hist.iloc[recent_price_lows[0]]
        macd_at_price_low_2 = macd_hist.iloc[recent_price_lows[1]]
        
        price_lower_low = price_low_2 < price_low_1
        macd_not_lower_low = macd_at_price_low_2 >= macd_at_price_low_1
        
        result = price_lower_low and macd_not_lower_low
        
        details = {
            'price_low_index_1': recent_price_lows[0],
            'price_low_index_2': recent_price_lows[1],
            'price_low_1': round(price_low_1, 2),
            'price_low_2': round(price_low_2, 2),
            'macd_at_low_1': round(macd_at_price_low_1, 4),
            'macd_at_low_2': round(macd_at_price_low_2, 4),
            'price_lower_low': price_lower_low,
            'macd_not_lower_low': macd_not_lower_low,
            'divergence_detected': result
        }
        
        return result, details
    
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
    
    def check_stop_loss(self, current_price, timestamp) -> Tuple[bool, str, float]:
        """
        检查止损条件
        
        Args:
            current_price: 当前价格
            timestamp: 当前回测时间戳
            
        Returns:
            (是否触发, 止损类型, 止损价格)
        """
        if self.position <= 0:
            return False, '', 0
        
        # 买入后至少等待1根K线再检查止损（避免买入后立即被止损）
        self.buy_bar_count += 1
        if self.buy_bar_count <= 1:
            return False, '', 0
        
        # 更新最高/最低价
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price:
            self.lowest_price = current_price
        
        # 1. 移动止损（盈利后才启用，优先检查）
        if current_price > self.buy_price:
            trailing_stop_price = self.highest_price * (1 - self.trailing_stop_pct)
            if current_price <= trailing_stop_price:
                return True, '移动止损', trailing_stop_price
        
        # 2. 固定比例止损（其次检查，仅启用时生效）
        if self.use_fixed_stop:
            stop_price_fixed = self.buy_price * (1 - self.stop_loss_pct)
            if current_price <= stop_price_fixed:
                return True, '固定止损', stop_price_fixed
        
        # 3. 趋势止损（亏损3%以上时检查15分钟均线趋势）
        if current_price < self.buy_price * 0.97:  # 亏损超过3%才检查
            if self.check_15min_trend_stop(timestamp):
                return True, '趋势止损', current_price
        
        return False, '', 0
    
    def check_15min_trend_stop(self, timestamp) -> bool:
        """
        检查15分钟级别趋势是否满足止损条件
        止损条件: 15分钟24线 < 15分钟55线 < 15分钟89线
        
        Args:
            timestamp: 当前回测时间戳，用于获取历史数据
            
        Returns:
            bool: 是否满足趋势止损条件
        """
        # 获取15分钟级别数据
        data = tq.get_market_data(
            field_list=[],
            stock_list=[self.stock_code],
            period='15m',
            count=100,  # 足够的数据计算均线
            end_time=timestamp.strftime('%Y%m%d%H%M%S'),
            dividend_type='front'
        )
        
        if not data:
            return False
        
        # 查找收盘价字段
        close_field = None
        for field in ['Close', 'close', 'CLOSE']:
            if field in data:
                close_field = field
                break
        
        if close_field is None:
            return False
        
        close_prices = data[close_field][self.stock_code]
        
        # 计算均线
        ma24 = close_prices.rolling(24).mean()
        ma55 = close_prices.rolling(55).mean()
        ma89 = close_prices.rolling(89).mean()
        
        # 检查均线条件：24线 < 55线 < 89线
        if len(ma24) > 0 and len(ma55) > 0 and len(ma89) > 0:
            condition = ma24.iloc[-1] < ma55.iloc[-1] < ma89.iloc[-1]
            return condition
        
        return False
    
    def is_downward_trend(self, timestamp) -> tuple:
        """
        判断5分钟级别是否处于下降趋势
        
        Args:
            timestamp: 当前回测时间戳，用于获取历史数据
            
        返回True表示下降趋势，此时不应买入
        判断条件：
            1. 5分钟级别181线 < 5分钟级别420线
            2. 5分钟级别89线 < 5分钟级别181线
            3. 5分钟级别55线 < 5分钟级别89线
        
        Returns:
            tuple: (是否下降趋势, 条件详情字典)
        """
        if not self.enable_trend_filter:
            return False, {}
        
        # 获取5分钟级别数据（需要420根数据计算均线）
        # 使用timestamp作为结束时间，确保获取的是历史数据而非最新数据
        data = tq.get_market_data(
            field_list=[],
            stock_list=[self.stock_code],
            period=self.trend_period,
            count=420,  # 420均线所需数据
            end_time=timestamp.strftime('%Y%m%d%H%M%S'),
            dividend_type='front'
        )
        
        if not data:
            return False, {}
        
        # 查找收盘价字段
        close_field = None
        for field in ['Close', 'close', 'CLOSE']:
            if field in data:
                close_field = field
                break
        
        if close_field is None:
            return False, {}
        
        close_prices = data[close_field][self.stock_code]
        
        # 计算各周期均线
        ma55 = close_prices.rolling(55).mean()
        ma89 = close_prices.rolling(89).mean()
        ma181 = close_prices.rolling(181).mean()
        ma420 = close_prices.rolling(420).mean()
        
        # 条件1: 181线 < 420线
        condition1 = ma181.iloc[-1] < ma420.iloc[-1]
        
        # 条件2: 89线 < 181线
        condition2 = ma89.iloc[-1] < ma181.iloc[-1]
        
        # 条件3: 55线 < 89线
        condition3 = ma55.iloc[-1] < ma89.iloc[-1]
        
        # 收集条件详情
        details = {
            'ma55': round(ma55.iloc[-1], 2) if len(ma55) > 0 else None,
            'ma89': round(ma89.iloc[-1], 2) if len(ma89) > 0 else None,
            'ma181': round(ma181.iloc[-1], 2) if len(ma181) > 0 else None,
            'ma420': round(ma420.iloc[-1], 2) if len(ma420) > 0 else None,
            'condition1': condition1,  # 181 < 420
            'condition2': condition2,  # 89 < 181
            'condition3': condition3   # 55 < 89
        }
        
        # 所有条件都满足才认为是下降趋势
        is_downward = condition1 and condition2 and condition3
        
        return is_downward, details
    
    def is_downward_trend_at_period(self, timestamp, period) -> tuple:
        """
        判断指定周期是否处于下降趋势
        
        Args:
            timestamp: 当前回测时间戳，用于获取历史数据
            period: K线周期（如 '30m', '60m'）
            
        返回True表示下降趋势
        - 对于60分钟周期：使用MACD柱子判断，连续3根MACD柱子上涨则认为非下降趋势
        - 对于其他周期：使用均线判断 MA55 < MA89 < MA181 < MA420
        
        Returns:
            tuple: (是否下降趋势, 条件详情字典)
        """
        # 获取指定周期数据
        if period == '60m':
            # 60分钟周期使用MACD柱子判断，只需少量数据
            count = 20
        else:
            # 其他周期使用均线判断，需要420根数据
            count = 420
        
        data = tq.get_market_data(
            field_list=[],
            stock_list=[self.stock_code],
            period=period,
            count=count,
            end_time=timestamp.strftime('%Y%m%d%H%M%S'),
            dividend_type='front'
        )
        
        if not data:
            return False, {}
        
        # 查找收盘价字段
        close_field = None
        for field in ['Close', 'close', 'CLOSE']:
            if field in data:
                close_field = field
                break
        
        if close_field is None:
            return False, {}
        
        close_prices = data[close_field][self.stock_code]
        
        # 60分钟周期使用MACD柱子判断
        if period == '60m':
            # 计算MACD
            macd_df = self.calculate_macd(close_prices)
            macd_hist = macd_df['MACD']
            
            # 检查连续3根MACD柱子是否上涨（柱子值递增）
            if len(macd_hist) >= 3:
                # 获取最近3根柱子
                recent_macd = macd_hist.iloc[-3:]
                
                # 判断是否连续上涨（后一根 > 前一根）
                macd_rising_1 = recent_macd.iloc[1] > recent_macd.iloc[0]
                macd_rising_2 = recent_macd.iloc[2] > recent_macd.iloc[1]
                macd_continuous_rising = macd_rising_1 and macd_rising_2
                
                details = {
                    'macd_bar_1': round(recent_macd.iloc[0], 4),
                    'macd_bar_2': round(recent_macd.iloc[1], 4),
                    'macd_bar_3': round(recent_macd.iloc[2], 4),
                    'bar_1_to_2_rising': macd_rising_1,
                    'bar_2_to_3_rising': macd_rising_2,
                    'continuous_rising': macd_continuous_rising
                }
                
                # 如果连续3根MACD柱子上涨，则认为不是下降趋势
                is_downward = not macd_continuous_rising
            else:
                details = {
                    'error': '数据不足，无法判断'
                }
                is_downward = False
        
        # 其他周期使用均线判断
        else:
            # 计算各周期均线
            ma55 = close_prices.rolling(55).mean()
            ma89 = close_prices.rolling(89).mean()
            ma181 = close_prices.rolling(181).mean()
            ma420 = close_prices.rolling(420).mean()
            
            # 条件1: MA55 < MA89
            condition1 = ma55.iloc[-1] < ma89.iloc[-1]
            
            # 条件2: MA89 < MA181
            condition2 = ma89.iloc[-1] < ma181.iloc[-1]
            
            # 条件3: MA181 < MA420
            condition3 = ma181.iloc[-1] < ma420.iloc[-1]
            
            # 收集条件详情
            details = {
                'ma55': round(ma55.iloc[-1], 2) if len(ma55) > 0 else None,
                'ma89': round(ma89.iloc[-1], 2) if len(ma89) > 0 else None,
                'ma181': round(ma181.iloc[-1], 2) if len(ma181) > 0 else None,
                'ma420': round(ma420.iloc[-1], 2) if len(ma420) > 0 else None,
                'condition1': condition1,  # 55 < 89
                'condition2': condition2,  # 89 < 181
                'condition3': condition3   # 181 < 420
            }
            
            # 所有条件都满足才认为是下降趋势
            is_downward = condition1 and condition2 and condition3
        
        return is_downward, details
    
    def detect_bullish_divergence_at_period(self, timestamp, period) -> bool:
        """
        检测指定周期的底背离信号
        
        Args:
            timestamp: 当前回测时间戳，用于获取历史数据
            period: K线周期（如 '30m', '60m'）
            
        Returns:
            bool: 是否检测到该周期的底背离
        """
        # 获取指定周期数据（需要足够的数据计算MACD和极值点）
        data = tq.get_market_data(
            field_list=[],
            stock_list=[self.stock_code],
            period=period,
            count=100,  # 足够的数据用于检测底背离
            end_time=timestamp.strftime('%Y%m%d%H%M%S'),
            dividend_type='front'
        )
        
        if not data:
            return False
        
        # 查找收盘价字段
        close_field = None
        for field in ['Close', 'close', 'CLOSE']:
            if field in data:
                close_field = field
                break
        
        if close_field is None:
            return False
        
        close_prices = data[close_field][self.stock_code]
        
        # 计算MACD
        macd_df = self.calculate_macd(close_prices)
        
        # 检测底背离（忽略details，只返回布尔值）
        result, _ = self.detect_bullish_divergence(close_prices, macd_df['MACD'])
        return result
    
    def get_share_count(self, price):
        """计算可买入的股票数量"""
        available_funds = self.current_capital * 0.9
        max_shares = int(available_funds / price / 100) * 100
        return max_shares if max_shares >= 100 else 0
    
    def buy(self, timestamp, price, signal_type='底背离'):
        """执行买入"""
        if self.position > 0:
            return False
        
        # 使用90%资金买入
        available_funds = self.current_capital * 0.9
        max_shares = int(available_funds / price / 100) * 100
        
        if max_shares < 100:
            return False
        
        amount = price * max_shares
        self.current_capital -= amount
        self.position = max_shares
        self.position_value = amount
        self.buy_price = price
        self.buy_amount = amount
        self.highest_price = price
        self.lowest_price = price
        self.buy_bar_count = 0  # 重置买入后K线计数
        
        self.trade_id += 1
        trade = TradeRecord(
            trade_id=self.trade_id,
            trade_time=timestamp,
            stock_code=self.stock_code,
            trade_type='BUY',
            price=price,
            volume=max_shares,
            amount=amount,
            signal_type=signal_type
        )
        self.trades.append(trade)
        
        print(f"  [买入] {timestamp} @ {price:.2f} | {max_shares}股")
        
        return True
    
    def sell(self, timestamp, price, signal_type='顶背离'):
        """执行卖出"""
        if self.position <= 0:
            return False
        
        amount = price * self.position
        pnl = amount - self.buy_amount
        pnl_pct = (pnl / self.buy_amount) * 100
        
        self.current_capital += amount
        
        self.trade_id += 1
        trade = TradeRecord(
            trade_id=self.trade_id,
            trade_time=timestamp,
            stock_code=self.stock_code,
            trade_type='SELL',
            price=price,
            volume=self.position,
            amount=amount,
            signal_type=signal_type,
            pnl=pnl
        )
        self.trades.append(trade)
        
        # 统计止损
        if '止损' in signal_type:
            self.stop_loss_count += 1
            if '固定' in signal_type:
                self.stop_loss_types['FIXED'] += 1
            elif '移动' in signal_type:
                self.stop_loss_types['TRAILING'] += 1
            elif '趋势' in signal_type:
                self.stop_loss_types['TREND'] += 1
            elif '时间' in signal_type:
                self.stop_loss_types['TIME'] += 1
        
        print(f"  [卖出] {timestamp} @ {price:.2f} | {signal_type} | 盈亏: {pnl:+.2f} ({pnl_pct:+.2f}%)")
        
        # 清空持仓
        self.position = 0
        self.position_value = 0
        self.buy_price = 0
        self.buy_amount = 0
        self.buy_bar_index = 0
        self.highest_price = 0
        self.lowest_price = 0
        
        return True
    
    def update_equity(self, timestamp, price):
        """更新权益曲线"""
        total_value = self.current_capital + (self.position * price if self.position > 0 else 0)
        self.equity_curve_data.append({
            'timestamp': timestamp,
            'capital': self.current_capital,
            'position_value': self.position_value,
            'position': self.position,
            'total_value': total_value,
            'price': price
        })
    
    def run_backtest(self, start_time, end_time):
        """执行回测"""
        print(f"\n{'='*70}")
        print(f"MACD背离策略回测（带止损）")
        print(f"{'='*70}")
        print(f"股票代码: {self.stock_code}")
        print(f"K线周期: {self.period}")
        print(f"回测区间: {start_time} - {end_time}")
        print(f"初始资金: {self.initial_capital:,.2f} 元")
        print(f"\n止损参数:")
        print(f"  固定止损: {self.stop_loss_pct*100:.1f}%")
        print(f"  移动止损: {self.trailing_stop_pct*100:.1f}%")
        print(f"{'='*70}\n")
        
        # 获取历史数据
        print("正在获取历史数据...")
        data = tq.get_market_data(
            field_list=[],
            stock_list=[self.stock_code],
            period=self.period,
            start_time=start_time,
            end_time=end_time,
            dividend_type='front'
        )
        
        if not data:
            print("获取数据失败：返回数据为空")
            return None
        
        close_field = None
        for field in ['Close', 'close', 'CLOSE']:
            if field in data:
                close_field = field
                break
        
        if close_field is None:
            print(f"获取数据失败：未找到收盘价字段")
            return None
        
        close_prices = data[close_field][self.stock_code]
        
        print(f"获取到 {len(close_prices)} 根K线数据")
        print(f"时间范围: {close_prices.index[0]} 至 {close_prices.index[-1]}")
        print(f"价格范围: {close_prices.min():.2f} - {close_prices.max():.2f}\n")
        
        # 计算MACD
        macd_df = self.calculate_macd(close_prices)
        
        # 开始回测
        print("开始回测...\n")
        
        min_lookback = 50
        
        for i in range(min_lookback, len(close_prices)):
            current_time = close_prices.index[i]
            current_price = close_prices.iloc[i]
            
            # 获取回看数据
            lookback_prices = close_prices.iloc[:i+1]
            lookback_macd = macd_df['MACD'].iloc[:i+1]
            
            # 如果有持仓，先检查止损和卖出信号
            if self.position > 0:
                # 检查止损
                should_stop, stop_type, stop_price = self.check_stop_loss(current_price, current_time)
                
                if should_stop:
                    self.sell(current_time, current_price, stop_type)
                else:
                    # 检查顶背离卖出信号
                    bearish_div = self.detect_bearish_divergence(lookback_prices, lookback_macd)
                    if bearish_div:
                        self.sell(current_time, current_price, '顶背离')
                    else:
                        print(f"  [持有] {current_time} 继续持有（未触发卖出条件）")
            
            else:
                # 无持仓，检查买入信号
                bullish_div, div_details = self.detect_bullish_divergence(lookback_prices, lookback_macd)
                if bullish_div:
                    # 趋势过滤：检查5分钟级别是否处于下降趋势（传入当前时间戳获取历史数据）
                    is_downward, trend_details = self.is_downward_trend(current_time)
                    if is_downward:
                        self.filtered_signals_count += 1
                        print(f"  [过滤] {current_time} 底背离信号被过滤（5分钟趋势向下）")
                        # 二次确认：检查30分钟级别是否存在底背离
                        div_30min = self.detect_bullish_divergence_at_period(current_time, self.confirm_period_30m)
                        if div_30min:
                            print(f"  [过滤] {current_time} 30分钟底背离也被5分钟趋势过滤")
                            # 三次确认：检查60分钟级别是否存在底背离
                            div_60min = self.detect_bullish_divergence_at_period(current_time, self.confirm_period_60m)
                            if div_60min:
                                print(f"  [买入] {current_time} @ {current_price:.2f} | {self.get_share_count(current_price)}股")
                                print(f"        ┌─────────────────────────────────────────────────────────────┐")
                                print(f"        │ 【买入详细分析】")
                                print(f"        │")
                                print(f"        │ ┌─ 15分钟底背离检测结果:")
                                print(f"        │ │   • 第1个低点(索引{div_details.get('price_low_index_1')}): 价格={div_details.get('price_low_1')}, MACD={div_details.get('macd_at_low_1')}")
                                print(f"        │ │   • 第2个低点(索引{div_details.get('price_low_index_2')}): 价格={div_details.get('price_low_2')}, MACD={div_details.get('macd_at_low_2')}")
                                print(f"        │ │   • 价格创更低: {div_details.get('price_lower_low')}")
                                print(f"        │ │   • MACD未创更低: {div_details.get('macd_not_lower_low')}")
                                print(f"        │ │   • 底背离判定: {div_details.get('divergence_detected')}")
                                print(f"        │")
                                print(f"        │ ┌─ 5分钟趋势过滤结果:")
                                print(f"        │ │   • MA55({trend_details.get('ma55')}) < MA89({trend_details.get('ma89')}): {trend_details.get('condition3')}")
                                print(f"        │ │   • MA89({trend_details.get('ma89')}) < MA181({trend_details.get('ma181')}): {trend_details.get('condition2')}")
                                print(f"        │ │   • MA181({trend_details.get('ma181')}) < MA420({trend_details.get('ma420')}): {trend_details.get('condition1')}")
                                print(f"        │ │   • 下降趋势判定: {is_downward} (被过滤)")
                                print(f"        │")
                                print(f"        │ ┌─ 多级别确认:")
                                print(f"        │ │   • 30分钟底背离: 检测到 ✓")
                                print(f"        │ │   • 60分钟底背离: 检测到 ✓")
                                print(f"        │ │   • 触发级联买入条件")
                                print(f"        │")
                                print(f"        │ ┌─ 交易信息:")
                                print(f"        │ │   • 可用资金: {self.current_capital:.2f}")
                                print(f"        │ │   • 买入价格: {current_price:.2f}")
                                print(f"        │ │   • 买入股数: {self.get_share_count(current_price)}")
                                print(f"        │ │   • 买入金额: {(current_price * self.get_share_count(current_price)):.2f}")
                                print(f"        │ │   • 剩余资金: {(self.current_capital - current_price * self.get_share_count(current_price) * 0.9):.2f}")
                                print(f"        │")
                                print(f"        │ 买入信号类型: 底背离(60分钟确认)")
                                print(f"        └─────────────────────────────────────────────────────────────┘")
                                self.buy(current_time, current_price, '底背离(60分钟确认)')
                            else:
                                # 60分钟无底背离，检查60分钟趋势是否非向下
                                is_60min_downward, trend_60min_details = self.is_downward_trend_at_period(current_time, '60m')
                                if not is_60min_downward:
                                    print(f"  [买入] {current_time} @ {current_price:.2f} | {self.get_share_count(current_price)}股")
                                    print(f"        ┌─────────────────────────────────────────────────────────────┐")
                                    print(f"        │ 【买入详细分析】")
                                    print(f"        │")
                                    print(f"        │ ┌─ 15分钟底背离检测结果:")
                                    print(f"        │ │   • 第1个低点(索引{div_details.get('price_low_index_1')}): 价格={div_details.get('price_low_1')}, MACD={div_details.get('macd_at_low_1')}")
                                    print(f"        │ │   • 第2个低点(索引{div_details.get('price_low_index_2')}): 价格={div_details.get('price_low_2')}, MACD={div_details.get('macd_at_low_2')}")
                                    print(f"        │ │   • 价格创更低: {div_details.get('price_lower_low')}")
                                    print(f"        │ │   • MACD未创更低: {div_details.get('macd_not_lower_low')}")
                                    print(f"        │ │   • 底背离判定: {div_details.get('divergence_detected')}")
                                    print(f"        │")
                                    print(f"        │ ┌─ 5分钟趋势过滤结果:")
                                    print(f"        │ │   • MA55({trend_details.get('ma55')}) < MA89({trend_details.get('ma89')}): {trend_details.get('condition3')}")
                                    print(f"        │ │   • MA89({trend_details.get('ma89')}) < MA181({trend_details.get('ma181')}): {trend_details.get('condition2')}")
                                    print(f"        │ │   • MA181({trend_details.get('ma181')}) < MA420({trend_details.get('ma420')}): {trend_details.get('condition1')}")
                                    print(f"        │ │   • 下降趋势判定: {is_downward} (被过滤)")
                                    print(f"        │")
                                    print(f"        │ ┌─ 多级别确认:")
                                    print(f"        │ │   • 30分钟底背离: 检测到 ✓")
                                    print(f"        │ │   • 60分钟底背离: 未检测到")
                                    print(f"        │ │   • 60分钟趋势非向下: {not is_60min_downward}")
                                    print(f"        │ │   • 触发级联买入条件")
                                    print(f"        │")
                                    print(f"        │ ┌─ 60分钟MACD柱子判断:")
                                    print(f"        │ │   • MACD柱子1: {trend_60min_details.get('macd_bar_1')}")
                                    print(f"        │ │   • MACD柱子2: {trend_60min_details.get('macd_bar_2')}")
                                    print(f"        │ │   • MACD柱子3: {trend_60min_details.get('macd_bar_3')}")
                                    print(f"        │ │   • 柱子1→2上涨: {trend_60min_details.get('bar_1_to_2_rising')}")
                                    print(f"        │ │   • 柱子2→3上涨: {trend_60min_details.get('bar_2_to_3_rising')}")
                                    print(f"        │ │   • 连续3根上涨: {trend_60min_details.get('continuous_rising')}")
                                    print(f"        │ │   • 下降趋势判定: {is_60min_downward}")
                                    print(f"        │")
                                    print(f"        │ ┌─ 交易信息:")
                                    print(f"        │ │   • 可用资金: {self.current_capital:.2f}")
                                    print(f"        │ │   • 买入价格: {current_price:.2f}")
                                    print(f"        │ │   • 买入股数: {self.get_share_count(current_price)}")
                                    print(f"        │ │   • 买入金额: {(current_price * self.get_share_count(current_price)):.2f}")
                                    print(f"        │ │   • 剩余资金: {(self.current_capital - current_price * self.get_share_count(current_price) * 0.9):.2f}")
                                    print(f"        │")
                                    print(f"        │ 买入信号类型: 底背离(30分钟确认+60分钟趋势)")
                                    print(f"        └─────────────────────────────────────────────────────────────┘")
                                    self.buy(current_time, current_price, '底背离(30分钟确认+60分钟趋势)')
                                else:
                                    print(f"  [等待] {current_time} 60分钟级别未确认底背离且趋势向下")
                        else:
                            print(f"  [等待] {current_time} 30分钟级别未确认底背离")
                    else:
                        print(f"  [买入] {current_time} @ {current_price:.2f} | {self.get_share_count(current_price)}股")
                        print(f"        ┌─────────────────────────────────────────────────────────────┐")
                        print(f"        │ 【买入详细分析】")
                        print(f"        │")
                        print(f"        │ ┌─ 15分钟底背离检测结果:")
                        print(f"        │ │   • 第1个低点(索引{div_details.get('price_low_index_1')}): 价格={div_details.get('price_low_1')}, MACD={div_details.get('macd_at_low_1')}")
                        print(f"        │ │   • 第2个低点(索引{div_details.get('price_low_index_2')}): 价格={div_details.get('price_low_2')}, MACD={div_details.get('macd_at_low_2')}")
                        print(f"        │ │   • 价格创更低: {div_details.get('price_lower_low')}")
                        print(f"        │ │   • MACD未创更低: {div_details.get('macd_not_lower_low')}")
                        print(f"        │ │   • 底背离判定: {div_details.get('divergence_detected')}")
                        print(f"        │")
                        print(f"        │ ┌─ 5分钟趋势过滤结果:")
                        print(f"        │ │   • MA55({trend_details.get('ma55')}) < MA89({trend_details.get('ma89')}): {trend_details.get('condition3')}")
                        print(f"        │ │   • MA89({trend_details.get('ma89')}) < MA181({trend_details.get('ma181')}): {trend_details.get('condition2')}")
                        print(f"        │ │   • MA181({trend_details.get('ma181')}) < MA420({trend_details.get('ma420')}): {trend_details.get('condition1')}")
                        print(f"        │ │   • 下降趋势判定: {is_downward} (趋势过滤通过)")
                        print(f"        │")
                        print(f"        │ ┌─ 交易信息:")
                        print(f"        │ │   • 可用资金: {self.current_capital:.2f}")
                        print(f"        │ │   • 买入价格: {current_price:.2f}")
                        print(f"        │ │   • 买入股数: {self.get_share_count(current_price)}")
                        print(f"        │ │   • 买入金额: {(current_price * self.get_share_count(current_price)):.2f}")
                        print(f"        │ │   • 剩余资金: {(self.current_capital - current_price * self.get_share_count(current_price) * 0.9):.2f}")
                        print(f"        │")
                        print(f"        │ 买入信号类型: 底背离")
                        print(f"        └─────────────────────────────────────────────────────────────┘")
                        self.buy(current_time, current_price, '底背离')
                else:
                    print(f"  [等待] {current_time} 无买入信号（未检测到15分钟级别底背离）")
            
            # 更新权益曲线
            self.update_equity(current_time, current_price)
        
        # 回测结束，平仓
        if self.position > 0:
            final_price = close_prices.iloc[-1]
            self.sell(close_prices.index[-1], final_price, '回测结束平仓')
        
        print(f"\n{'='*70}")
        print("回测完成")
        print(f"{'='*70}\n")
        
        return self.generate_result(close_prices.index[0], close_prices.index[-1])
    
    def generate_result(self, start_date, end_date):
        """生成回测结果报告"""
        equity_df = pd.DataFrame(self.equity_curve_data)
        equity_df.set_index('timestamp', inplace=True)
        
        total_return = self.current_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        buy_trades = [t for t in self.trades if t.trade_type == 'BUY']
        sell_trades = [t for t in self.trades if t.trade_type == 'SELL']
        
        winning_trades = len([t for t in sell_trades if t.pnl > 0])
        losing_trades = len([t for t in sell_trades if t.pnl <= 0])
        total_trades = len(sell_trades)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        profits = [t.pnl for t in sell_trades if t.pnl > 0]
        losses = [t.pnl for t in sell_trades if t.pnl <= 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        total_profit = sum(profits) if profits else 0
        total_loss = abs(sum(losses)) if losses else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        equity_values = equity_df['total_value'].values
        max_drawdown = 0
        max_drawdown_pct = 0
        peak = equity_values[0]
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = peak - value
            drawdown_pct = (drawdown / peak) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        returns = equity_df['total_value'].pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() * 252 - 0.03) / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        result = BacktestResult(
            stock_code=self.stock_code,
            period=self.period,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.current_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            stop_loss_count=self.stop_loss_count,
            stop_loss_types=self.stop_loss_types,
            filtered_signals_count=self.filtered_signals_count,
            trades=self.trades,
            equity_curve=equity_df
        )
        
        return result
    
    def print_report(self, result: BacktestResult):
        """打印回测报告"""
        print(f"\n{'='*70}")
        print(f"回测报告")
        print(f"{'='*70}\n")
        
        print(f"【基本信息】")
        print(f"  股票代码: {result.stock_code}")
        print(f"  K线周期: {result.period}")
        print(f"  回测区间: {result.start_date} 至 {result.end_date}")
        print()
        
        print(f"【资金情况】")
        print(f"  初始资金: {result.initial_capital:,.2f} 元")
        print(f"  最终资金: {result.final_capital:,.2f} 元")
        print(f"  总盈亏: {result.total_return:,.2f} 元 ({result.total_return_pct:+.2f}%)")
        print()
        
        print(f"【交易统计】")
        print(f"  总交易次数: {result.total_trades} 次")
        print(f"  盈利次数: {result.winning_trades} 次")
        print(f"  亏损次数: {result.losing_trades} 次")
        print(f"  胜率: {result.win_rate:.2f}%")
        print()
        
        print(f"【盈亏分析】")
        print(f"  平均盈利: {result.avg_profit:,.2f} 元")
        print(f"  平均亏损: {result.avg_loss:,.2f} 元")
        print(f"  盈亏比: {result.profit_factor:.2f}")
        print()
        
        print(f"【止损统计】")
        print(f"  止损次数: {result.stop_loss_count} 次")
        if result.stop_loss_count > 0:
            print(f"    - 固定止损: {result.stop_loss_types.get('FIXED', 0)} 次")
            print(f"    - 移动止损: {result.stop_loss_types.get('TRAILING', 0)} 次")
            print(f"    - 趋势止损: {result.stop_loss_types.get('TREND', 0)} 次")
        print()
        
        print(f"【趋势过滤统计】")
        print(f"  被过滤的底背离信号: {result.filtered_signals_count} 次")
        print()
        
        print(f"【风险指标】")
        print(f"  最大回撤: {result.max_drawdown:,.2f} 元 ({result.max_drawdown_pct:.2f}%)")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print()
        
        print(f"【交易明细】")
        print(f"{'-'*90}")
        print(f"{'ID':<4} {'时间':<20} {'类型':<8} {'价格':<10} {'数量':<10} {'信号':<12} {'盈亏':<12}")
        print(f"{'-'*90}")
        
        for trade in result.trades:
            pnl_str = f"{trade.pnl:+.2f}" if trade.trade_type == 'SELL' else "-"
            print(f"{trade.trade_id:<4} {str(trade.trade_time):<20} {trade.trade_type:<8} "
                  f"{trade.price:<10.2f} {trade.volume:<10} {trade.signal_type:<12} {pnl_str:<12}")
        
        print(f"{'-'*90}\n")


def run_backtest():
    """运行回测示例"""
    tq.initialize(__file__)
    
    try:
        # 创建回测实例
        backtest = MACDDivergenceBacktestWithStopLoss(
            stock_code='601360.SH',
            period='15m',
            initial_capital=100000
        )
        
        # 可以调整止损参数
        # backtest.stop_loss_pct = 0.05      # 5%固定止损
        # backtest.trailing_stop_pct = 0.08  # 8%移动止损
        # backtest.time_stop_bars = 30       # 30根K线时间止损
        
        # 执行回测
        result = backtest.run_backtest(
            start_time='20250101',
            end_time='20260509'
        )
        
        if result:
            backtest.print_report(result)
            
    except Exception as e:
        print(f"回测运行异常: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        tq.close()
        print("回测系统已关闭")


if __name__ == '__main__':
    run_backtest()
