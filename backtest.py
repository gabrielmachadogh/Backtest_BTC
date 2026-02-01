import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import requests
from itertools import product
import scipy.stats as stats

class BTCBacktest:
    def __init__(self, timeframe='1d', ma_period=8, initial_capital=10000, 
                 body_pct_min=0, close_position_min=0, candle_size_multiplier=0,
                 take_profit_multiplier=0, exit_first_profit=False, exit_on_ma_turn=False):
        self.timeframe = timeframe
        self.ma_period = ma_period
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        
        self.body_pct_min = body_pct_min
        self.close_position_min = close_position_min
        self.candle_size_multiplier = candle_size_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.exit_first_profit = exit_first_profit
        self.exit_on_ma_turn = exit_on_ma_turn  # NOVO: sair quando MA virar + romper
        
    def download_from_mexc(self, days):
        """Baixa dados da MEXC API"""
        print(f"Baixando dados da MEXC...")
        
        interval_map = {
            '4h': '4h',
            '1d': '1d', 
            '1D': '1d',
            '1wk': '1w',
            '1W': '1w'
        }
        
        interval = interval_map.get(self.timeframe, '1d')
        
        endpoint = 'https://api.mexc.com/api/v3/klines'
        
        all_data = []
        limit = 1000
        
        end_time = int(time.time() * 1000)
        
        if interval == '4h':
            candles_per_day = 6
        elif interval == '1d':
            candles_per_day = 1
        elif interval == '1w':
            candles_per_day = 1/7
        else:
            candles_per_day = 1
        
        total_candles_needed = int(days * candles_per_day)
        iterations = (total_candles_needed // limit) + 1
        
        print(f"  Buscando ~{total_candles_needed} candles em {iterations} requests...")
        
        for i in range(iterations):
            params = {
                'symbol': 'BTCUSDT',
                'interval': interval,
                'limit': limit,
                'endTime': end_time
            }
            
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data and isinstance(data, list):
                    all_data.extend(data)
                    
                    if len(data) < limit:
                        break
                    
                    end_time = data[0][0] - 1
                    time.sleep(0.2)
                else:
                    break
                    
            except Exception as e:
                print(f"  Erro ao baixar chunk {i}: {str(e)}")
                break
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def download_from_cryptocompare(self, days):
        """Baixa dados do CryptoCompare API"""
        print(f"Baixando dados do CryptoCompare...")
        
        if self.timeframe in ['4h', '4H']:
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histohour'
            limit = min(days * 6, 2000)
        elif self.timeframe in ['1d', '1D']:
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
            limit = min(days, 2000)
        elif self.timeframe in ['1wk', '1W']:
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
            limit = min(days // 7, 2000)
        else:
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
            limit = min(days, 2000)
        
        all_data = []
        to_timestamp = int(time.time())
        
        iterations = (limit // 2000) + 1
        
        for i in range(iterations):
            params = {
                'fsym': 'BTC',
                'tsym': 'USD',
                'limit': min(2000, limit - len(all_data)),
                'toTs': to_timestamp
            }
            
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data['Response'] == 'Success':
                    df_chunk = pd.DataFrame(data['Data']['Data'])
                    all_data.append(df_chunk)
                    
                    if len(df_chunk) < 2000:
                        break
                    
                    to_timestamp = int(df_chunk['time'].min()) - 86400
                    time.sleep(1)
                else:
                    break
                    
            except Exception as e:
                print(f"  Erro ao baixar chunk {i}: {str(e)}")
                break
        
        if not all_data:
            return None
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('time')
        df = df.drop_duplicates(subset=['time'])
        
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('Date', inplace=True)
        
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volumefrom': 'Volume'
        })
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        if self.timeframe in ['4h', '4H'] and endpoint == 'histohour':
            df = df.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        if self.timeframe in ['1wk', '1W']:
            df = df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        return df
    
    def download_from_binance(self, days):
        """Baixa dados da API pública da Binance"""
        print(f"Baixando dados da Binance...")
        
        interval_map = {
            '4h': '4h',
            '1d': '1d',
            '1D': '1d',
            '1wk': '1w',
            '1W': '1w'
        }
        
        interval = interval_map.get(self.timeframe, '1d')
        
        endpoint = 'https://api.binance.com/api/v3/klines'
        
        all_data = []
        limit = 1000
        end_time = int(time.time() * 1000)
        
        if interval == '4h':
            candles_needed = days * 6
        elif interval == '1d':
            candles_needed = days
        elif interval == '1w':
            candles_needed = days // 7
        else:
            candles_needed = days
        
        iterations = (candles_needed // limit) + 1
        
        for i in range(iterations):
            params = {
                'symbol': 'BTCUSDT',
                'interval': interval,
                'limit': limit,
                'endTime': end_time
            }
            
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data:
                    all_data.extend(data)
                    
                    if len(data) < limit:
                        break
                    
                    end_time = data[0][0] - 1
                    time.sleep(0.5)
                else:
                    break
                    
            except Exception as e:
                print(f"  Erro ao baixar chunk {i}: {str(e)}")
                break
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.sort_index()
        
        return df
    
    def download_data(self, years=15):
        """Baixa dados históricos do BTC"""
        days = years * 365
        df = None
        
        try:
            df = self.download_from_mexc(days)
            if df is not None and len(df) > 100:
                print(f"✅ MEXC: {len(df)} candles de {df.index[0]} até {df.index[-1]}")
                return df
        except Exception as e:
            print(f"  MEXC falhou: {str(e)}")
        
        try:
            df = self.download_from_binance(days)
            if df is not None and len(df) > 100:
                print(f"✅ Binance: {len(df)} candles de {df.index[0]} até {df.index[-1]}")
                return df
        except Exception as e:
            print(f"  Binance falhou: {str(e)}")
        
        try:
            df = self.download_from_cryptocompare(days)
            if df is not None and len(df) > 100:
                print(f"✅ CryptoCompare: {len(df)} candles de {df.index[0]} até {df.index[-1]}")
                return df
        except Exception as e:
            print(f"  CryptoCompare falhou: {str(e)}")
        
        if self.timeframe in ['1d', '1D', '1wk', '1W']:
            try:
                import yfinance as yf
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                interval_map = {
                    '1d': '1d',
                    '1D': '1d',
                    '1wk': '1wk',
                    '1W': '1wk'
                }
                
                interval = interval_map.get(self.timeframe, '1d')
                
                for ticker in ["BTC-USD", "BTCUSD=X"]:
                    try:
                        data = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            interval=interval,
                            progress=False,
                            auto_adjust=True
                        )
                        
                        if not data.empty and len(data) > 100:
                            df = data
                            print(f"✅ Yahoo Finance: {len(df)} candles")
                            break
                    except:
                        continue
            except Exception as e:
                print(f"  yfinance falhou: {str(e)}")
        
        if df is None or df.empty:
            raise Exception("Não foi possível baixar dados de nenhuma fonte")
        
        return df
    
    def calculate_candle_metrics(self, df):
        """Calcula métricas dos candles para filtros"""
        df['Candle_Range'] = df['High'] - df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Body_Pct'] = (df['Body_Size'] / df['Candle_Range'] * 100).fillna(0)
        df['Close_Position'] = ((df['Close'] - df['Low']) / df['Candle_Range'] * 100).fillna(50)
        df['Avg_Candle_Size'] = df['Candle_Range'].rolling(window=20).mean()
        df['Size_vs_Avg'] = df['Candle_Range'] / df['Avg_Candle_Size']
        
        return df
    
    def calculate_ma(self, df):
        """Calcula média móvel de 8 períodos"""
        df['MA8'] = df['Close'].rolling(window=self.ma_period).mean()
        
        df['MA_Direction'] = 0
        df.loc[df['MA8'] > df['MA8'].shift(1), 'MA_Direction'] = 1
        df.loc[df['MA8'] < df['MA8'].shift(1), 'MA_Direction'] = -1
        
        df['MA_Turn_Up'] = (df['MA_Direction'] == 1) & (df['MA_Direction'].shift(1) != 1)
        df['MA_Turn_Down'] = (df['MA_Direction'] == -1) & (df['MA_Direction'].shift(1) != -1)
        
        return df
    
    def check_entry_filters(self, df, idx):
        """Verifica se o candle passa nos filtros de entrada"""
        candle = df.loc[idx]
        
        if self.body_pct_min > 0:
            if candle['Body_Pct'] < self.body_pct_min:
                return False
        
        if self.close_position_min > 0:
            if candle['Close_Position'] < self.close_position_min:
                return False
        
        if self.candle_size_multiplier > 0:
            if pd.notna(candle['Size_vs_Avg']) and candle['Size_vs_Avg'] < self.candle_size_multiplier:
                return False
        
        return True
    
    def run_backtest(self, df):
        """Executa o backtest com a estratégia"""
        entry_trigger = None
        exit_trigger = None
        stop_loss = None
        take_profit = None
        
        for i in range(self.ma_period + 1, len(df)):
            current_idx = df.index[i]
            current_price = df.loc[current_idx, 'High']
            current_low = df.loc[current_idx, 'Low']
            current_close = df.loc[current_idx, 'Close']
            
            if self.position is None:
                if df.loc[df.index[i-1], 'MA_Turn_Up']:
                    if self.check_entry_filters(df, df.index[i-1]):
                        entry_trigger = df.loc[df.index[i-1], 'High']
                        stop_loss = df.loc[df.index[i-1], 'Low']
                
                if entry_trigger and current_price >= entry_trigger:
                    quantity = self.capital / entry_trigger
                    
                    if self.take_profit_multiplier > 0:
                        risk = entry_trigger - stop_loss
                        take_profit = entry_trigger + (risk * self.take_profit_multiplier)
                    else:
                        take_profit = None
                    
                    self.position = {
                        'entry_date': current_idx,
                        'entry_price': entry_trigger,
                        'quantity': quantity,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    entry_trigger = None
                    
            else:
                # PRIORIDADE 1: First Profit (se ativado)
                if self.exit_first_profit:
                    if current_close > self.position['entry_price']:
                        exit_price = current_close
                        self.close_position(current_idx, exit_price, 'First Profit')
                        exit_trigger = None
                        continue
                
                # PRIORIDADE 2: Take Profit Fixo (se ativado e não tem FirstProfit)
                if not self.exit_first_profit and self.position['take_profit']:
                    if current_price >= self.position['take_profit']:
                        exit_price = self.position['take_profit']
                        self.close_position(current_idx, exit_price, 'Take Profit')
                        exit_trigger = None
                        continue
                
                # PRIORIDADE 3: Stop Loss (sempre ativo)
                if current_low <= self.position['stop_loss']:
                    exit_price = self.position['stop_loss']
                    self.close_position(current_idx, exit_price, 'Stop Loss')
                    exit_trigger = None
                    continue
                
                # PRIORIDADE 4: MA Turn + Rompimento (se ativado, sem TP/FirstProfit)
                if self.exit_on_ma_turn and not self.exit_first_profit and self.take_profit_multiplier == 0:
                    if df.loc[df.index[i-1], 'MA_Turn_Down']:
                        exit_trigger = df.loc[df.index[i-1], 'Low']
                    
                    if df.loc[df.index[i-1], 'MA_Turn_Up'] and exit_trigger:
                        exit_trigger = None
                    
                    if exit_trigger and current_low <= exit_trigger:
                        self.close_position(current_idx, exit_trigger, 'MA Turn')
                        exit_trigger = None
            
            if self.position:
                current_equity = self.position['quantity'] * df.loc[current_idx, 'Close']
            else:
                current_equity = self.capital
            
            self.equity_curve.append({
                'date': current_idx,
                'equity': current_equity
            })
        
        if self.position:
            last_price = df.iloc[-1]['Close']
            self.close_position(df.index[-1], last_price, 'Fim do Backtest')
    
    def close_position(self, exit_date, exit_price, reason):
        """Fecha a posição atual"""
        pnl = (exit_price - self.position['entry_price']) * self.position['quantity']
        pnl_pct = ((exit_price / self.position['entry_price']) - 1) * 100
        
        self.capital += pnl
        
        trade = {
            'entry_date': self.position['entry_date'],
            'entry_price': self.position['entry_price'],
            'exit_date': exit_date,
            'exit_price': exit_price,
            'quantity': self.position['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'capital': self.capital
        }
        
        self.trades.append(trade)
        self.position = None
    
    def calculate_metrics(self):
        """Calcula métricas de performance"""
        if not self.trades:
            return {}, pd.DataFrame(), pd.DataFrame(self.equity_curve)
        
        df_trades = pd.DataFrame(self.trades)
        df_equity = pd.DataFrame(self.equity_curve)
        
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]
        
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        df_equity['cummax'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['cummax']) / df_equity['cummax'] * 100
        max_drawdown = df_equity['drawdown'].min()
        
        df_trades['duration'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.total_seconds() / 3600
        
        returns = df_equity['equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        total_days = (df_equity['date'].max() - df_equity['date'].min()).days
        days_in_market = df_trades['duration'].sum() / 24
        time_in_market = (days_in_market / total_days * 100) if total_days > 0 else 0
        
        n = len(self.trades)
        win_rate = (len(winning_trades) / n) * 100 if n > 0 else 0
        
        if n >= 10:
            z_score = 1.96
            p = win_rate / 100
            margin_error = z_score * np.sqrt((p * (1 - p)) / n) * 100
            ci_lower = max(0, win_rate - margin_error)
            ci_upper = min(100, win_rate + margin_error)
        else:
            ci_lower = 0
            ci_upper = 100
        
        if n >= 10:
            wins = len(winning_trades)
            p_value = stats.binomtest(wins, n, 0.5, alternative='greater').pvalue
            statistically_significant = p_value < 0.05
        else:
            p_value = 1.0
            statistically_significant = False
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_ci_lower': ci_lower,
            'win_rate_ci_upper': ci_upper,
            'p_value': p_value,
            'statistically_significant': statistically_significant,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'avg_win_pct': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss_pct': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': df_trades['pnl'].max(),
            'largest_loss': df_trades['pnl'].min(),
            'largest_win_pct': df_trades['pnl_pct'].max(),
            'largest_loss_pct': df_trades['pnl_pct'].min(),
            'max_drawdown': max_drawdown,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0,
            'avg_trade_duration': df_trades['duration'].mean(),
            'sharpe_ratio': sharpe,
            'time_in_market': time_in_market,
            'expectancy': df_trades['pnl'].mean(),
            'expectancy_pct': df_trades['pnl_pct'].mean()
        }
        
        return metrics, df_trades, df_equity
    
    def print_results(self, metrics, show_full=True):
        """Imprime resultados"""
        if show_full:
            print("\n" + "="*70)
            print("RESULTADOS DO BACKTEST")
            print("="*70)
            print(f"Capital Inicial:      ${metrics['initial_capital']:,.2f}")
            print(f"Capital Final:        ${metrics['final_capital']:,.2f}")
            print(f"Retorno Total:        {metrics['total_return']:.2f}%")
            print(f"Expectância/Trade:    ${metrics['expectancy']:,.2f} ({metrics['expectancy_pct']:.2f}%)")
            
            print(f"\n{'TRADES':-^70}")
            print(f"Total de Trades:      {metrics['total_trades']}")
            print(f"Trades Vencedores:    {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
            print(f"Trades Perdedores:    {metrics['losing_trades']} ({100-metrics['win_rate']:.1f}%)")
            print(f"Win Rate IC 95%:      [{metrics['win_rate_ci_lower']:.1f}%, {metrics['win_rate_ci_upper']:.1f}%]")
            
            if metrics['statistically_significant']:
                print(f"Significância:        ✅ SIM (p={metrics['p_value']:.4f})")
            else:
                print(f"Significância:        ⚠️ NÃO (p={metrics['p_value']:.4f})")
            
            print(f"Duração Média:        {metrics['avg_trade_duration']:.1f} horas")
            print(f"Tempo em Mercado:     {metrics['time_in_market']:.1f}%")
            
            print(f"\n{'GANHOS/PERDAS':-^70}")
            print(f"Ganho Médio:          ${metrics['avg_win']:,.2f} ({metrics['avg_win_pct']:.2f}%)")
            print(f"Perda Média:          ${metrics['avg_loss']:,.2f} ({metrics['avg_loss_pct']:.2f}%)")
            print(f"Maior Ganho:          ${metrics['largest_win']:,.2f} ({metrics['largest_win_pct']:.2f}%)")
            print(f"Maior Perda:          ${metrics['largest_loss']:,.2f} ({metrics['largest_loss_pct']:.2f}%)")
            
            print(f"\n{'MÉTRICAS DE RISCO':-^70}")
            print(f"Max Drawdown:         {metrics['max_drawdown']:.2f}%")
            print(f"Profit Factor:        {metrics['profit_factor']:.2f}")
            print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
            
            print("="*70)
        else:
            sig = "✅" if metrics['statistically_significant'] else "⚠️"
            print(f"T:{metrics['total_trades']:3d} | WR:{metrics['win_rate']:5.1f}% {sig} | "
                  f"Exp:{metrics['expectancy_pct']:6.2f}% | Ret:{metrics['total_return']:8.2f}% | PF:{metrics['profit_factor']:5.2f}")


def massive_4h_optimization(df_data_4h):
    """Otimização MASSIVA para 4H com todos os fatores"""
    print("\n" + "="*80)
    print("🚀 OTIMIZAÇÃO MASSIVA 4H - BUSCANDO A CONFIGURAÇÃO VENCEDORA 🚀")
    print("="*80)
    print("\nFatores testados:")
    print("  ✓ Body %: 0, 20, 30, 40, 50, 60")
    print("  ✓ Candle Size: 0, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5")
    print("  ✓ Take Profit: 0 (sem), 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0")
    print("  ✓ Exit First Profit: Sim/Não")
    print("  ✓ Exit on MA Turn: Sim/Não")
    print()
    
    # Grid expandido
    body_pct_values = [0, 20, 30, 40, 50, 60]
    candle_size_values = [0, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
    take_profit_values = [0, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    exit_first_profit_values = [False, True]
    exit_on_ma_turn_values = [False, True]
    
    results = []
    tested = 0
    valid = 0
    
    total_combinations = (len(body_pct_values) * len(candle_size_values) * 
                         len(take_profit_values) * len(exit_first_profit_values) *
                         len(exit_on_ma_turn_values))
    
    print(f"📊 Total de combinações possíveis: {total_combinations}")
    print(f"⏱️  Tempo estimado: ~{total_combinations * 0.3 / 60:.1f} minutos\n")
    
    for body_pct, size_mult, tp_mult, exit_fp, exit_ma in product(
        body_pct_values, candle_size_values, take_profit_values,
        exit_first_profit_values, exit_on_ma_turn_values):
        
        tested += 1
        
        # Pular combinações inválidas
        if exit_fp and tp_mult > 0:  # FirstProfit e TP fixo não fazem sentido juntos
            continue
        
        if exit_fp and exit_ma:  # FirstProfit e MA turn não fazem sentido juntos
            continue
        
        if tp_mult > 0 and exit_ma:  # TP fixo e MA turn não fazem sentido juntos
            continue
        
        if tested % 100 == 0:
            print(f"  Progresso: {tested}/{total_combinations} ({tested/total_combinations*100:.1f}%) | Válidas: {valid}")
        
        bt = BTCBacktest(
            timeframe='4h',
            ma_period=8,
            initial_capital=10000,
            body_pct_min=body_pct,
            close_position_min=0,
            candle_size_multiplier=size_mult,
            take_profit_multiplier=tp_mult,
            exit_first_profit=exit_fp,
            exit_on_ma_turn=exit_ma
        )
        
        df_test = df_data_4h.copy()
        df_test = bt.calculate_candle_metrics(df_test)
        df_test = bt.calculate_ma(df_test)
        bt.run_backtest(df_test)
        
        metrics, trades, equity = bt.calculate_metrics()
        
        # Critério: mínimo 20 trades para 4H (menos restritivo que 30)
        if metrics['total_trades'] >= 20:
            valid += 1
            
            # Determinar estratégia de saída
            if exit_fp:
                exit_strategy = "FirstProfit"
            elif tp_mult > 0:
                exit_strategy = f"TP{tp_mult}x"
            elif exit_ma:
                exit_strategy = "MA_Turn"
            else:
                exit_strategy = "None"
            
            results.append({
                'body_pct': body_pct,
                'candle_size_mult': size_mult,
                'take_profit_mult': tp_mult,
                'exit_first_profit': exit_fp,
                'exit_on_ma_turn': exit_ma,
                'exit_strategy': exit_strategy,
                **metrics
            })
    
    print(f"\n✅ Otimização concluída!")
    print(f"   Testadas: {tested} combinações")
    print(f"   Válidas: {valid} (com 20+ trades)\n")
    
    if not results:
        print("❌ Nenhuma configuração válida encontrada!")
        return None
    
    df_results = pd.DataFrame(results)
    
    # Salvar
    os.makedirs('results/optimization', exist_ok=True)
    df_results.to_csv('results/optimization/4h_massive_optimization.csv', index=False)
    
    # === RANKING 1: Máximo Retorno ===
    print("="*100)
    print("💰 TOP 20 - MÁXIMO RETORNO ABSOLUTO")
    print("="*100)
    
    df_by_return = df_results.sort_values('total_return', ascending=False)
    print(f"{'#':<3} {'Body%':<6} {'Size':<5} {'Exit':<12} {'Trades':<7} {'WR%':<7} "
          f"{'Exp%':<7} {'Ret%':<10} {'PF':<5} {'DD%':<7} {'Sig':<4}")
    print("-"*100)
    
    for idx, row in df_by_return.head(20).iterrows():
        sig = "✅" if row['statistically_significant'] else "⚠️"
        print(f"{df_by_return.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<6.0f} "
              f"{row['candle_size_mult']:<5.1f} "
              f"{row['exit_strategy']:<12} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<7.1f} "
              f"{row['expectancy_pct']:<7.2f} "
              f"{row['total_return']:<10.2f} "
              f"{row['profit_factor']:<5.2f} "
              f"{row['max_drawdown']:<7.2f} "
              f"{sig:<4}")
    
    # === RANKING 2: Máxima Expectância ===
    print("\n" + "="*100)
    print("📈 TOP 20 - MÁXIMA EXPECTÂNCIA POR TRADE")
    print("="*100)
    
    df_by_exp = df_results.sort_values('expectancy_pct', ascending=False)
    print(f"{'#':<3} {'Body%':<6} {'Size':<5} {'Exit':<12} {'Trades':<7} {'WR%':<7} "
          f"{'Exp%':<7} {'Ret%':<10} {'PF':<5} {'DD%':<7} {'Sig':<4}")
    print("-"*100)
    
    for idx, row in df_by_exp.head(20).iterrows():
        sig = "✅" if row['statistically_significant'] else "⚠️"
        print(f"{df_by_exp.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<6.0f} "
              f"{row['candle_size_mult']:<5.1f} "
              f"{row['exit_strategy']:<12} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<7.1f} "
              f"{row['expectancy_pct']:<7.2f} "
              f"{row['total_return']:<10.2f} "
              f"{row['profit_factor']:<5.2f} "
              f"{row['max_drawdown']:<7.2f} "
              f"{sig:<4}")
    
    # === RANKING 3: Máximo Win Rate ===
    print("\n" + "="*100)
    print("🏆 TOP 20 - MÁXIMO WIN RATE")
    print("="*100)
    
    df_by_wr = df_results.sort_values('win_rate', ascending=False)
    print(f"{'#':<3} {'Body%':<6} {'Size':<5} {'Exit':<12} {'Trades':<7} {'WR%':<7} "
          f"{'Exp%':<7} {'Ret%':<10} {'PF':<5} {'DD%':<7} {'Sig':<4}")
    print("-"*100)
    
    for idx, row in df_by_wr.head(20).iterrows():
        sig = "✅" if row['statistically_significant'] else "⚠️"
        print(f"{df_by_wr.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<6.0f} "
              f"{row['candle_size_mult']:<5.1f} "
              f"{row['exit_strategy']:<12} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<7.1f} "
              f"{row['expectancy_pct']:<7.2f} "
              f"{row['total_return']:<10.2f} "
              f"{row['profit_factor']:<5.2f} "
              f"{row['max_drawdown']:<7.2f} "
              f"{sig:<4}")
    
    # === RANKING 4: Melhor Balanceado (Score Composto) ===
    print("\n" + "="*100)
    print("⚖️  TOP 20 - MELHOR BALANCEADO (Retorno + Expectância + WR)")
    print("="*100)
    
    # Normalizar métricas
    df_results['return_norm'] = (df_results['total_return'] - df_results['total_return'].min()) / (df_results['total_return'].max() - df_results['total_return'].min())
    df_results['exp_norm'] = (df_results['expectancy_pct'] - df_results['expectancy_pct'].min()) / (df_results['expectancy_pct'].max() - df_results['expectancy_pct'].min())
    df_results['wr_norm'] = (df_results['win_rate'] - df_results['win_rate'].min()) / (df_results['win_rate'].max() - df_results['win_rate'].min())
    
    # Score: 40% Retorno + 30% Expectância + 30% WR
    df_results['composite_score'] = (df_results['return_norm'] * 0.4 + 
                                      df_results['exp_norm'] * 0.3 + 
                                      df_results['wr_norm'] * 0.3)
    
    df_balanced = df_results.sort_values('composite_score', ascending=False)
    print(f"{'#':<3} {'Body%':<6} {'Size':<5} {'Exit':<12} {'Trades':<7} {'WR%':<7} "
          f"{'Exp%':<7} {'Ret%':<10} {'PF':<5} {'Score':<6}")
    print("-"*100)
    
    for idx, row in df_balanced.head(20).iterrows():
        print(f"{df_balanced.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<6.0f} "
              f"{row['candle_size_mult']:<5.1f} "
              f"{row['exit_strategy']:<12} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<7.1f} "
              f"{row['expectancy_pct']:<7.2f} "
              f"{row['total_return']:<10.2f} "
              f"{row['profit_factor']:<5.2f} "
              f"{row['composite_score']:<6.3f}")
    
    # Melhor configuração
    best = df_balanced.iloc[0]
    
    print("\n" + "="*70)
    print("🎯 CONFIGURAÇÃO VENCEDORA 4H (Melhor Balanceada)")
    print("="*70)
    print(f"Body % mínimo:          {best['body_pct']:.0f}%")
    print(f"Candle Size multiplier: {best['candle_size_mult']:.1f}x")
    print(f"Estratégia de Saída:    {best['exit_strategy']}")
    
    print(f"\n📈 RESULTADOS:")
    print(f"Win Rate:               {best['win_rate']:.2f}%")
    print(f"Total Trades:           {best['total_trades']:.0f}")
    print(f"Expectância:            {best['expectancy_pct']:.2f}% por trade")
    print(f"Retorno Total:          {best['total_return']:.2f}%")
    print(f"Profit Factor:          {best['profit_factor']:.2f}")
    print(f"Max Drawdown:           {best['max_drawdown']:.2f}%")
    print(f"Score Composto:         {best['composite_score']:.3f}")
    
    if best['statistically_significant']:
        print(f"Significância:          ✅ SIM (p={best['p_value']:.4f})")
    else:
        print(f"Significância:          ⚠️ NÃO (p={best['p_value']:.4f})")
    
    print("="*70)
    
    # Análise por estratégia de saída
    print("\n" + "="*70)
    print("📊 ANÁLISE POR ESTRATÉGIA DE SAÍDA (Médias)")
    print("="*70)
    
    exit_analysis = df_results.groupby('exit_strategy').agg({
        'total_trades': 'mean',
        'win_rate': 'mean',
        'expectancy_pct': 'mean',
        'total_return': 'mean',
        'profit_factor': 'mean'
    }).sort_values('total_return', ascending=False)
    
    print(exit_analysis.to_string())
    
    return best


def main():
    print("="*70)
    print("🚀 BACKTEST 4H - OTIMIZAÇÃO MASSIVA 🚀")
    print("="*70)
    
    print("\nBaixando dados 4H...")
    bt_temp = BTCBacktest(timeframe='4h', ma_period=8, initial_capital=10000)
    
    try:
        df_4h = bt_temp.download_data(years=5)
        print(f"✅ Dados 4H carregados!\n")
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return
    
    # Otimização massiva
    best_4h = massive_4h_optimization(df_4h)
    
    if best_4h is not None:
        print("\n" + "="*70)
        print("🏆 EXECUTANDO BACKTEST FINAL COM CONFIGURAÇÃO VENCEDORA")
        print("="*70)
        
        bt_best = BTCBacktest(
            timeframe='4h',
            ma_period=8,
            initial_capital=10000,
            body_pct_min=best_4h['body_pct'],
            close_position_min=0,
            candle_size_multiplier=best_4h['candle_size_mult'],
            take_profit_multiplier=best_4h['take_profit_mult'],
            exit_first_profit=best_4h['exit_first_profit'],
            exit_on_ma_turn=best_4h['exit_on_ma_turn']
        )
        
        df_4h_best = df_4h.copy()
        df_4h_best = bt_best.calculate_candle_metrics(df_4h_best)
        df_4h_best = bt_best.calculate_ma(df_4h_best)
        bt_best.run_backtest(df_4h_best)
        
        metrics_best, trades_best, equity_best = bt_best.calculate_metrics()
        bt_best.print_results(metrics_best, show_full=True)
    
    print("\n" + "="*70)
    print("✅ OTIMIZAÇÃO 4H CONCLUÍDA!")
    print("="*70)
    print("\nResultados salvos em:")
    print("  • results/optimization/4h_massive_optimization.csv")


if __name__ == "__main__":
    main()
