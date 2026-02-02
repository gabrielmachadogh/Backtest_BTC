import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import requests

class RSIBacktest:
    def __init__(self, timeframe='1d', initial_capital=10000, mode='single'):
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.mode = mode  # 'single' ou 'scale_in'
        self.position = None
        self.trades = []
        self.equity_curve = []
        
        # Parâmetros fixos do setup
        self.rsi_period = 2
        self.rsi_limit = 25
        self.time_stop = 5
        
    def calculate_rsi(self, series, period=2):
        """Calcula IFR (RSI) com suavização de Wilder"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def download_data(self, years=5):
        """Baixa dados da Binance (API Pública)"""
        print(f"[{self.timeframe}] Baixando dados...")
        
        # Mapeamento
        interval_map = {'4h': '4h', '1d': '1d'}
        interval = interval_map.get(self.timeframe, '1d')
        
        base_url = "https://data-api.binance.vision/api/v3/klines"
        limit = 1000
        start_time = int(datetime(2020, 1, 1).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            params = {'symbol': 'BTCUSDT', 'interval': interval, 'startTime': current_start, 'limit': limit}
            try:
                r = requests.get(base_url, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if not data: break
                    all_data.extend(data)
                    current_start = data[-1][0] + 1
                    if len(data) < limit: break
                else:
                    break
            except:
                break
        
        if not all_data: return None
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'w', 'k', 'l'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
        
        return df[['open', 'high', 'low', 'close']]

    def prepare_data(self, df):
        """Calcula indicadores"""
        df['RSI2'] = self.calculate_rsi(df['close'], self.rsi_period)
        # Alvo dinâmico: Máxima dos 2 anteriores
        df['Target_Price'] = df['high'].rolling(window=2).max().shift(1)
        return df

    def run(self, df):
        """Executa backtest"""
        print(f"Executando IFR2 ({self.mode})...")
        
        for i in range(5, len(df)):
            idx = df.index[i]
            row = df.iloc[i]
            
            # --- SAÍDA ---
            if self.position:
                self.position['bars_held'] += 1
                
                # 1. Stop Temporal (5 candles)
                if self.position['bars_held'] >= self.time_stop:
                    self.close_position(idx, row['close'], 'Time Stop')
                    continue
                
                # 2. Alvo (Máxima dos 2 últimos)
                # Verifica se a máxima do dia atingiu o alvo
                target = row['Target_Price']
                if row['high'] >= target:
                    # Executa no alvo ou na abertura se abrir acima (gap)
                    exit_price = max(target, row['open'])
                    self.close_position(idx, exit_price, 'Target Max2')
                    continue
                
                # Scale-in (Segunda entrada)
                if self.mode == 'scale_in' and not self.position['scaled_in']:
                    if row['RSI2'] < self.rsi_limit:
                        # Compra mais 50%
                        price = row['close']
                        cost = self.capital * 0.5
                        qty = cost / price
                        
                        # Preço médio
                        total_qty = self.position['qty'] + qty
                        avg_price = ((self.position['entry_price'] * self.position['qty']) + (price * qty)) / total_qty
                        
                        self.position['qty'] = total_qty
                        self.position['entry_price'] = avg_price
                        self.position['scaled_in'] = True
                        # Nota: Stop temporal conta da primeira entrada
                        
            # --- ENTRADA ---
            elif row['RSI2'] < self.rsi_limit:
                entry_price = row['close']
                
                if self.mode == 'single':
                    qty = self.capital / entry_price
                    scaled = False
                else:
                    qty = (self.capital * 0.5) / entry_price # 50% da mão
                    scaled = False
                
                self.position = {
                    'entry_date': idx,
                    'entry_price': entry_price,
                    'qty': qty,
                    'bars_held': 0,
                    'scaled_in': scaled
                }

    def close_position(self, date, price, reason):
        pnl = (price - self.position['entry_price']) * self.position['qty']
        pnl_pct = (price / self.position['entry_price']) - 1
        
        self.capital += pnl
        self.trades.append({
            'entry_date': self.position['entry_date'],
            'exit_date': date,
            'reason': reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'bars': self.position['bars_held']
        })
        self.position = None

    def get_results(self):
        if not self.trades: return None
        df_t = pd.DataFrame(self.trades)
        
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        
        return {
            'Mode': self.mode,
            'Timeframe': self.timeframe,
            'Trades': len(df_t),
            'Win Rate': len(wins) / len(df_t) * 100,
            'Return': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'Avg PnL': df_t['pnl_pct'].mean(),
            'Max Drawdown': 0, # Simplificado
            'Profit Factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else 999
        }

def main():
    print("="*60)
    print("🚀 BACKTEST IFR2 (RSI 2) - SETUP LARRY CONNORS 🚀")
    print("="*60)
    print("Regras:")
    print("1. Compra: Fechamento se RSI(2) < 25")
    print("2. Alvo: Máxima dos 2 candles anteriores (Dinâmico)")
    print("3. Stop: Tempo (5 candles)")
    print("4. Filtro: Long Only")
    print("-" * 60)

    results = []

    # 1. Testar Diário
    bt_d1 = RSIBacktest(timeframe='1d')
    df_d1 = bt_d1.download_data()
    if df_d1 is not None:
        df_d1 = bt_d1.prepare_data(df_d1)
        
        # Single Entry
        bt_d1.mode = 'single'
        bt_d1.capital = 10000
        bt_d1.trades = []
        bt_d1.run(df_d1)
        results.append(bt_d1.get_results())
        
        # Scale In
        bt_d1.mode = 'scale_in'
        bt_d1.capital = 10000
        bt_d1.trades = []
        bt_d1.run(df_d1)
        results.append(bt_d1.get_results())

    # 2. Testar 4H
    bt_4h = RSIBacktest(timeframe='4h')
    df_4h = bt_4h.download_data()
    if df_4h is not None:
        df_4h = bt_4h.prepare_data(df_4h)
        
        # Single Entry
        bt_4h.mode = 'single'
        bt_4h.capital = 10000
        bt_4h.trades = []
        bt_4h.run(df_4h)
        results.append(bt_4h.get_results())
        
        # Scale In
        bt_4h.mode = 'scale_in'
        bt_4h.capital = 10000
        bt_4h.trades = []
        bt_4h.run(df_4h)
        results.append(bt_4h.get_results())

    # Relatório Final
    print("\n" + "="*80)
    print(f"{'TIMEFRAME':<10} {'MODO':<10} {'TRADES':<8} {'WIN RATE':<10} {'RETORNO':<10} {'P. FACTOR':<10}")
    print("-" * 80)
    
    for r in results:
        if r:
            print(f"{r['Timeframe']:<10} {r['Mode']:<10} {r['Trades']:<8} {r['Win Rate']:<10.2f}% {r['Return']:<10.2f}% {r['Profit Factor']:<10.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
