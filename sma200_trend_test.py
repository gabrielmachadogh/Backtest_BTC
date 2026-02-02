import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

class SMA200Backtest:
    def __init__(self, timeframe='4h'):
        self.timeframe = timeframe
        self.initial_capital = 10000
        self.capital = 10000
        self.trades = []
        self.equity_curve = []
        
    def download_data(self):
        print(f"[{self.timeframe}] Baixando dados da Binance (API Pública)...")
        base_url = "https://data-api.binance.vision/api/v3/klines"
        limit = 1000
        # Desde 2018 para ter bastante histórico
        start_time = int(datetime(2018, 1, 1).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            params = {'symbol': 'BTCUSDT', 'interval': self.timeframe, 'startTime': current_start, 'limit': limit}
            try:
                r = requests.get(base_url, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if not data: break
                    all_data.extend(data)
                    current_start = data[-1][0] + 1
                    if len(data) < limit: break
                else: break
            except: break
            time.sleep(0.1)
            
        if not all_data: return None
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'w', 'k', 'l'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
        
        return df[['open', 'high', 'low', 'close']]

    def run(self, df):
        df = df.copy()
        df['SMA200'] = df['close'].rolling(window=200).mean()
        
        self.trades = []
        self.position = None
        self.capital = self.initial_capital
        
        closes = df['close'].values
        smas = df['SMA200'].values
        dates = df.index
        
        # Começa após 200 candles
        for i in range(200, len(df)):
            idx = dates[i]
            curr_close = closes[i]
            curr_sma = smas[i]
            
            # --- Lógica de Posição ---
            
            # Se já estamos comprados
            if self.position is not None:
                # Sinal de Saída: Fechou ABAIXO da média
                if curr_close < curr_sma:
                    self._close_trade(idx, curr_close, 'Close < SMA200')
            
            # Se estamos fora
            else:
                # Sinal de Entrada: Fechou ACIMA da média
                if curr_close > curr_sma:
                    self._open_trade(idx, curr_close)
            
            # Equity Curve (Valor de mercado no fechamento)
            val = (self.position['qty'] * curr_close) if self.position else self.capital
            self.equity_curve.append({'Date': idx, 'Equity': val})

    def _open_trade(self, date, price):
        qty = self.capital / price
        self.position = {'entry_date': date, 'entry_price': price, 'qty': qty}

    def _close_trade(self, date, price, reason):
        val = price * self.position['qty']
        pnl = val - self.capital
        pnl_pct = (price / self.position['entry_price']) - 1
        duration = (date - self.position['entry_date']).total_seconds() / 3600 # Horas
        
        self.capital = val
        self.trades.append({
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration': duration
        })
        self.position = None

    def get_results(self):
        if not self.trades: return None
        
        df_t = pd.DataFrame(self.trades)
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 999
        
        # Calcular Drawdown
        df_eq = pd.DataFrame(self.equity_curve)
        df_eq['cummax'] = df_eq['Equity'].cummax()
        df_eq['dd'] = (df_eq['Equity'] - df_eq['cummax']) / df_eq['cummax']
        max_dd = df_eq['dd'].min() * 100
        
        # Buy & Hold para comparação
        start_price = self.equity_curve[0]['Equity'] if self.equity_curve else 10000 # Preço inicial do BTC no periodo
        # Ajuste para comparar retorno do ativo, não da equity inicial que é caixa
        # Mas aqui comparamos o retorno da estratégia
        
        return {
            'Timeframe': self.timeframe,
            'Trades': len(df_t),
            'Win Rate': (len(wins) / len(df_t)) * 100,
            'Avg Duration': df_t['duration'].mean(),
            'Return %': total_return,
            'Profit Factor': pf,
            'Max Drawdown': max_dd
        }

def main():
    print("="*100)
    print("🚀 BTC TREND FOLLOWING (SMA 200) - 4H vs 1H 🚀")
    print("="*100)
    print("Regra: Compra Close > SMA200 | Vende Close < SMA200")
    print("-" * 100)
    
    results = []
    
    for tf in ['4h', '1h']:
        bt = SMA200Backtest(timeframe=tf)
        df = bt.download_data()
        
        if df is not None:
            print(f"  Dados: {len(df)} candles ({df.index[0]} -> {df.index[-1]})")
            bt.run(df)
            res = bt.get_results()
            if res: results.append(res)
            
    print("\n" + "="*100)
    print(f"{'TF':<6} | {'TRADES':<8} | {'WIN RATE':<10} | {'RETORNO':<12} | {'P. FACTOR':<10} | {'MAX DD':<10} | {'DURAÇÃO (h)':<12}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['Timeframe']:<6} | {r['Trades']:<8} | {r['Win Rate']:>8.2f}% | {r['Return %']:>10.2f}% | {r['Profit Factor']:>8.2f} | {r['Max Drawdown']:>8.2f}% | {r['Avg Duration']:>10.1f}")
        
    print("="*100)

if __name__ == "__main__":
    main()
