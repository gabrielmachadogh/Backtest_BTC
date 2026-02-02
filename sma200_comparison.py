import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

class SMA200Comparison:
    def __init__(self, strategy='original', timeframe='4h'):
        self.strategy = strategy # 'original' ou 'hybrid'
        self.timeframe = timeframe
        self.initial_capital = 10000
        self.capital = 10000
        self.trades = []
        self.daily_equity = []
        
    def download_data(self):
        print(f"[{self.strategy.upper()}] Baixando dados...")
        base_url = "https://data-api.binance.vision/api/v3/klines"
        limit = 1000
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
        
        # Filtra futuro
        df = df[df.index <= pd.Timestamp.now()]
        
        return df[['open', 'high', 'low', 'close']]

    def run(self, df):
        df = df.copy()
        df['SMA200'] = df['close'].rolling(window=200).mean()
        
        self.trades = []
        self.position = None
        self.capital = self.initial_capital
        self.daily_equity = []
        
        buy_trigger = None
        
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        smas = df['SMA200'].values
        dates = df.index
        
        for i in range(200, len(df)):
            idx = dates[i]
            curr_open = opens[i].item()
            curr_high = highs[i].item()
            curr_close = closes[i].item()
            curr_sma = smas[i].item()
            
            # Dados anteriores
            prev_open = opens[i-1].item()
            prev_high = highs[i-1].item()
            prev_sma = smas[i-1].item()
            
            # --- SAÍDA (Igual para ambas: Fechou < Média) ---
            if self.position is not None:
                if curr_close < curr_sma:
                    self._close_trade(idx, curr_close)
                    buy_trigger = None
            
            # --- ENTRADA ---
            else:
                if self.strategy == 'original':
                    # Entra no fechamento se fechar acima
                    if curr_close > curr_sma:
                        self._open_trade(idx, curr_close)
                        
                elif self.strategy == 'hybrid':
                    # Gatilho armado no candle anterior?
                    if buy_trigger is not None:
                        if curr_high > buy_trigger:
                            # Entra no rompimento
                            entry_price = max(curr_open, buy_trigger)
                            self._open_trade(idx, entry_price)
                            buy_trigger = None
                        else:
                            buy_trigger = None # Cancela se não rompeu
                    
                    # Armar novo gatilho
                    # Se candle anterior abriu acima da média (ou fechou, dependendo da interpretação de tendência)
                    # Vamos usar: Se candle anterior fechou acima da média -> arma gatilho na máxima dele
                    # Isso garante que a tendência existe
                    if closes[i-1].item() > smas[i-1].item():
                        buy_trigger = prev_high

            # Equity
            val = (self.position['qty'] * curr_close) if self.position else self.capital
            self.daily_equity.append({'Date': idx, 'Equity': val})

    def _open_trade(self, date, price):
        qty = self.capital / price
        self.position = {'entry_date': date, 'entry_price': price, 'qty': qty}

    def _close_trade(self, date, price):
        val = price * self.position['qty']
        pnl = val - self.capital
        self.capital = val
        self.trades.append({'pnl': pnl})
        self.position = None

    def get_results(self):
        if not self.trades: return None
        
        df_t = pd.DataFrame(self.trades)
        wins = len(df_t[df_t['pnl'] > 0])
        total = len(df_t)
        wr = (wins / total) * 100 if total > 0 else 0
        ret = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        pf = df_t[df_t['pnl'] > 0]['pnl'].sum() / abs(df_t[df_t['pnl'] < 0]['pnl'].sum()) if len(df_t[df_t['pnl'] < 0]) > 0 else 0
        
        # Anual
        df_eq = pd.DataFrame(self.daily_equity)
        df_eq.set_index('Date', inplace=True)
        df_eq['Year'] = df_eq.index.year
        yearly_equity = df_eq.groupby('Year')['Equity'].last()
        
        yearly_ret = {}
        prev = self.initial_capital
        for y in range(2018, 2026):
            if y in yearly_equity.index:
                curr = yearly_equity.loc[y]
                # Se for o primeiro ano, ajusta base se começou no meio
                base = yearly_equity.loc[y-1] if (y-1) in yearly_equity.index else prev
                r = ((curr - base) / base) * 100
                yearly_ret[y] = r
            else:
                yearly_ret[y] = 0.0
                
        return {
            'Trades': total,
            'Win Rate': wr,
            'Return': ret,
            'PF': pf,
            'Yearly': yearly_ret
        }

def main():
    print("="*100)
    print("🚀 COMPARAÇÃO SMA200 (4H): ORIGINAL vs HÍBRIDA 🚀")
    print("="*100)
    
    # Download único
    loader = SMA200Comparison()
    df = loader.download_data()
    
    if df is not None:
        print(f"Dados: {len(df)} candles ({df.index[0]} -> {df.index[-1]})")
        
        # Original
        bt_orig = SMA200Comparison('original')
        bt_orig.run(df)
        res_orig = bt_orig.get_results()
        
        # Híbrida
        bt_hyb = SMA200Comparison('hybrid')
        bt_hyb.run(df)
        res_hyb = bt_hyb.get_results()
        
        print("\n" + "="*100)
        print(f"{'METRICA':<15} | {'ORIGINAL':<15} | {'HÍBRIDA':<15} | {'DIFERENÇA':<15}")
        print("-" * 100)
        
        metrics = [
            ('Trades', 'Trades', '{:.0f}'),
            ('Win Rate', 'Win Rate', '{:.2f}%'),
            ('Profit Factor', 'PF', '{:.2f}'),
            ('Retorno Total', 'Return', '{:.2f}%')
        ]
        
        for label, key, fmt in metrics:
            v1 = res_orig[key]
            v2 = res_hyb[key]
            diff = v2 - v1
            print(f"{label:<15} | {fmt.format(v1):<15} | {fmt.format(v2):<15} | {fmt.format(diff):<15}")
            
        print("-" * 100)
        print("RETORNO ANUAL:")
        print("-" * 100)
        
        for y in range(2018, 2026):
            r1 = res_orig['Yearly'].get(y, 0)
            r2 = res_hyb['Yearly'].get(y, 0)
            print(f"{y:<15} | {r1:>14.2f}% | {r2:>14.2f}%")
            
    print("="*100)

if __name__ == "__main__":
    main()
