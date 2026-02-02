import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

class BTC4HTrendTest:
    def __init__(self, target_mult, trend_ema=0):
        self.target_mult = target_mult
        self.trend_ema = trend_ema  # 0 (sem filtro), 80, 200
        self.initial_capital = 10000
        self.capital = 10000
        self.trades = []
        
    def download_data(self):
        print(f"Baixando dados 4H da Binance (API Pública)...")
        base_url = "https://data-api.binance.vision/api/v3/klines"
        limit = 1000
        start_time = int(datetime(2020, 1, 1).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            params = {'symbol': 'BTCUSDT', 'interval': '4h', 'startTime': current_start, 'limit': limit}
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
        df['MA8'] = df['close'].rolling(window=8).mean()
        
        # Calcular EMA de filtro se necessário
        if self.trend_ema > 0:
            df['Trend_EMA'] = df['close'].ewm(span=self.trend_ema, adjust=False).mean()
        
        # Lógica de Virada
        ma_diff = df['MA8'].diff()
        df['Turn_Up'] = (ma_diff > 0) & (ma_diff.shift(1) <= 0)
        
        self.trades = []
        self.position = None
        self.capital = self.initial_capital
        
        buy_trigger = None
        stop_loss = None
        
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        turn_up = df['Turn_Up'].values
        
        # Se tem filtro, pega valores, senão cria array dummy
        if self.trend_ema > 0:
            trend_vals = df['Trend_EMA'].values
        else:
            trend_vals = None
            
        dates = df.index
        start_idx = max(20, self.trend_ema) # Começa depois que EMA estiver calculada
        
        for i in range(start_idx, len(df)):
            curr_open = opens[i].item()
            curr_high = highs[i].item()
            curr_low = lows[i].item()
            curr_close = closes[i].item()
            
            prev_turn_up = bool(turn_up[i-1])
            prev_high = highs[i-1].item()
            prev_low = lows[i-1].item()
            prev_close = closes[i-1].item()
            
            # Valor do filtro no candle anterior (sinal)
            trend_val = trend_vals[i-1].item() if trend_vals is not None else 0
            
            # --- SAÍDA ---
            if self.position is not None:
                if curr_low <= self.position['stop_price']:
                    exit_price = min(curr_open, self.position['stop_price'])
                    self._close_trade(exit_price)
                    buy_trigger = None
                    continue
                
                elif curr_high >= self.position['take_profit']:
                    exit_price = max(curr_open, self.position['take_profit'])
                    self._close_trade(exit_price)
                    buy_trigger = None
                    continue

            # --- ENTRADA ---
            else:
                if buy_trigger is not None:
                    if curr_high > buy_trigger:
                        entry_price = max(curr_open, buy_trigger)
                        risk = entry_price - stop_loss
                        take_profit = entry_price + (risk * self.target_mult)
                        
                        self._open_trade(entry_price, stop_loss, take_profit)
                        buy_trigger = None
                    else:
                        buy_trigger = None
                        stop_loss = None

                # Novo Sinal + Filtro de Tendência
                if prev_turn_up:
                    # Verifica Filtro: Preço de fechamento > EMA
                    trend_ok = True
                    if self.trend_ema > 0:
                        if prev_close <= trend_val:
                            trend_ok = False
                    
                    if trend_ok:
                        buy_trigger = prev_high
                        stop_loss = prev_low

    def _open_trade(self, price, stop, tp):
        qty = self.capital / price
        self.position = {'qty': qty, 'stop_price': stop, 'take_profit': tp}

    def _close_trade(self, price):
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
        win_rate = (wins / total) * 100 if total > 0 else 0
        total_ret = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        pf = df_t[df_t['pnl'] > 0]['pnl'].sum() / abs(df_t[df_t['pnl'] < 0]['pnl'].sum()) if len(df_t[df_t['pnl'] < 0]) > 0 else 0
        
        filter_name = f"EMA {self.trend_ema}" if self.trend_ema > 0 else "Sem Filtro"
        return {
            'Filtro': filter_name,
            'Alvo': f"{self.target_mult}x",
            'Trades': total,
            'Win Rate': win_rate,
            'Retorno': total_ret,
            'PF': pf
        }

def main():
    targets = [5.0, 5.5, 6.0] # Focando nos melhores do teste anterior
    filters = [0, 80, 200]
    
    print("="*100)
    print("🚀 BTC 4H - TESTE DE FILTROS DE TENDÊNCIA (EMA 80/200) 🚀")
    print("="*100)
    
    loader = BTC4HTrendTest(0)
    df = loader.download_data()
    
    if df is not None:
        results = []
        for f in filters:
            for t in targets:
                print(f"Testando Alvo {t}x com Filtro EMA {f}...", end='\r')
                bt = BTC4HTrendTest(target_mult=t, trend_ema=f)
                bt.run(df)
                res = bt.get_results()
                if res: results.append(res)
        
        print(" " * 50)
        print(f"{'FILTRO':<12} | {'ALVO':<6} | {'TRADES':<8} | {'WIN RATE':<10} | {'RETORNO':<12} | {'PF':<8}")
        print("-" * 100)
        
        # Ordenar por Retorno
        results.sort(key=lambda x: x['Retorno'], reverse=True)
        
        for r in results:
            print(f"{r['Filtro']:<12} | {r['Alvo']:<6} | {r['Trades']:<8} | {r['Win Rate']:>8.2f}% | {r['Retorno']:>10.2f}% | {r['PF']:>8.2f}")
            
        print("="*100)

if __name__ == "__main__":
    main()
