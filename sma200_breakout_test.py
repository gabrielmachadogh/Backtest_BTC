import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

class SMA200BreakoutBacktest:
    def __init__(self, timeframe='4h'):
        self.timeframe = timeframe
        self.initial_capital = 10000
        self.capital = 10000
        self.trades = []
        self.daily_equity = []
        
    def download_data(self):
        print(f"[{self.timeframe}] Baixando dados da Binance (API Pública)...")
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
        
        # Filtra dados futuros se houver
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
        sell_trigger = None
        
        # Converter para numpy
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        smas = df['SMA200'].values
        dates = df.index
        
        # Início após 200 candles
        for i in range(200, len(df)):
            idx = dates[i]
            curr_open = opens[i].item()
            curr_high = highs[i].item()
            curr_low = lows[i].item()
            curr_close = closes[i].item()
            curr_sma = smas[i].item()
            
            # Dados anteriores (para ver abertura em relação à média)
            # A regra diz: "Se um candle abre acima da média..."
            # Então olhamos o candle atual. Se ele abriu acima, armamos gatilho para romper sua máxima?
            # Ou olhamos o candle anterior? 
            # Interpretação Padrão de Setup de Gatilho:
            # 1. Candle [i-1] abre acima da média.
            # 2. Gatilho de compra = High[i-1].
            # 3. Candle [i] rompe esse gatilho -> Entra.
            
            prev_open = opens[i-1].item()
            prev_high = highs[i-1].item()
            prev_low = lows[i-1].item()
            prev_sma = smas[i-1].item()
            
            # --- SAÍDA ---
            if self.position is not None:
                # 1. Gatilho de Saída (Breakout)
                if sell_trigger is not None:
                    # Se preço perdeu a mínima do gatilho
                    if curr_low <= sell_trigger:
                        exit_price = min(curr_open, sell_trigger)
                        self._close_trade(idx, exit_price, 'Breakout Exit')
                        # Reseta triggers
                        sell_trigger = None
                        buy_trigger = None
                        continue
                        
                    # Se abriu ACIMA da média de novo, cancela gatilho de venda?
                    # Regra: "sair quando abre abaixo e romper"
                    # Se este candle abriu acima, ele invalida a condição de "estar abaixo para armar venda".
                    # Mas se já tinha armado no anterior... geralmente setups de rompimento valem por 1 candle.
                    # Vamos assumir validade de 1 candle.
                    sell_trigger = None 

                # 2. Armar Novo Gatilho de Saída
                # Condição: Candle anterior abriu ABAIXO da média
                if prev_open < prev_sma:
                    sell_trigger = prev_low
            
            # --- ENTRADA ---
            else:
                # 1. Gatilho de Entrada (Breakout)
                if buy_trigger is not None:
                    if curr_high > buy_trigger:
                        entry_price = max(curr_open, buy_trigger)
                        self._open_trade(idx, entry_price)
                        buy_trigger = None
                        continue
                    
                    # Validade de 1 candle
                    buy_trigger = None
                
                # 2. Armar Novo Gatilho de Entrada
                # Condição: Candle anterior abriu ACIMA da média
                if prev_open > prev_sma:
                    buy_trigger = prev_high
            
            # Equity Curve
            val = (self.position['qty'] * curr_close) if self.position else self.capital
            self.daily_equity.append({'Date': idx, 'Equity': val})

    def _open_trade(self, date, price):
        qty = self.capital / price
        self.position = {'entry_date': date, 'entry_price': price, 'qty': qty}

    def _close_trade(self, date, price, reason):
        val = price * self.position['qty']
        pnl = val - self.capital
        pnl_pct = (price / self.position['entry_price']) - 1
        duration = (date - self.position['entry_date']).total_seconds() / 3600
        
        self.capital = val
        self.trades.append({
            'entry_date': self.position['entry_date'],
            'exit_date': date,
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
        
        # Calcular Retorno Anual
        df_eq = pd.DataFrame(self.daily_equity)
        df_eq.set_index('Date', inplace=True)
        df_eq['Year'] = df_eq.index.year
        yearly_equity = df_eq.groupby('Year')['Equity'].last()
        
        # Ajuste base
        start_val = self.initial_capital
        yearly_returns = {}
        
        prev = start_val
        # Garante que 2018 (ano inicial) seja calculado corretamente desde o capital inicial
        # Se o primeiro dado for do meio de 2018, ok.
        
        for y in range(2018, 2026):
            if y in yearly_equity.index:
                curr = yearly_equity.loc[y]
                # Se for o primeiro ano, compara com capital inicial, senão com ano anterior
                base = yearly_equity.loc[y-1] if (y-1) in yearly_equity.index else start_val
                
                ret = ((curr - base) / base) * 100
                yearly_returns[y] = ret
            else:
                yearly_returns[y] = 0.0
        
        return {
            'Timeframe': self.timeframe,
            'Trades': len(df_t),
            'Win Rate': (len(wins) / len(df_t)) * 100,
            'Return %': total_return,
            'Profit Factor': pf,
            'Yearly': yearly_returns
        }

def main():
    print("="*100)
    print("🚀 BTC SMA200 BREAKOUT (4H vs 1H) - ANÁLISE ANUAL 🚀")
    print("="*100)
    print("Lógica: Compra se abre > SMA200 e rompe máxima.")
    print("        Vende se abre < SMA200 e rompe mínima.")
    print("-" * 100)
    
    results = []
    
    for tf in ['4h', '1h']:
        bt = SMA200BreakoutBacktest(timeframe=tf)
        df = bt.download_data()
        
        if df is not None:
            print(f"  [{tf.upper()}] Dados: {len(df)} candles ({df.index[0]} -> {df.index[-1]})")
            bt.run(df)
            res = bt.get_results()
            if res: results.append(res)
            
    print("\n" + "="*100)
    print(f"{'METRICA':<15} | {'4H (Breakout)':<15} | {'1H (Breakout)':<15}")
    print("-" * 100)
    
    if len(results) == 2:
        r4h = results[0]
        r1h = results[1]
        
        print(f"{'Total Trades':<15} | {r4h['Trades']:<15} | {r1h['Trades']:<15}")
        print(f"{'Win Rate':<15} | {r4h['Win Rate']:>14.2f}% | {r1h['Win Rate']:>14.2f}%")
        print(f"{'Profit Factor':<15} | {r4h['Profit Factor']:>15.2f} | {r1h['Profit Factor']:>15.2f}")
        print(f"{'Retorno Total':<15} | {r4h['Return %']:>14.2f}% | {r1h['Return %']:>14.2f}%")
        
        print("-" * 100)
        print("RETORNO ANUAL:")
        print("-" * 100)
        
        for y in range(2018, 2026):
            ret4h = r4h['Yearly'].get(y, 0)
            ret1h = r1h['Yearly'].get(y, 0)
            print(f"{y:<15} | {ret4h:>14.2f}% | {ret1h:>14.2f}%")
            
    print("="*100)

if __name__ == "__main__":
    main()
