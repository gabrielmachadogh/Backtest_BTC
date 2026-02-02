import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class MA8Backtest:
    def __init__(self, timeframe='1d', exit_mode='simple'):
        """
        timeframe: '1d' (Diário) ou '1wk' (Semanal)
        exit_mode: 'simple' ou 'breakout'
        """
        self.ticker = 'BTC-USD'
        self.timeframe = timeframe
        self.exit_mode = exit_mode
        self.capital = 10000
        self.initial_capital = 10000
        self.position = None
        self.trades = []
        self.daily_equity = [] 
        
    def download_data(self):
        print(f"[{self.timeframe.upper()}] Baixando dados históricos...")
        df = yf.download(self.ticker, period="max", interval=self.timeframe, progress=False, auto_adjust=True)
        if len(df) == 0: raise Exception("Erro ao baixar dados.")
        
        # Filtra dados vazios
        df.dropna(inplace=True)
        print(f"Dados: {len(df)} candles ({df.index[0].date()} -> {df.index[-1].date()})")
        return df[['Open', 'High', 'Low', 'Close']]

    def prepare_indicators(self, df):
        df = df.copy()
        df['MA8'] = df['Close'].rolling(window=8).mean()
        df['MA_Dir'] = 0
        df.loc[df['MA8'] > df['MA8'].shift(1), 'MA_Dir'] = 1
        df.loc[df['MA8'] < df['MA8'].shift(1), 'MA_Dir'] = -1
        df['Turn_Up'] = (df['MA_Dir'] == 1) & (df['MA_Dir'].shift(1) != 1)
        df['Turn_Down'] = (df['MA_Dir'] == -1) & (df['MA_Dir'].shift(1) != -1)
        return df

    def run(self, df):
        self.trades = []
        self.position = None
        self.capital = self.initial_capital
        self.daily_equity = []
        
        buy_trigger = None
        stop_loss = None
        sell_trigger = None
        
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        dates = df.index
        turn_up = df['Turn_Up'].values
        turn_down = df['Turn_Down'].values
        
        for i in range(10, len(df)):
            idx = dates[i]
            curr_open = float(opens[i])
            curr_high = float(highs[i])
            curr_low = float(lows[i])
            curr_close = float(closes[i])
            
            prev_turn_up = bool(turn_up[i-1])
            prev_turn_down = bool(turn_down[i-1])
            prev_high = float(highs[i-1])
            prev_low = float(lows[i-1])
            
            # --- SAÍDA ---
            if self.position is not None:
                # 1. Stop Loss
                if curr_low <= self.position['stop_price']:
                    exit_price = min(curr_open, self.position['stop_price'])
                    self._close_trade(idx, exit_price, 'Stop Loss')
                    buy_trigger = None 
                
                # 2. Gatilho de Saída (Breakout)
                elif self.exit_mode == 'breakout' and sell_trigger is not None:
                    if prev_turn_up:
                        sell_trigger = None # Desarma
                    elif curr_low <= sell_trigger:
                        exit_price = min(curr_open, sell_trigger)
                        self._close_trade(idx, exit_price, 'Exit Trigger')
                
                # 3. Sinal Simples (Virou p/ Baixo)
                elif prev_turn_down and self.exit_mode == 'simple':
                    self._close_trade(idx, curr_open, 'MA Turn Down')
                
                # 4. Atualiza Sell Trigger
                elif prev_turn_down and self.exit_mode == 'breakout':
                    sell_trigger = prev_low

            # --- ENTRADA ---
            else:
                if buy_trigger is not None:
                    if curr_high > buy_trigger:
                        entry_price = max(curr_open, buy_trigger)
                        self._open_trade(idx, entry_price, stop_loss)
                        buy_trigger = None 
                    else:
                        buy_trigger = None
                        stop_loss = None

                if prev_turn_up:
                    buy_trigger = prev_high
                    stop_loss = prev_low
            
            # Equity Curve
            val = self.position['qty'] * curr_close if self.position else self.capital
            self.daily_equity.append({'Date': idx, 'Equity': val})

    def _open_trade(self, date, price, stop):
        qty = self.capital / price
        self.position = {'entry_date': date, 'entry_price': price, 'qty': qty, 'stop_price': stop}

    def _close_trade(self, date, price, reason):
        pnl = (price - self.position['entry_price']) * self.position['qty']
        duration = (date - self.position['entry_date']).days
        self.capital += pnl
        self.trades.append({
            'entry_date': self.position['entry_date'], 
            'exit_date': date, 
            'pnl': pnl, 
            'reason': reason, 
            'duration': duration
        })
        self.position = None

    def get_metrics(self):
        if not self.trades: return None
        df_t = pd.DataFrame(self.trades)
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        
        # Calcular Retorno Anual
        df_eq = pd.DataFrame(self.daily_equity)
        df_eq.set_index('Date', inplace=True)
        df_eq['Year'] = df_eq.index.year
        yearly_returns = df_eq.groupby('Year')['Equity'].last().pct_change() * 100
        # Ajuste 1o ano
        if not yearly_returns.empty:
            fy = df_eq.index[0].year
            if fy in yearly_returns.index:
                yearly_returns.loc[fy] = (df_eq[df_eq['Year'] == fy]['Equity'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        return {
            'Total Trades': len(df_t),
            'Win Rate': (len(wins) / len(df_t)) * 100,
            'Total Return %': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'Profit Factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0,
            'Yearly Returns': yearly_returns
        }

def run_scenario(timeframe):
    # Carregar dados
    bt_loader = MA8Backtest(timeframe=timeframe)
    df = bt_loader.download_data()
    df = bt_loader.prepare_indicators(df)
    
    # Rodar cenários
    bt_simple = MA8Backtest(timeframe=timeframe, exit_mode='simple')
    bt_simple.run(df)
    res_simple = bt_simple.get_metrics()
    
    bt_breakout = MA8Backtest(timeframe=timeframe, exit_mode='breakout')
    bt_breakout.run(df)
    res_breakout = bt_breakout.get_metrics()
    
    return res_simple, res_breakout

def main():
    print("="*100)
    print("COMPARATIVO MA8 (D1 vs W1) - ANÁLISE COMPLETA")
    print("="*100)
    
    # Rodar D1
    d1_simple, d1_breakout = run_scenario('1d')
    # Rodar W1
    w1_simple, w1_breakout = run_scenario('1wk')
    
    # 1. Tabela Geral
    print("\n" + "="*100)
    print(f"{'METRICA':<20} | {'D1 SIMPLES':<15} | {'D1 ROMPIM.':<15} | {'W1 SIMPLES':<15} | {'W1 ROMPIM.':<15}")
    print("-" * 100)
    
    metrics = ['Total Trades', 'Win Rate', 'Profit Factor', 'Total Return %']
    formats = ['{:.0f}', '{:.2f}%', '{:.2f}', '{:.2f}%']
    
    for m, fmt in zip(metrics, formats):
        v1 = d1_simple[m]
        v2 = d1_breakout[m]
        v3 = w1_simple[m]
        v4 = w1_breakout[m]
        print(f"{m:<20} | {fmt.format(v1):<15} | {fmt.format(v2):<15} | {fmt.format(v3):<15} | {fmt.format(v4):<15}")
    
    # 2. Tabela Anual
    print("\n" + "="*100)
    print("PERFORMANCE ANUAL (CAGR)")
    print("-" * 100)
    print(f"{'ANO':<6} | {'D1 SIMPLES':<18} | {'D1 ROMPIM.':<18} | {'W1 SIMPLES':<18} | {'W1 ROMPIM.':<18}")
    print("-" * 100)
    
    # Unir todos os anos
    all_years = sorted(list(set(d1_simple['Yearly Returns'].index) | set(w1_simple['Yearly Returns'].index)))
    
    for year in all_years:
        r1 = d1_simple['Yearly Returns'].get(year, 0)
        r2 = d1_breakout['Yearly Returns'].get(year, 0)
        r3 = w1_simple['Yearly Returns'].get(year, 0)
        r4 = w1_breakout['Yearly Returns'].get(year, 0)
        
        print(f"{year:<6} | {r1:>17.2f}% | {r2:>17.2f}% | {r3:>17.2f}% | {r4:>17.2f}%")
        
    print("="*100)

if __name__ == "__main__":
    main()
