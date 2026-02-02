import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class MA8Backtest:
    def __init__(self, exit_mode='simple'):
        """
        exit_mode: 
            'simple' -> Sai no fechamento/abertura quando a média vira para baixo.
            'breakout' -> Sai apenas se romper a mínima do candle que virou para baixo.
        """
        self.ticker = 'BTC-USD'
        self.exit_mode = exit_mode
        self.capital = 10000
        self.initial_capital = 10000
        self.position = None
        self.trades = []
        self.daily_equity = [] # Para calcular retorno anual
        
    def download_data(self):
        print(f"Baixando dados históricos de {self.ticker} via Yahoo Finance...")
        df = yf.download(self.ticker, period="max", interval="1d", progress=False, auto_adjust=True)
        if len(df) == 0: raise Exception("Erro ao baixar dados.")
        print(f"Dados obtidos: {len(df)} dias (De {df.index[0].date()} até {df.index[-1].date()})")
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
        
        # Converter para arrays numpy para performance
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
            
            # Dados anteriores
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
                
                # 4. Atualiza Sell Trigger (apenas modo breakout)
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
            
            # Registra Equity Diário (Mark-to-Market)
            if self.position:
                current_val = self.position['qty'] * curr_close
                self.daily_equity.append({'Date': idx, 'Equity': current_val})
            else:
                self.daily_equity.append({'Date': idx, 'Equity': self.capital})

    def _open_trade(self, date, price, stop):
        qty = self.capital / price
        self.position = {'entry_date': date, 'entry_price': price, 'qty': qty, 'stop_price': stop}

    def _close_trade(self, date, price, reason):
        pnl = (price - self.position['entry_price']) * self.position['qty']
        pnl_pct = (price / self.position['entry_price']) - 1
        duration = (date - self.position['entry_date']).days
        self.capital += pnl
        self.trades.append({
            'entry_date': self.position['entry_date'], 
            'exit_date': date, 
            'pnl': pnl, 
            'pnl_pct': pnl_pct * 100, 
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
        # Ajuste para o primeiro ano (comparar com capital inicial)
        first_year = df_eq.index[0].year
        first_year_ret = (df_eq[df_eq['Year'] == first_year]['Equity'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        yearly_returns.loc[first_year] = first_year_ret
        
        return {
            'Total Trades': len(df_t),
            'Win Rate': (len(wins) / len(df_t)) * 100,
            'Avg Duration': df_t['duration'].mean(),
            'Avg Win %': wins['pnl_pct'].mean() if not wins.empty else 0,
            'Avg Loss %': losses['pnl_pct'].mean() if not losses.empty else 0,
            'Total Return %': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'Profit Factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0,
            'Yearly Returns': yearly_returns
        }

def main():
    print("="*80)
    print("COMPARATIVO MA8 - PERFORMANCE ANUAL")
    print("="*80)
    
    bt_loader = MA8Backtest()
    df = bt_loader.download_data()
    df = bt_loader.prepare_indicators(df)
    
    bt_simple = MA8Backtest(exit_mode='simple')
    bt_simple.run(df)
    res_simple = bt_simple.get_metrics()
    
    bt_breakout = MA8Backtest(exit_mode='breakout')
    bt_breakout.run(df)
    res_breakout = bt_breakout.get_metrics()
    
    # Exibir Métricas Gerais
    print("\n" + "="*80)
    print(f"{'MÉTRICA':<25} | {'SAÍDA 1 (Simples)':<20} | {'SAÍDA 2 (Rompimento)':<20}")
    print("-" * 80)
    metrics = [
        ('Total Trades', 'Total Trades', '{:.0f}'),
        ('Taxa de Acerto', 'Win Rate', '{:.2f}%'),
        ('Profit Factor', 'Profit Factor', '{:.2f}'),
        ('Retorno Total', 'Total Return %', '{:.2f}%')
    ]
    for label, key, fmt in metrics:
        val1 = res_simple[key]
        val2 = res_breakout[key]
        print(f"{label:<25} | {fmt.format(val1):<20} | {fmt.format(val2):<20}")
    
    # Exibir Retorno Anual
    print("\n" + "="*80)
    print(f"{'ANO':<10} | {'SAÍDA 1 (Simples)':<20} | {'SAÍDA 2 (Rompimento)':<20}")
    print("-" * 80)
    
    years = sorted(res_simple['Yearly Returns'].index)
    for year in years:
        ret1 = res_simple['Yearly Returns'].get(year, 0)
        ret2 = res_breakout['Yearly Returns'].get(year, 0)
        
        # Colorir output (opcional, aqui apenas texto)
        r1_str = f"{ret1:+.2f}%"
        r2_str = f"{ret2:+.2f}%"
        print(f"{year:<10} | {r1_str:<20} | {r2_str:<20}")
        
    print("="*80)

if __name__ == "__main__":
    main()
