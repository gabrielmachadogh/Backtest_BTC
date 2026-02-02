import yfinance as yf
import pandas as pd
import numpy as np
import time

class MA8DetailedTest:
    def __init__(self, ticker, name, target_mult):
        self.ticker = ticker
        self.name = name
        self.target_mult = target_mult
        self.initial_capital = 10000
        self.capital = 10000
        self.trades = []
        self.daily_equity = []
        
    def download_data(self):
        try:
            # Baixa dados desde 2013 para ter base para 2014
            df = yf.download(self.ticker, start="2013-01-01", interval="1wk", progress=False, auto_adjust=True)
            if len(df) < 50: return None
            return df[['Open', 'High', 'Low', 'Close']]
        except:
            return None

    def run(self, df):
        df = df.copy()
        df['MA8'] = df['Close'].rolling(window=8).mean()
        
        # Lógica de Virada
        ma_diff = df['MA8'].diff()
        df['Turn_Up'] = (ma_diff > 0) & (ma_diff.shift(1) <= 0)
        
        self.trades = []
        self.position = None
        self.capital = self.initial_capital
        self.daily_equity = []
        
        buy_trigger = None
        stop_loss = None
        take_profit = None
        
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        turn_up = df['Turn_Up'].values
        dates = df.index
        
        for i in range(10, len(df)):
            idx = dates[i]
            curr_open = opens[i].item()
            curr_high = highs[i].item()
            curr_low = lows[i].item()
            curr_close = closes[i].item()
            
            prev_turn_up = bool(turn_up[i-1])
            prev_high = highs[i-1].item()
            prev_low = lows[i-1].item()
            
            # --- SAÍDA ---
            if self.position is not None:
                # 1. Stop Loss
                if curr_low <= self.position['stop_price']:
                    exit_price = min(curr_open, self.position['stop_price'])
                    self._close_trade(idx, exit_price, 'Stop Loss')
                    buy_trigger = None
                
                # 2. Take Profit (Alvo Fixo)
                elif curr_high >= self.position['take_profit']:
                    exit_price = max(curr_open, self.position['take_profit'])
                    self._close_trade(idx, exit_price, 'Target')
                    buy_trigger = None
            
            # --- ENTRADA ---
            else:
                if buy_trigger is not None:
                    if curr_high > buy_trigger:
                        entry_price = max(curr_open, buy_trigger)
                        risk = entry_price - stop_loss
                        take_profit = entry_price + (risk * self.target_mult)
                        
                        self._open_trade(idx, entry_price, stop_loss, take_profit)
                        buy_trigger = None
                    else:
                        buy_trigger = None
                        stop_loss = None

                if prev_turn_up:
                    buy_trigger = prev_high
                    stop_loss = prev_low
            
            # Equity Curve (Mark to market simplificado pelo fechamento)
            val = (self.position['qty'] * curr_close) if self.position else self.capital
            self.daily_equity.append({'Date': idx, 'Equity': val})

    def _open_trade(self, date, price, stop, tp):
        qty = self.capital / price
        self.position = {'qty': qty, 'stop_price': stop, 'take_profit': tp, 'entry_price': price}

    def _close_trade(self, date, price, reason):
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
        
        # Retorno Anual
        df_eq = pd.DataFrame(self.daily_equity)
        df_eq.set_index('Date', inplace=True)
        df_eq['Year'] = df_eq.index.year
        yearly_equity = df_eq.groupby('Year')['Equity'].last()
        
        # Ajuste base 2013
        try:
            base_2013 = df_eq[df_eq['Year'] == 2013]['Equity'].iloc[-1]
        except:
            base_2013 = 10000
            
        returns = {}
        prev = base_2013
        for y in range(2014, 2025): # Até 2024
            if y in yearly_equity.index:
                curr = yearly_equity.loc[y]
                ret = ((curr - prev) / prev) * 100
                returns[y] = ret
                prev = curr
            else:
                returns[y] = 0.0
                
        return {
            'Win Rate': win_rate,
            'Yearly': returns
        }

def main():
    assets = [
        ('GLD', 'ETF Ouro'),
        ('BTC-USD', 'Bitcoin'),
        ('GC=F', 'Ouro Fut'),
        ('SI=F', 'Prata Fut'),
        ('BZ=F', 'Brent'),
        ('GF=F', 'Bezerro'),
        ('PL=F', 'Platina'),
        ('PA=F', 'Paladio'),
        ('DBC', 'ETF Cmdty'),
        ('SLV', 'ETF Prata'),
        ('OJ=F', 'Laranja'),
        ('HG=F', 'Cobre'),
        ('PDBC', 'ETF Cmdt+'),
        ('CL=F', 'WTI'),
        ('CT=F', 'Algodao')
    ]
    
    targets = [4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5]
    
    print("="*100)
    print("🚀 DETALHAMENTO DE ALVOS (WIN RATE & RETORNO ANUAL) 🚀")
    print("="*100)
    
    # Estrutura para armazenar média dos portfólios
    portfolio_stats = {t: {'WR': [], 'Years': {y: [] for y in range(2014, 2025)}} for t in targets}
    
    for ticker, name in assets:
        print(f"Processando {name}...", end='\r')
        
        dummy = MA8DetailedTest(ticker, name, 0)
        df = dummy.download_data()
        
        if df is not None:
            for t in targets:
                bt = MA8DetailedTest(ticker, name, target_mult=t)
                bt.run(df)
                res = bt.get_results()
                
                if res:
                    portfolio_stats[t]['WR'].append(res['Win Rate'])
                    for y, ret in res['Yearly'].items():
                        portfolio_stats[t]['Years'][y].append(ret)
            
        time.sleep(0.1)
    
    print(" " * 50)
    
    print("\nRESULTADOS MÉDIOS DO PORTFÓLIO (Todos os 15 Ativos):")
    print("-" * 100)
    print(f"{'ALVO':<8} | {'WIN RATE':<10} | {'2014':<8} {'2015':<8} {'2016':<8} {'2017':<8} {'2018':<8} {'2019':<8} {'2020':<8} {'2021':<8} {'2022':<8} {'2023':<8} {'2024':<8}")
    print("-" * 100)
    
    for t in targets:
        avg_wr = np.mean(portfolio_stats[t]['WR'])
        
        # Calcular média de retorno do portfólio por ano
        # (Soma dos retornos / 15 ativos) - Retorno médio simples do portfólio equiponderado
        year_str = ""
        for y in range(2014, 2025):
            avg_y_ret = np.mean(portfolio_stats[t]['Years'][y])
            year_str += f"{avg_y_ret:>7.1f}% "
            
        print(f"{t}x      | {avg_wr:>8.2f}% | {year_str}")

    print("="*100)

if __name__ == "__main__":
    main()
