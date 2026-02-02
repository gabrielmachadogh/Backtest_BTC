import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class MA8BreakoutBacktest:
    def __init__(self, ticker, name):
        self.ticker = ticker
        self.name = name
        self.initial_capital = 10000
        self.capital = 10000
        self.trades = []
        self.daily_equity = [] 
        
    def download_data(self):
        try:
            # Baixa dados semanais
            df = yf.download(self.ticker, start="2013-01-01", interval="1wk", progress=False, auto_adjust=True)
            if len(df) < 50: return None
            # Garantir colunas
            df = df[['Open', 'High', 'Low', 'Close']]
            return df
        except:
            return None

    def run(self, df):
        # Preparar indicadores
        df = df.copy()
        df['MA8'] = df['Close'].rolling(window=8).mean()
        df['Turn_Up'] = (df['MA8'] > df['MA8'].shift(1)) & (df['MA8'].shift(1) <= df['MA8'].shift(2))
        df['Turn_Down'] = (df['MA8'] < df['MA8'].shift(1)) & (df['MA8'].shift(1) >= df['MA8'].shift(2))
        
        self.trades = []
        self.position = None
        self.capital = self.initial_capital
        self.daily_equity = []
        
        buy_trigger = None
        stop_loss = None
        sell_trigger = None
        
        # Converter para numpy para velocidade
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        dates = df.index
        turn_up = df['Turn_Up'].values
        turn_down = df['Turn_Down'].values
        
        # Começa do índice 10 para ter média calculada
        for i in range(10, len(df)):
            idx = dates[i]
            curr_open = float(opens[i])
            curr_high = float(highs[i])
            curr_low = float(lows[i])
            curr_close = float(closes[i])
            
            # Candle anterior (onde o sinal ocorreu)
            prev_turn_up = bool(turn_up[i-1])
            prev_turn_down = bool(turn_down[i-1])
            prev_high = float(highs[i-1])
            prev_low = float(lows[i-1])
            
            # --- SAÍDA (Breakout) ---
            if self.position is not None:
                # 1. Stop Loss Inicial
                if curr_low <= self.position['stop_price']:
                    exit_price = min(curr_open, self.position['stop_price'])
                    self._close_trade(idx, exit_price)
                    buy_trigger = None
                
                # 2. Gatilho de Saída (Breakout da média virada)
                elif sell_trigger is not None:
                    if prev_turn_up:
                        sell_trigger = None # Desarma se virou pra cima
                    elif curr_low <= sell_trigger:
                        exit_price = min(curr_open, sell_trigger)
                        self._close_trade(idx, exit_price)
                
                # 3. Atualiza Gatilho se média virou pra baixo
                elif prev_turn_down:
                    sell_trigger = prev_low

            # --- ENTRADA (Breakout) ---
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
            
            # Registrar Equity (Mark to market no fechamento)
            val = (self.position['qty'] * curr_close) if self.position else self.capital
            self.daily_equity.append({'Date': idx, 'Equity': val})

    def _open_trade(self, date, price, stop):
        qty = self.capital / price
        self.position = {'qty': qty, 'stop_price': stop}

    def _close_trade(self, date, price):
        val = price * self.position['qty']
        self.capital = val
        self.position = None

    def get_yearly_returns(self):
        if not self.daily_equity: return pd.Series()
        
        df_eq = pd.DataFrame(self.daily_equity)
        df_eq.set_index('Date', inplace=True)
        df_eq['Year'] = df_eq.index.year
        
        # Pega o último valor de equity de cada ano
        yearly_equity = df_eq.groupby('Year')['Equity'].last()
        
        # Calcula variação percentual
        # Adiciona o ano inicial (2013) com valor 10000 para calcular o retorno de 2014 corretamente
        start_series = pd.Series([self.initial_capital], index=[yearly_equity.index[0]-1])
        # yearly_equity = pd.concat([start_series, yearly_equity]) # Método antigo
        # yearly_returns = yearly_equity.pct_change().dropna() * 100
        
        # Cálculo manual seguro
        returns = {}
        prev_eq = self.initial_capital
        
        # Filtra para começar de 2014
        years = [y for y in yearly_equity.index if y >= 2014]
        
        # Ajuste para pegar o equity do final de 2013 se existir, senão usa 10000
        try:
            prev_eq = df_eq[df_eq['Year'] == 2013]['Equity'].iloc[-1]
        except:
            prev_eq = 10000

        for year in years:
            curr_eq = yearly_equity.loc[year]
            ret = ((curr_eq - prev_eq) / prev_eq) * 100
            returns[year] = ret
            prev_eq = curr_eq
            
        return pd.Series(returns)

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
    
    print("="*100)
    print("🚀 ANÁLISE ANUAL (2014-2024) - TOP 15 ATIVOS - SETUP MA8 BREAKOUT 🚀")
    print("="*100)
    
    all_returns = pd.DataFrame()
    
    for ticker, name in assets:
        print(f"Processando {name}...", end='\r')
        bt = MA8BreakoutBacktest(ticker, name)
        df = bt.download_data()
        
        if df is not None:
            bt.run(df)
            yearly = bt.get_yearly_returns()
            if not yearly.empty:
                all_returns[name] = yearly
    
    print(" " * 50) # Limpar linha
    
    # Filtrar anos de interesse (2014 a 2024)
    all_returns = all_returns.loc[2014:2024]
    
    # Formatação da Tabela
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:,.2f}%'.format)
    
    # Transpor para ter Ativos nas linhas e Anos nas colunas
    final_table = all_returns.T
    
    # Adicionar Média Anual e Total
    final_table['Média Anual'] = final_table.mean(axis=1)
    
    print(final_table)
    
    print("\n" + "="*100)
    print("RESUMO DE PERFORMANCE:")
    print("-" * 100)
    
    # Melhor ano de cada ativo
    print(f"{'ATIVO':<15} | {'MELHOR ANO':<12} | {'PIOR ANO':<12} | {'MÉDIA ANUAL':<12}")
    for idx, row in final_table.iterrows():
        best_year = row.drop('Média Anual').idxmax()
        best_val = row.drop('Média Anual').max()
        worst_year = row.drop('Média Anual').idxmin()
        worst_val = row.drop('Média Anual').min()
        avg = row['Média Anual']
        
        print(f"{idx:<15} | {best_year} ({best_val:+.0f}%) | {worst_year} ({worst_val:+.0f}%) | {avg:+.2f}%")

    print("="*100)

if __name__ == "__main__":
    main()
