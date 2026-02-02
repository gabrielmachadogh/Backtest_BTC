import yfinance as yf
import pandas as pd
import numpy as np
import time

class MA8TargetTest:
    def __init__(self, ticker, name, target_mult):
        self.ticker = ticker
        self.name = name
        self.target_mult = target_mult
        self.initial_capital = 10000
        self.capital = 10000
        self.trades = []
        
    def download_data(self):
        try:
            df = yf.download(self.ticker, period="max", interval="1wk", progress=False, auto_adjust=True)
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
        
        buy_trigger = None
        stop_loss = None
        take_profit = None
        
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values # Não usado na lógica fixa, mas mantido
        turn_up = df['Turn_Up'].values
        dates = df.index
        
        for i in range(10, len(df)):
            idx = dates[i]
            curr_open = opens[i].item()
            curr_high = highs[i].item()
            curr_low = lows[i].item()
            
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
                    continue
                
                # 2. Take Profit (Alvo Fixo)
                if curr_high >= self.position['take_profit']:
                    exit_price = max(curr_open, self.position['take_profit'])
                    self._close_trade(idx, exit_price, 'Target')
                    buy_trigger = None
                    continue

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

    def _open_trade(self, date, price, stop, tp):
        qty = self.capital / price
        self.position = {'qty': qty, 'stop_price': stop, 'take_profit': tp, 'entry_price': price}

    def _close_trade(self, date, price, reason):
        val = price * self.position['qty']
        self.capital = val
        self.position = None

    def get_return(self):
        return ((self.capital - self.initial_capital) / self.initial_capital) * 100

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
    print("🚀 OTIMIZAÇÃO DE ALVO CORINGA (W1) - MA8 🚀")
    print("="*100)
    
    results = {}
    
    for ticker, name in assets:
        print(f"Processando {name}...", end='\r')
        
        # Download único
        dummy = MA8TargetTest(ticker, name, 0)
        df = dummy.download_data()
        
        if df is not None:
            asset_res = {}
            for t in targets:
                bt = MA8TargetTest(ticker, name, target_mult=t)
                bt.run(df)
                asset_res[f"{t}x"] = bt.get_return()
            results[name] = asset_res
            
        time.sleep(0.2)
    
    print(" " * 50)
    
    # Criar DataFrame
    df_final = pd.DataFrame(results).T
    
    # Adicionar linha de SOMA TOTAL (O fator decisivo)
    df_final.loc['SOMA TOTAL DO PORTFÓLIO'] = df_final.sum()
    
    # Formatação
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:,.2f}%'.format)
    
    print("\nRETORNO POR ALVO:")
    print("-" * 100)
    print(df_final)
    
    # Encontrar o vencedor
    totals = df_final.loc['SOMA TOTAL DO PORTFÓLIO']
    winner_target = totals.idxmax()
    winner_val = totals.max()
    
    print("\n" + "="*100)
    print(f"🏆 O ALVO CORINGA É: {winner_target}")
    print(f"💰 Retorno Total Acumulado (Todos ativos): {winner_val:,.2f}%")
    print("="*100)

if __name__ == "__main__":
    main()
