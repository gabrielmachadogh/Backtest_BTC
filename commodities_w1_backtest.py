import yfinance as yf
import pandas as pd
import numpy as np
import time

class MA8Backtest:
    def __init__(self, ticker, name, exit_mode='breakout'):
        self.ticker = ticker
        self.name = name
        self.exit_mode = exit_mode
        self.capital = 10000
        self.initial_capital = 10000
        self.trades = []
        self.equity_curve = []
        
    def download_data(self):
        try:
            # Baixa dados semanais
            df = yf.download(self.ticker, period="max", interval="1wk", progress=False, auto_adjust=True)
            if len(df) < 50: return None
            return df[['Open', 'High', 'Low', 'Close']]
        except:
            return None

    def prepare_indicators(self, df):
        df = df.copy()
        df['MA8'] = df['Close'].rolling(window=8).mean()
        
        # Detectar viradas
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
                
                # 2. Saída
                elif self.exit_mode == 'breakout' and sell_trigger is not None:
                    if prev_turn_up:
                        sell_trigger = None
                    elif curr_low <= sell_trigger:
                        exit_price = min(curr_open, sell_trigger)
                        self._close_trade(idx, exit_price, 'Exit Trigger')
                
                elif self.exit_mode == 'simple' and prev_turn_down:
                    self._close_trade(idx, curr_open, 'MA Turn')
                
                # Atualiza gatilho de venda (Breakout)
                elif self.exit_mode == 'breakout' and prev_turn_down:
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

    def _open_trade(self, date, price, stop):
        qty = self.capital / price
        self.position = {'entry_date': date, 'entry_price': price, 'qty': qty, 'stop_price': stop}

    def _close_trade(self, date, price, reason):
        pnl = (price - self.position['entry_price']) * self.position['qty']
        pnl_pct = (price / self.position['entry_price']) - 1
        self.capital += pnl
        self.trades.append({'pnl': pnl, 'pnl_pct': pnl_pct})
        self.position = None

    def get_results(self):
        if not self.trades: return None
        df_t = pd.DataFrame(self.trades)
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        
        return {
            'Ticker': self.name,
            'Mode': self.exit_mode,
            'Trades': len(df_t),
            'Win Rate': (len(wins) / len(df_t)) * 100,
            'Return %': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'Profit Factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0
        }

def main():
    assets = [
        # Crypto
        ('BTC-USD', 'Bitcoin'),
        
        # Metais (Futuros)
        ('GC=F', 'Ouro (GC)'),
        ('SI=F', 'Prata (SI)'),
        ('PL=F', 'Platina (PL)'),
        ('PA=F', 'Paladio (PA)'),
        ('HG=F', 'Cobre (HG)'),
        
        # Energia (Futuros)
        ('CL=F', 'WTI Crude (CL)'),
        ('BZ=F', 'Brent Crude (BZ)'),
        ('NG=F', 'Natural Gas (NG)'),
        ('HO=F', 'Heating Oil (HO)'),
        ('RB=F', 'RBOB Gas (RB)'),
        
        # Grãos e Softs (Futuros)
        ('ZC=F', 'Milho (ZC)'),
        ('ZS=F', 'Soja (ZS)'),
        ('ZW=F', 'Trigo (ZW)'),
        ('SB=F', 'Acucar (SB)'),
        ('KC=F', 'Cafe (KC)'),
        ('CT=F', 'Algodao (CT)'),
        ('CC=F', 'Cacau (CC)'),
        ('OJ=F', 'Laranja (OJ)'),
        
        # Pecuária (Futuros)
        ('LE=F', 'Gado Vivo (LE)'),
        ('HE=F', 'Porco (HE)'),
        ('GF=F', 'Bezerro (GF)'),
        
        # ETFs
        ('GLD', 'ETF Ouro'),
        ('SLV', 'ETF Prata'),
        ('USO', 'ETF WTI'),
        ('BNO', 'ETF Brent'),
        ('UNG', 'ETF Gas'),
        ('DBC', 'ETF Commod.'),
        ('PDBC', 'ETF Commod. Opt')
    ]
    
    print("="*100)
    print("🚀 BACKTEST MASSIVO W1 - SETUP MA8 (LONG ONLY) 🚀")
    print("="*100)
    print(f"Total de ativos: {len(assets)}")
    print("Estratégia: MA8 Virada + Rompimento (Breakout)")
    print("-" * 100)
    
    results = []
    
    for ticker, name in assets:
        print(f"Processando {name}...", end='\r')
        
        # Baixar e preparar
        bt = MA8Backtest(ticker, name)
        df = bt.download_data()
        
        if df is not None:
            df = bt.prepare_indicators(df)
            
            # Testar Modo Breakout (Geralmente melhor em semanal)
            bt.exit_mode = 'breakout'
            bt.run(df)
            res = bt.get_results()
            if res: results.append(res)
            
            # Testar Modo Simple
            bt.exit_mode = 'simple'
            bt.run(df)
            res = bt.get_results()
            if res: results.append(res)
            
        time.sleep(0.5) # Evitar rate limit
        
    # Ordenar por Profit Factor (Descendente)
    results.sort(key=lambda x: x['Profit Factor'], reverse=True)
    
    print("\n" + "="*100)
    print(f"{'ATIVO':<15} | {'MODO':<10} | {'TRADES':<8} | {'WIN RATE':<10} | {'RETORNO':<12} | {'PROFIT FACTOR':<12}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['Ticker']:<15} | {r['Mode']:<10} | {r['Trades']:<8} | {r['Win Rate']:>8.2f}% | {r['Return %']:>10.2f}% | {r['Profit Factor']:>10.2f}")
    
    print("="*100)

if __name__ == "__main__":
    main()
