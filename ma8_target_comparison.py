import yfinance as yf
import pandas as pd
import numpy as np
import time

class MA8MultiExitBacktest:
    def __init__(self, ticker, name, exit_mode='breakout'):
        self.ticker = ticker
        self.name = name
        self.exit_mode = exit_mode
        self.initial_capital = 10000
        self.capital = 10000
        self.trades = []
        
    def download_data(self):
        try:
            # Baixa dados semanais (W1)
            df = yf.download(self.ticker, period="max", interval="1wk", progress=False, auto_adjust=True)
            if len(df) < 50: return None
            return df[['Open', 'High', 'Low', 'Close']]
        except:
            return None

    def run(self, df):
        # Preparar indicadores
        df = df.copy()
        df['MA8'] = df['Close'].rolling(window=8).mean()
        
        # Lógica de Virada
        # Virou pra Cima: Candle atual > anterior E anterior <= antepenultimo
        # Simplificação robusta: MA sobe vs MA cai
        ma_diff = df['MA8'].diff()
        df['Turn_Up'] = (ma_diff > 0) & (ma_diff.shift(1) <= 0)
        df['Turn_Down'] = (ma_diff < 0) & (ma_diff.shift(1) >= 0)
        
        self.trades = []
        self.position = None
        self.capital = self.initial_capital
        
        buy_trigger = None
        stop_loss = None
        sell_trigger = None # Para breakout
        take_profit = None  # Para alvos fixos
        
        # Converter para numpy para velocidade e segurança
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        turn_up = df['Turn_Up'].values
        turn_down = df['Turn_Down'].values
        dates = df.index
        
        # Determinar multiplicador de alvo se for modo fixo
        target_mult = 0
        if self.exit_mode.startswith('fixed_'):
            try:
                target_mult = float(self.exit_mode.split('_')[1])
            except:
                target_mult = 0

        for i in range(10, len(df)):
            idx = dates[i]
            # .item() resolve o DeprecationWarning do NumPy
            curr_open = opens[i].item()
            curr_high = highs[i].item()
            curr_low = lows[i].item()
            curr_close = closes[i].item()
            
            prev_turn_up = bool(turn_up[i-1])
            prev_turn_down = bool(turn_down[i-1])
            prev_high = highs[i-1].item()
            prev_low = lows[i-1].item()
            
            # --- SAÍDA ---
            if self.position is not None:
                # 1. Stop Loss (Prioridade Máxima)
                if curr_low <= self.position['stop_price']:
                    exit_price = min(curr_open, self.position['stop_price'])
                    self._close_trade(idx, exit_price, 'Stop Loss')
                    buy_trigger = None
                    continue
                
                # 2. Take Profit (Modo Fixo)
                if target_mult > 0 and take_profit is not None:
                    if curr_high >= take_profit:
                        exit_price = max(curr_open, take_profit)
                        self._close_trade(idx, exit_price, f'Target {target_mult}x')
                        buy_trigger = None
                        continue
                
                # 3. Saída Breakout (Modo Dinâmico)
                if self.exit_mode == 'breakout':
                    if sell_trigger is not None:
                        if prev_turn_up:
                            sell_trigger = None # Desarma
                        elif curr_low <= sell_trigger:
                            exit_price = min(curr_open, sell_trigger)
                            self._close_trade(idx, exit_price, 'Breakout Exit')
                    
                    if prev_turn_down:
                        sell_trigger = prev_low

            # --- ENTRADA ---
            else:
                if buy_trigger is not None:
                    if curr_high > buy_trigger:
                        entry_price = max(curr_open, buy_trigger)
                        
                        # Definir Stop e Alvo
                        risk = entry_price - stop_loss
                        if target_mult > 0:
                            take_profit = entry_price + (risk * target_mult)
                        else:
                            take_profit = None
                            
                        self._open_trade(idx, entry_price, stop_loss)
                        buy_trigger = None
                    else:
                        # Cancela se não ativou no candle seguinte
                        buy_trigger = None
                        stop_loss = None

                # Novo Sinal
                if prev_turn_up:
                    buy_trigger = prev_high
                    stop_loss = prev_low

    def _open_trade(self, date, price, stop):
        qty = self.capital / price
        self.position = {'qty': qty, 'stop_price': stop}

    def _close_trade(self, date, price, reason):
        val = price * self.position['qty']
        self.capital = val
        self.trades.append({'pnl_pct': (price / self.position.get('entry_price', price)) - 1}) # Simplificado p/ velocidade
        self.position = None

    def get_results(self):
        if not self.trades: return None
        return {
            'Ticker': self.name,
            'Mode': self.exit_mode,
            'Return %': ((self.capital - self.initial_capital) / self.initial_capital) * 100
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
    
    modes = ['breakout', 'fixed_3.0', 'fixed_3.5', 'fixed_3.75', 'fixed_4.0', 'fixed_4.5']
    
    print("="*120)
    print("🚀 COMPARAÇÃO DE SAÍDAS: BREAKOUT vs ALVOS FIXOS (W1) 🚀")
    print("="*120)
    
    # Dicionário para agregar resultados
    # Chave: Ativo, Valor: Dict com retornos de cada modo
    final_data = {}
    
    for ticker, name in assets:
        print(f"Processando {name}...", end='\r')
        
        # Baixar dados uma vez por ativo
        loader = MA8MultiExitBacktest(ticker, name)
        df = loader.download_data()
        
        if df is not None:
            asset_results = {}
            for mode in modes:
                bt = MA8MultiExitBacktest(ticker, name, exit_mode=mode)
                bt.run(df)
                res = bt.get_results()
                asset_results[mode] = res['Return %'] if res else 0.0
            
            final_data[name] = asset_results
            
        time.sleep(0.2)
        
    print(" " * 50)
    
    # Criar DataFrame para exibição
    df_results = pd.DataFrame(final_data).T
    
    # Ordenar colunas para manter a lógica
    df_results = df_results[modes]
    
    # Formatação
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:,.2f}%'.format)
    
    print("\nRETORNO TOTAL ACUMULADO POR ESTRATÉGIA:")
    print("-" * 120)
    print(df_results)
    
    print("\n" + "="*120)
    print("VENCEDOR POR ATIVO:")
    print("-" * 120)
    
    wins_count = {m: 0 for m in modes}
    
    for asset in df_results.index:
        best_mode = df_results.loc[asset].idxmax()
        best_val = df_results.loc[asset].max()
        breakout_val = df_results.loc[asset]['breakout']
        
        diff = best_val - breakout_val
        status = "✅ Breakout Venceu" if best_mode == 'breakout' else f"⚠️ Melhor: {best_mode} (+{diff:.2f}%)"
        
        print(f"{asset:<15} | {status}")
        wins_count[best_mode] += 1
        
    print("-" * 120)
    print("PLACAR FINAL (Nº de Ativos onde foi o melhor):")
    for mode, count in wins_count.items():
        print(f"{mode:<15}: {count}")
    print("="*120)

if __name__ == "__main__":
    main()
