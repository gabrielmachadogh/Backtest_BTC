import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time

class RSIBacktest:
    def __init__(self, timeframe='1d', initial_capital=10000, mode='single', rsi_limit=25):
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.mode = mode  # 'single' ou 'scale_in'
        self.rsi_limit = rsi_limit
        self.rsi_period = 2
        self.time_stop = 5
        
        self.position = None
        self.trades = []
        
    def calculate_rsi(self, series, period=2):
        """Calcula IFR (RSI) com suavização de Wilder"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def download_data(self):
        """Baixa dados da Binance (API Pública)"""
        print(f"[{self.timeframe}] Baixando dados...")
        
        interval_map = {'4h': '4h', '1d': '1d'}
        interval = interval_map.get(self.timeframe, '1d')
        
        base_url = "https://data-api.binance.vision/api/v3/klines"
        limit = 1000
        start_time = int(datetime(2020, 1, 1).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            params = {'symbol': 'BTCUSDT', 'interval': interval, 'startTime': current_start, 'limit': limit}
            try:
                r = requests.get(base_url, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if not data: break
                    all_data.extend(data)
                    current_start = data[-1][0] + 1
                    if len(data) < limit: break
                    time.sleep(0.1)
                else:
                    break
            except:
                break
        
        if not all_data: return None
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'w', 'k', 'l'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
        
        return df[['open', 'high', 'low', 'close']]

    def prepare_data(self, df):
        """Calcula indicadores"""
        df = df.copy()
        df['RSI2'] = self.calculate_rsi(df['close'], self.rsi_period)
        # Alvo dinâmico: Máxima dos 2 anteriores (Shift 1 para pegar os 2 fechados anteriores ao atual)
        df['Target_Price'] = df['high'].rolling(window=2).max().shift(1)
        return df

    def run(self, df):
        """Executa backtest"""
        self.capital = self.initial_capital
        self.trades = []
        self.position = None
        
        for i in range(5, len(df)):
            idx = df.index[i]
            row = df.iloc[i]
            
            # --- VERIFICA SAÍDAS ---
            if self.position:
                self.position['bars_held'] += 1
                
                # 1. Stop Temporal (5 candles)
                if self.position['bars_held'] >= self.time_stop:
                    self.close_position(idx, row['close'], 'Time Stop')
                    continue
                
                # 2. Alvo (Máxima dos 2 últimos fechados)
                target = row['Target_Price']
                if row['high'] >= target:
                    # Se abriu acima do alvo (gap), sai na abertura, senão sai no alvo
                    exit_price = max(target, row['open'])
                    self.close_position(idx, exit_price, 'Target Max2')
                    continue
                
                # 3. Scale-in (Segunda entrada)
                # Se modo scale_in, não fez a segunda entrada ainda, e RSI deu sinal de novo
                if self.mode == 'scale_in' and not self.position['scaled_in']:
                    if row['RSI2'] < self.rsi_limit:
                        # Usa o caixa restante (que deve ser aprox 50% do capital inicial ajustado)
                        # Mas para simplificar e manter proporção 50/50 do risco:
                        # Vamos assumir que reservamos metade do capital na entrada.
                        
                        price = row['close']
                        # Gasta o "restante" reservado
                        cost = self.position['reserved_cash'] 
                        qty_new = cost / price
                        
                        # Novo Preço Médio
                        total_qty = self.position['qty'] + qty_new
                        total_cost = (self.position['entry_price'] * self.position['qty']) + cost
                        avg_price = total_cost / total_qty
                        
                        self.position['qty'] = total_qty
                        self.position['entry_price'] = avg_price
                        self.position['scaled_in'] = True
                        self.position['reserved_cash'] = 0
                        # Nota: Stop temporal continua contando da primeira entrada para não ficar preso
                        
            # --- VERIFICA ENTRADAS ---
            elif row['RSI2'] < self.rsi_limit:
                entry_price = row['close']
                
                if self.mode == 'single':
                    qty = self.capital / entry_price
                    reserved = 0
                else:
                    # Entra com 50% do capital atual
                    invest_amount = self.capital * 0.5
                    qty = invest_amount / entry_price
                    reserved = self.capital - invest_amount # Guarda o resto para o scale-in ou volta pro caixa se sair
                
                self.position = {
                    'entry_date': idx,
                    'entry_price': entry_price,
                    'qty': qty,
                    'bars_held': 0,
                    'scaled_in': False,
                    'reserved_cash': reserved
                }

    def close_position(self, date, price, reason):
        # Valor de venda da posição
        exit_value = price * self.position['qty']
        
        # Custo da posição (Preço Médio * Qtd)
        cost_value = self.position['entry_price'] * self.position['qty']
        
        pnl = exit_value - cost_value
        pnl_pct = (price / self.position['entry_price']) - 1
        
        # Recupera o caixa reservado se houver (caso scale_in e saiu antes da segunda entrada)
        unused_cash = self.position.get('reserved_cash', 0)
        
        # Capital = Caixa não usado + Valor de Venda
        # Mas na lógica simplificada de backtest iterativo:
        # Capital Anterior - Custo + Venda = Capital Atual
        # Ou simplesmente Capital Antes do Trade + PnL
        self.capital += pnl 
        
        self.trades.append({
            'entry_date': self.position['entry_date'],
            'exit_date': date,
            'reason': reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'bars': self.position['bars_held'],
            'scaled': self.position['scaled_in']
        })
        self.position = None

    def get_results(self):
        if not self.trades: return None
        df_t = pd.DataFrame(self.trades)
        
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        
        return {
            'RSI Limit': self.rsi_limit,
            'Mode': self.mode,
            'Timeframe': self.timeframe,
            'Trades': len(df_t),
            'Win Rate': len(wins) / len(df_t) * 100,
            'Return': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'Profit Factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 999
        }

def main():
    print("="*90)
    print("🚀 BACKTEST IFR2 (RSI 2) - VARIAÇÃO DE LIMITES 🚀")
    print("="*90)
    print("Parâmetros Fixos:")
    print("- Alvo: Máxima dos 2 últimos candles")
    print("- Stop: Temporal (5 candles)")
    print("- Scale In: 50% entrada + 50% recompra")
    print("-" * 90)

    thresholds = [25, 20, 15, 10]
    modes = ['single', 'scale_in']
    timeframes = ['1d', '4h']
    
    results_summary = []

    # Download de dados uma vez para cada TF para ganhar tempo
    # 1. Diário
    bt_loader = RSIBacktest(timeframe='1d')
    df_d1 = bt_loader.download_data()
    if df_d1 is not None:
        df_d1 = bt_loader.prepare_data(df_d1)
        print(f"Dados D1 carregados: {len(df_d1)} candles")
        
        for limit in thresholds:
            for mode in modes:
                bt = RSIBacktest(timeframe='1d', mode=mode, rsi_limit=limit)
                bt.run(df_d1)
                res = bt.get_results()
                if res: results_summary.append(res)

    # 2. 4H
    bt_loader_4h = RSIBacktest(timeframe='4h')
    df_4h = bt_loader_4h.download_data()
    if df_4h is not None:
        df_4h = bt_loader_4h.prepare_data(df_4h)
        print(f"Dados 4H carregados: {len(df_4h)} candles")
        
        for limit in thresholds:
            for mode in modes:
                bt = RSIBacktest(timeframe='4h', mode=mode, rsi_limit=limit)
                bt.run(df_4h)
                res = bt.get_results()
                if res: results_summary.append(res)

    # Relatório
    print("\n" + "="*100)
    print(f"{'TF':<5} {'IFR <':<6} {'MODO':<10} {'TRADES':<8} {'WIN RATE':<10} {'RETORNO':<10} {'P. FACTOR':<10}")
    print("-" * 100)
    
    # Ordenar por TF, depois IFR descendente
    results_summary.sort(key=lambda x: (x['Timeframe'], -x['RSI Limit'], x['Mode']))
    
    for r in results_summary:
        print(f"{r['Timeframe']:<5} {r['RSI Limit']:<6} {r['Mode']:<10} {r['Trades']:<8} {r['Win Rate']:<10.2f}% {r['Return']:<10.2f}% {r['Profit Factor']:<10.2f}")
    print("="*100)

if __name__ == "__main__":
    main()
