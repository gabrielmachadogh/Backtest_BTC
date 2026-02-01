import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import requests
from itertools import product
import scipy.stats as stats

class BTCBacktest:
    def __init__(self, timeframe='1d', ma_period=8, initial_capital=10000, 
                 body_pct_min=0, close_position_min=0, candle_size_multiplier=0,
                 take_profit_multiplier=0, exit_first_profit=False, exit_on_ma_turn=False):
        self.timeframe = timeframe
        self.ma_period = ma_period
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        
        self.body_pct_min = body_pct_min
        self.close_position_min = close_position_min
        self.candle_size_multiplier = candle_size_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.exit_first_profit = exit_first_profit
        self.exit_on_ma_turn = exit_on_ma_turn
    
    def download_from_binance_requests(self):
        """Download Binance via requests (contorna bloqueio)"""
        print(f"Baixando Binance via requests (4h desde 2020)...")
        
        try:
            # URL base
            base_url = "https://data-api.binance.vision/api/v3/klines"
            
            # Parâmetros
            symbol = "BTCUSDT"
            interval = "4h"
            limit = 1000
            
            # Desde 01/01/2020
            start_time = int(datetime(2020, 1, 1).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)
            
            all_data = []
            current_start = start_time
            
            print(f"  Período: 2020-01-01 até agora")
            print(f"  Buscando em lotes de {limit} candles...")
            
            iterations = 0
            max_iterations = 100  # Segurança
            
            while current_start < end_time and iterations < max_iterations:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'limit': limit
                }
                
                try:
                    response = requests.get(base_url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if not data or len(data) == 0:
                            break
                        
                        all_data.extend(data)
                        
                        # Próximo lote: começar após o último timestamp
                        current_start = data[-1][0] + 1
                        
                        iterations += 1
                        
                        if iterations % 10 == 0:
                            print(f"    Baixados {len(all_data)} candles...")
                        
                        time.sleep(0.1)  # Rate limiting
                        
                        if len(data) < limit:
                            break
                    else:
                        print(f"  Erro HTTP {response.status_code}")
                        break
                        
                except Exception as e:
                    print(f"  Erro na request: {str(e)}")
                    break
            
            if not all_data:
                return None
            
            print(f"  Total baixado: {len(all_data)} candles")
            
            # Converter para DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Converter tipos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Renomear colunas
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Filtrar apenas passado
            now = pd.Timestamp.now()
            df = df[df.index <= now]
            
            # Desde 2020
            min_date = pd.Timestamp('2020-01-01')
            df = df[df.index >= min_date]
            
            print(f"  Período final: {df.index[0]} até {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"  Binance requests falhou: {str(e)}")
            return None
    
    def download_data(self, years=5):
        """Baixa dados históricos"""
        df = None
        
        print(f"Baixando dados 4H desde 2020...")
        
        # Binance via requests (mais confiável)
        try:
            df = self.download_from_binance_requests()
            if df is not None and len(df) > 100:
                return df
        except Exception as e:
            print(f"  Erro: {str(e)}")
        
        if df is None or df.empty:
            raise Exception("Não foi possível baixar dados")
        
        return df
    
    def calculate_candle_metrics(self, df):
        """Calcula métricas"""
        df['Candle_Range'] = df['High'] - df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Body_Pct'] = (df['Body_Size'] / df['Candle_Range'] * 100).fillna(0)
        df['Close_Position'] = ((df['Close'] - df['Low']) / df['Candle_Range'] * 100).fillna(50)
        df['Avg_Candle_Size'] = df['Candle_Range'].rolling(window=20).mean()
        df['Size_vs_Avg'] = df['Candle_Range'] / df['Avg_Candle_Size']
        
        return df
    
    def calculate_ma(self, df):
        """Calcula MA"""
        df['MA8'] = df['Close'].rolling(window=self.ma_period).mean()
        
        df['MA_Direction'] = 0
        df.loc[df['MA8'] > df['MA8'].shift(1), 'MA_Direction'] = 1
        df.loc[df['MA8'] < df['MA8'].shift(1), 'MA_Direction'] = -1
        
        df['MA_Turn_Up'] = (df['MA_Direction'] == 1) & (df['MA_Direction'].shift(1) != 1)
        df['MA_Turn_Down'] = (df['MA_Direction'] == -1) & (df['MA_Direction'].shift(1) != -1)
        
        return df
    
    def check_entry_filters(self, df, idx):
        """Verifica filtros"""
        candle = df.loc[idx]
        
        if self.body_pct_min > 0:
            if candle['Body_Pct'] < self.body_pct_min:
                return False
        
        if self.close_position_min > 0:
            if candle['Close_Position'] < self.close_position_min:
                return False
        
        if self.candle_size_multiplier > 0:
            if pd.notna(candle['Size_vs_Avg']) and candle['Size_vs_Avg'] < self.candle_size_multiplier:
                return False
        
        return True
    
    def run_backtest(self, df):
        """Executa backtest"""
        entry_trigger = None
        exit_trigger = None
        stop_loss = None
        take_profit = None
        
        for i in range(self.ma_period + 1, len(df)):
            current_idx = df.index[i]
            current_price = df.loc[current_idx, 'High']
            current_low = df.loc[current_idx, 'Low']
            current_close = df.loc[current_idx, 'Close']
            
            if self.position is None:
                if df.loc[df.index[i-1], 'MA_Turn_Up']:
                    if self.check_entry_filters(df, df.index[i-1]):
                        entry_trigger = df.loc[df.index[i-1], 'High']
                        stop_loss = df.loc[df.index[i-1], 'Low']
                
                if entry_trigger and current_price >= entry_trigger:
                    quantity = self.capital / entry_trigger
                    
                    if self.take_profit_multiplier > 0:
                        risk = entry_trigger - stop_loss
                        take_profit = entry_trigger + (risk * self.take_profit_multiplier)
                    else:
                        take_profit = None
                    
                    self.position = {
                        'entry_date': current_idx,
                        'entry_price': entry_trigger,
                        'quantity': quantity,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    entry_trigger = None
                    
            else:
                if self.exit_first_profit:
                    if current_close > self.position['entry_price']:
                        exit_price = current_close
                        self.close_position(current_idx, exit_price, 'First Profit')
                        exit_trigger = None
                        continue
                
                if not self.exit_first_profit and self.position['take_profit']:
                    if current_price >= self.position['take_profit']:
                        exit_price = self.position['take_profit']
                        self.close_position(current_idx, exit_price, 'Take Profit')
                        exit_trigger = None
                        continue
                
                if current_low <= self.position['stop_loss']:
                    exit_price = self.position['stop_loss']
                    self.close_position(current_idx, exit_price, 'Stop Loss')
                    exit_trigger = None
                    continue
                
                if self.exit_on_ma_turn and not self.exit_first_profit and self.take_profit_multiplier == 0:
                    if df.loc[df.index[i-1], 'MA_Turn_Down']:
                        exit_trigger = df.loc[df.index[i-1], 'Low']
                    
                    if df.loc[df.index[i-1], 'MA_Turn_Up'] and exit_trigger:
                        exit_trigger = None
                    
                    if exit_trigger and current_low <= exit_trigger:
                        self.close_position(current_idx, exit_trigger, 'MA Turn')
                        exit_trigger = None
            
            if self.position:
                current_equity = self.position['quantity'] * df.loc[current_idx, 'Close']
            else:
                current_equity = self.capital
            
            self.equity_curve.append({
                'date': current_idx,
                'equity': current_equity
            })
        
        if self.position:
            last_price = df.iloc[-1]['Close']
            self.close_position(df.index[-1], last_price, 'Fim')
    
    def close_position(self, exit_date, exit_price, reason):
        """Fecha posição"""
        pnl = (exit_price - self.position['entry_price']) * self.position['quantity']
        pnl_pct = ((exit_price / self.position['entry_price']) - 1) * 100
        
        self.capital += pnl
        
        trade = {
            'entry_date': self.position['entry_date'],
            'entry_price': self.position['entry_price'],
            'exit_date': exit_date,
            'exit_price': exit_price,
            'quantity': self.position['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'capital': self.capital
        }
        
        self.trades.append(trade)
        self.position = None
    
    def calculate_metrics(self):
        """Calcula métricas"""
        if not self.trades:
            return {}, pd.DataFrame(), pd.DataFrame(self.equity_curve)
        
        df_trades = pd.DataFrame(self.trades)
        df_equity = pd.DataFrame(self.equity_curve)
        
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]
        
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        df_equity['cummax'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['cummax']) / df_equity['cummax'] * 100
        max_drawdown = df_equity['drawdown'].min()
        
        df_trades['duration'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.total_seconds() / 3600
        
        returns = df_equity['equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        total_days = (df_equity['date'].max() - df_equity['date'].min()).days
        days_in_market = df_trades['duration'].sum() / 24
        time_in_market = (days_in_market / total_days * 100) if total_days > 0 else 0
        
        n = len(self.trades)
        win_rate = (len(winning_trades) / n) * 100 if n > 0 else 0
        
        if n >= 10:
            z_score = 1.96
            p = win_rate / 100
            margin_error = z_score * np.sqrt((p * (1 - p)) / n) * 100
            ci_lower = max(0, win_rate - margin_error)
            ci_upper = min(100, win_rate + margin_error)
        else:
            ci_lower = 0
            ci_upper = 100
        
        if n >= 10:
            wins = len(winning_trades)
            p_value = stats.binomtest(wins, n, 0.5, alternative='greater').pvalue
            statistically_significant = p_value < 0.05
        else:
            p_value = 1.0
            statistically_significant = False
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_ci_lower': ci_lower,
            'win_rate_ci_upper': ci_upper,
            'p_value': p_value,
            'statistically_significant': statistically_significant,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'avg_win_pct': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss_pct': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': df_trades['pnl'].max(),
            'largest_loss': df_trades['pnl'].min(),
            'largest_win_pct': df_trades['pnl_pct'].max(),
            'largest_loss_pct': df_trades['pnl_pct'].min(),
            'max_drawdown': max_drawdown,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0,
            'avg_trade_duration': df_trades['duration'].mean(),
            'sharpe_ratio': sharpe,
            'time_in_market': time_in_market,
            'expectancy': df_trades['pnl'].mean(),
            'expectancy_pct': df_trades['pnl_pct'].mean()
        }
        
        return metrics, df_trades, df_equity
    
    def print_results(self, metrics, show_full=True):
        """Imprime resultados"""
        if show_full:
            print("\n" + "="*70)
            print("RESULTADOS")
            print("="*70)
            print(f"Capital Inicial:      ${metrics['initial_capital']:,.2f}")
            print(f"Capital Final:        ${metrics['final_capital']:,.2f}")
            print(f"Retorno Total:        {metrics['total_return']:.2f}%")
            print(f"Expectância:          {metrics['expectancy_pct']:.2f}%")
            print(f"\nTrades:               {metrics['total_trades']}")
            print(f"Win Rate:             {metrics['win_rate']:.1f}%")
            print(f"Profit Factor:        {metrics['profit_factor']:.2f}")
            print(f"Max Drawdown:         {metrics['max_drawdown']:.2f}%")
            print("="*70)
        else:
            sig = "✅" if metrics['statistically_significant'] else "⚠️"
            print(f"T:{metrics['total_trades']:3d} | WR:{metrics['win_rate']:5.1f}% {sig} | "
                  f"Exp:{metrics['expectancy_pct']:6.2f}% | Ret:{metrics['total_return']:8.2f}%")


def massive_4h_optimization(df_data_4h):
    """OTIMIZAÇÃO MASSIVA COMPLETA - TODOS OS FATORES"""
    print("\n" + "="*80)
    print("🚀 OTIMIZAÇÃO MASSIVA 4H - TODOS OS FATORES 🚀")
    print("="*80)
    print(f"\nDados: {len(df_data_4h)} candles ({df_data_4h.index[0]} - {df_data_4h.index[-1]})")
    print(f"\nFATORES TESTADOS:")
    print("  ✓ Body %: 0, 20, 30, 40, 50, 60")
    print("  ✓ Candle Size: 0, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5")
    print("  ✓ Take Profit: 0, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0")
    print("  ✓ Exit First Profit: Sim/Não")
    print("  ✓ Exit on MA Turn: Sim/Não")
    print()
    
    # Grid COMPLETO
    body_pct_values = [0, 20, 30, 40, 50, 60]
    candle_size_values = [0, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
    take_profit_values = [0, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    exit_first_profit_values = [False, True]
    exit_on_ma_turn_values = [False, True]
    
    results = []
    tested = 0
    valid = 0
    
    total_combinations = (len(body_pct_values) * len(candle_size_values) * 
                         len(take_profit_values) * len(exit_first_profit_values) *
                         len(exit_on_ma_turn_values))
    
    print(f"📊 Total de combinações: {total_combinations}")
    print(f"⏱️  Tempo estimado: ~{total_combinations * 0.2 / 60:.1f} minutos\n")
    
    for body_pct, size_mult, tp_mult, exit_fp, exit_ma in product(
        body_pct_values, candle_size_values, take_profit_values,
        exit_first_profit_values, exit_on_ma_turn_values):
        
        tested += 1
        
        # Pular combinações inválidas
        if exit_fp and tp_mult > 0:  # FirstProfit e TP fixo = redundante
            continue
        
        if exit_fp and exit_ma:  # FirstProfit e MA Turn = redundante
            continue
        
        if tp_mult > 0 and exit_ma:  # TP fixo e MA Turn = redundante
            continue
        
        if tested % 50 == 0:
            print(f"  [{tested}/{total_combinations}] {tested/total_combinations*100:.1f}% | Válidas: {valid}")
        
        bt = BTCBacktest(
            timeframe='4h',
            ma_period=8,
            initial_capital=10000,
            body_pct_min=body_pct,
            close_position_min=0,
            candle_size_multiplier=size_mult,
            take_profit_multiplier=tp_mult,
            exit_first_profit=exit_fp,
            exit_on_ma_turn=exit_ma
        )
        
        df_test = df_data_4h.copy()
        df_test = bt.calculate_candle_metrics(df_test)
        df_test = bt.calculate_ma(df_test)
        bt.run_backtest(df_test)
        
        metrics, trades, equity = bt.calculate_metrics()
        
        # Critério: mínimo 30 trades
        if metrics['total_trades'] >= 30:
            valid += 1
            
            # Determinar estratégia
            if exit_fp:
                exit_strategy = "FirstProfit"
            elif tp_mult > 0:
                exit_strategy = f"TP{tp_mult}x"
            elif exit_ma:
                exit_strategy = "MA_Turn"
            else:
                exit_strategy = "None"
            
            results.append({
                'body_pct': body_pct,
                'candle_size_mult': size_mult,
                'take_profit_mult': tp_mult,
                'exit_first_profit': exit_fp,
                'exit_on_ma_turn': exit_ma,
                'exit_strategy': exit_strategy,
                **metrics
            })
    
    print(f"\n✅ Otimização concluída!")
    print(f"   Testadas: {tested} combinações")
    print(f"   Válidas: {valid} (com 30+ trades)\n")
    
    if not results:
        print("❌ Nenhuma configuração válida!")
        return None
    
    df_results = pd.DataFrame(results)
    
    # Salvar CSV completo
    os.makedirs('results/optimization', exist_ok=True)
    df_results.to_csv('results/optimization/4h_massive_full.csv', index=False)
    print(f"💾 Resultados salvos em: results/optimization/4h_massive_full.csv\n")
    
    # === RANKING 1: Máximo Retorno ===
    print("="*100)
    print("💰 TOP 20 - MÁXIMO RETORNO")
    print("="*100)
    
    df_by_return = df_results.sort_values('total_return', ascending=False)
    print(f"{'#':<3} {'Body%':<6} {'Size':<5} {'Exit':<12} {'Trades':<7} {'WR%':<7} "
          f"{'Exp%':<7} {'Ret%':<10} {'PF':<5} {'DD%':<7} {'Sig':<4}")
    print("-"*100)
    
    for idx, row in df_by_return.head(20).iterrows():
        sig = "✅" if row['statistically_significant'] else "⚠️"
        print(f"{df_by_return.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<6.0f} "
              f"{row['candle_size_mult']:<5.1f} "
              f"{row['exit_strategy']:<12} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<7.1f} "
              f"{row['expectancy_pct']:<7.2f} "
              f"{row['total_return']:<10.2f} "
              f"{row['profit_factor']:<5.2f} "
              f"{row['max_drawdown']:<7.2f} "
              f"{sig:<4}")
    
    # === RANKING 2: Máxima Expectância ===
    print("\n" + "="*100)
    print("📈 TOP 20 - MÁXIMA EXPECTÂNCIA")
    print("="*100)
    
    df_by_exp = df_results.sort_values('expectancy_pct', ascending=False)
    print(f"{'#':<3} {'Body%':<6} {'Size':<5} {'Exit':<12} {'Trades':<7} {'WR%':<7} "
          f"{'Exp%':<7} {'Ret%':<10} {'PF':<5} {'DD%':<7} {'Sig':<4}")
    print("-"*100)
    
    for idx, row in df_by_exp.head(20).iterrows():
        sig = "✅" if row['statistically_significant'] else "⚠️"
        print(f"{df_by_exp.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<6.0f} "
              f"{row['candle_size_mult']:<5.1f} "
              f"{row['exit_strategy']:<12} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<7.1f} "
              f"{row['expectancy_pct']:<7.2f} "
              f"{row['total_return']:<10.2f} "
              f"{row['profit_factor']:<5.2f} "
              f"{row['max_drawdown']:<7.2f} "
              f"{sig:<4}")
    
    # === RANKING 3: Máximo Win Rate ===
    print("\n" + "="*100)
    print("🏆 TOP 20 - MÁXIMO WIN RATE")
    print("="*100)
    
    df_by_wr = df_results.sort_values('win_rate', ascending=False)
    print(f"{'#':<3} {'Body%':<6} {'Size':<5} {'Exit':<12} {'Trades':<7} {'WR%':<7} "
          f"{'Exp%':<7} {'Ret%':<10} {'PF':<5} {'DD%':<7} {'Sig':<4}")
    print("-"*100)
    
    for idx, row in df_by_wr.head(20).iterrows():
        sig = "✅" if row['statistically_significant'] else "⚠️"
        print(f"{df_by_wr.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<6.0f} "
              f"{row['candle_size_mult']:<5.1f} "
              f"{row['exit_strategy']:<12} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<7.1f} "
              f"{row['expectancy_pct']:<7.2f} "
              f"{row['total_return']:<10.2f} "
              f"{row['profit_factor']:<5.2f} "
              f"{row['max_drawdown']:<7.2f} "
              f"{sig:<4}")
    
    # === RANKING 4: Balanceado ===
    print("\n" + "="*100)
    print("⚖️  TOP 20 - MELHOR BALANCEADO (Ret 40% + Exp 30% + WR 30%)")
    print("="*100)
    
    # Normalizar
    df_results['return_norm'] = (df_results['total_return'] - df_results['total_return'].min()) / (df_results['total_return'].max() - df_results['total_return'].min() + 0.001)
    df_results['exp_norm'] = (df_results['expectancy_pct'] - df_results['expectancy_pct'].min()) / (df_results['expectancy_pct'].max() - df_results['expectancy_pct'].min() + 0.001)
    df_results['wr_norm'] = (df_results['win_rate'] - df_results['win_rate'].min()) / (df_results['win_rate'].max() - df_results['win_rate'].min() + 0.001)
    
    df_results['composite_score'] = (df_results['return_norm'] * 0.4 + 
                                      df_results['exp_norm'] * 0.3 + 
                                      df_results['wr_norm'] * 0.3)
    
    df_balanced = df_results.sort_values('composite_score', ascending=False)
    print(f"{'#':<3} {'Body%':<6} {'Size':<5} {'Exit':<12} {'Trades':<7} {'WR%':<7} "
          f"{'Exp%':<7} {'Ret%':<10} {'PF':<5} {'Score':<6}")
    print("-"*100)
    
    for idx, row in df_balanced.head(20).iterrows():
        print(f"{df_balanced.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<6.0f} "
              f"{row['candle_size_mult']:<5.1f} "
              f"{row['exit_strategy']:<12} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<7.1f} "
              f"{row['expectancy_pct']:<7.2f} "
              f"{row['total_return']:<10.2f} "
              f"{row['profit_factor']:<5.2f} "
              f"{row['composite_score']:<6.3f}")
    
    # Melhor balanceado
    best = df_balanced.iloc[0]
    
    print("\n" + "="*70)
    print("🎯 CONFIGURAÇÃO VENCEDORA (Melhor Balanceada)")
    print("="*70)
    print(f"Body %:                 {best['body_pct']:.0f}%")
    print(f"Candle Size:            {best['candle_size_mult']:.1f}x")
    print(f"Estratégia Saída:       {best['exit_strategy']}")
    if best['take_profit_mult'] > 0:
        print(f"Take Profit:            {best['take_profit_mult']:.1f}x risco")
    
    print(f"\n📈 RESULTADOS:")
    print(f"Win Rate:               {best['win_rate']:.2f}%")
    print(f"Total Trades:           {best['total_trades']:.0f}")
    print(f"Expectância:            {best['expectancy_pct']:.2f}%")
    print(f"Retorno Total:          {best['total_return']:.2f}%")
    print(f"Profit Factor:          {best['profit_factor']:.2f}")
    print(f"Max Drawdown:           {best['max_drawdown']:.2f}%")
    print(f"Score Composto:         {best['composite_score']:.3f}")
    
    if best['statistically_significant']:
        print(f"Significância:          ✅ SIM (p={best['p_value']:.4f})")
    else:
        print(f"Significância:          ⚠️ NÃO (p={best['p_value']:.4f})")
    
    print("="*70)
    
    # Análise por estratégia
    print("\n" + "="*70)
    print("📊 ANÁLISE POR ESTRATÉGIA DE SAÍDA (Médias)")
    print("="*70)
    
    exit_analysis = df_results.groupby('exit_strategy').agg({
        'total_trades': 'mean',
        'win_rate': 'mean',
        'expectancy_pct': 'mean',
        'total_return': 'mean',
        'profit_factor': 'mean',
        'max_drawdown': 'mean'
    }).sort_values('total_return', ascending=False)
    
    print(exit_analysis.to_string())
    
    return best.to_dict()


def main():
    print("="*70)
    print("🚀 BACKTEST 4H (2020-2024) - OTIMIZAÇÃO COMPLETA 🚀")
    print("="*70)
    
    bt_temp = BTCBacktest(timeframe='4h', ma_period=8, initial_capital=10000)
    
    try:
        df_4h = bt_temp.download_data(years=5)
        print(f"✅ OK: {len(df_4h)} candles\n")
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return
    
    best_4h = massive_4h_optimization(df_4h)
    
    if best_4h is not None:
        print("\n" + "="*70)
        print("🏆 EXECUTANDO BACKTEST FINAL COM CONFIGURAÇÃO VENCEDORA")
        print("="*70)
        
        bt = BTCBacktest(
            timeframe='4h',
            ma_period=8,
            initial_capital=10000,
            body_pct_min=best_4h['body_pct'],
            close_position_min=0,
            candle_size_multiplier=best_4h['candle_size_mult'],
            take_profit_multiplier=best_4h['take_profit_mult'],
            exit_first_profit=best_4h['exit_first_profit'],
            exit_on_ma_turn=best_4h['exit_on_ma_turn']
        )
        
        df_test = df_4h.copy()
        df_test = bt.calculate_candle_metrics(df_test)
        df_test = bt.calculate_ma(df_test)
        bt.run_backtest(df_test)
        
        metrics, trades, equity = bt.calculate_metrics()
        bt.print_results(metrics, show_full=True)
        
        print("\n✅ BACKTEST COMPLETO!")
        print(f"📁 CSV completo salvo: results/optimization/4h_massive_full.csv")


if __name__ == "__main__":
    main()
