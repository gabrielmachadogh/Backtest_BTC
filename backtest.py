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
                 take_profit_multiplier=0, exit_first_profit=False):
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
    
    def download_from_mexc(self, days):
        """Baixa dados da MEXC API (mais fácil que Binance)"""
        print(f"Baixando dados da MEXC...")
        
        # Mapear timeframe
        interval_map = {
            '4h': '4h',
            '1d': '1d', 
            '1D': '1d',
            '1wk': '1w',
            '1W': '1w'
        }
        
        interval = interval_map.get(self.timeframe, '1d')
        
        endpoint = 'https://api.mexc.com/api/v3/klines'
        
        all_data = []
        limit = 1000  # MEXC permite 1000 por request
        
        # Calcular timestamps
        end_time = int(time.time() * 1000)
        
        # Calcular quantas requests precisamos
        if interval == '4h':
            candles_per_day = 6
        elif interval == '1d':
            candles_per_day = 1
        elif interval == '1w':
            candles_per_day = 1/7
        else:
            candles_per_day = 1
        
        total_candles_needed = int(days * candles_per_day)
        iterations = (total_candles_needed // limit) + 1
        
        print(f"  Buscando ~{total_candles_needed} candles em {iterations} requests...")
        
        for i in range(iterations):
            params = {
                'symbol': 'BTCUSDT',
                'interval': interval,
                'limit': limit,
                'endTime': end_time
            }
            
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data and isinstance(data, list):
                    all_data.extend(data)
                    
                    if len(data) < limit:
                        break
                    
                    # Próximo batch - pega o timestamp do primeiro candle e subtrai 1ms
                    end_time = data[0][0] - 1
                    
                    # Rate limiting
                    time.sleep(0.2)
                else:
                    print(f"  Request {i+1}: Resposta vazia ou inválida")
                    break
                    
            except Exception as e:
                print(f"  Erro ao baixar chunk {i}: {str(e)}")
                break
        
        if not all_data:
            return None
        
        # Converter para DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Converter tipos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        
        # Selecionar apenas colunas necessárias
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.sort_index()
        
        # Remover duplicatas
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def download_from_cryptocompare(self, days):
        """Baixa dados do CryptoCompare API"""
        print(f"Baixando dados do CryptoCompare...")
        
        # Escolher endpoint baseado no timeframe
        if self.timeframe in ['4h', '4H']:
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histohour'
            limit = min(days * 6, 2000)  # 6 candles de 4h por dia
        elif self.timeframe in ['1d', '1D']:
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
            limit = min(days, 2000)
        elif self.timeframe in ['1wk', '1W']:
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
            limit = min(days // 7, 2000)
        else:
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
            limit = min(days, 2000)
        
        all_data = []
        to_timestamp = int(time.time())
        
        iterations = (limit // 2000) + 1
        
        for i in range(iterations):
            params = {
                'fsym': 'BTC',
                'tsym': 'USD',
                'limit': min(2000, limit - len(all_data)),
                'toTs': to_timestamp
            }
            
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data['Response'] == 'Success':
                    df_chunk = pd.DataFrame(data['Data']['Data'])
                    all_data.append(df_chunk)
                    
                    if len(df_chunk) < 2000:
                        break
                    
                    to_timestamp = int(df_chunk['time'].min()) - 86400
                    time.sleep(1)
                else:
                    break
                    
            except Exception as e:
                print(f"  Erro ao baixar chunk {i}: {str(e)}")
                break
        
        if not all_data:
            return None
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('time')
        df = df.drop_duplicates(subset=['time'])
        
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('Date', inplace=True)
        
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volumefrom': 'Volume'
        })
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Agregar para 4h se necessário
        if self.timeframe in ['4h', '4H'] and endpoint == 'histohour':
            df = df.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        # Agregar para semanal se necessário
        if self.timeframe in ['1wk', '1W']:
            df = df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        return df
    
    def download_from_binance(self, days):
        """Baixa dados da API pública da Binance"""
        print(f"Baixando dados da Binance...")
        
        # Mapear timeframe
        interval_map = {
            '4h': '4h',
            '1d': '1d',
            '1D': '1d',
            '1wk': '1w',
            '1W': '1w'
        }
        
        interval = interval_map.get(self.timeframe, '1d')
        
        endpoint = 'https://api.binance.com/api/v3/klines'
        
        all_data = []
        limit = 1000
        end_time = int(time.time() * 1000)
        
        # Calcular iterações necessárias
        if interval == '4h':
            candles_needed = days * 6
        elif interval == '1d':
            candles_needed = days
        elif interval == '1w':
            candles_needed = days // 7
        else:
            candles_needed = days
        
        iterations = (candles_needed // limit) + 1
        
        for i in range(iterations):
            params = {
                'symbol': 'BTCUSDT',
                'interval': interval,
                'limit': limit,
                'endTime': end_time
            }
            
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data:
                    all_data.extend(data)
                    
                    if len(data) < limit:
                        break
                    
                    end_time = data[0][0] - 1
                    time.sleep(0.5)
                else:
                    break
                    
            except Exception as e:
                print(f"  Erro ao baixar chunk {i}: {str(e)}")
                break
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.sort_index()
        
        return df
    
    def download_data(self, years=15):
        """Baixa dados históricos do BTC com múltiplas fontes"""
        days = years * 365
        df = None
        
        # 1. Tentar MEXC primeiro (mais fácil)
        try:
            df = self.download_from_mexc(days)
            if df is not None and len(df) > 100:
                print(f"✅ MEXC: {len(df)} candles de {df.index[0]} até {df.index[-1]}")
                return df
        except Exception as e:
            print(f"  MEXC falhou: {str(e)}")
        
        # 2. Tentar Binance
        try:
            df = self.download_from_binance(days)
            if df is not None and len(df) > 100:
                print(f"✅ Binance: {len(df)} candles de {df.index[0]} até {df.index[-1]}")
                return df
        except Exception as e:
            print(f"  Binance falhou: {str(e)}")
        
        # 3. Tentar CryptoCompare
        try:
            df = self.download_from_cryptocompare(days)
            if df is not None and len(df) > 100:
                print(f"✅ CryptoCompare: {len(df)} candles de {df.index[0]} até {df.index[-1]}")
                return df
        except Exception as e:
            print(f"  CryptoCompare falhou: {str(e)}")
        
        # 4. Tentar yfinance como último recurso (não tem 4h)
        if self.timeframe in ['1d', '1D', '1wk', '1W']:
            try:
                import yfinance as yf
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                interval_map = {
                    '1d': '1d',
                    '1D': '1d',
                    '1wk': '1wk',
                    '1W': '1wk'
                }
                
                interval = interval_map.get(self.timeframe, '1d')
                
                for ticker in ["BTC-USD", "BTCUSD=X"]:
                    try:
                        data = yf.download(
                            ticker,
                            start=start_date,
                            end=end_date,
                            interval=interval,
                            progress=False,
                            auto_adjust=True
                        )
                        
                        if not data.empty and len(data) > 100:
                            df = data
                            print(f"✅ Yahoo Finance: {len(df)} candles")
                            break
                    except:
                        continue
            except Exception as e:
                print(f"  yfinance falhou: {str(e)}")
        
        if df is None or df.empty:
            raise Exception("Não foi possível baixar dados de nenhuma fonte")
        
        return df
    
    def calculate_candle_metrics(self, df):
        """Calcula métricas dos candles para filtros"""
        df['Candle_Range'] = df['High'] - df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Body_Pct'] = (df['Body_Size'] / df['Candle_Range'] * 100).fillna(0)
        df['Close_Position'] = ((df['Close'] - df['Low']) / df['Candle_Range'] * 100).fillna(50)
        df['Avg_Candle_Size'] = df['Candle_Range'].rolling(window=20).mean()
        df['Size_vs_Avg'] = df['Candle_Range'] / df['Avg_Candle_Size']
        
        return df
    
    def calculate_ma(self, df):
        """Calcula média móvel de 8 períodos"""
        df['MA8'] = df['Close'].rolling(window=self.ma_period).mean()
        
        df['MA_Direction'] = 0
        df.loc[df['MA8'] > df['MA8'].shift(1), 'MA_Direction'] = 1
        df.loc[df['MA8'] < df['MA8'].shift(1), 'MA_Direction'] = -1
        
        df['MA_Turn_Up'] = (df['MA_Direction'] == 1) & (df['MA_Direction'].shift(1) != 1)
        df['MA_Turn_Down'] = (df['MA_Direction'] == -1) & (df['MA_Direction'].shift(1) != -1)
        
        return df
    
    def check_entry_filters(self, df, idx):
        """Verifica se o candle passa nos filtros de entrada"""
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
        """Executa o backtest com a estratégia"""
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
                        self.close_position(current_idx, exit_price, 'First Profit Close')
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
                
                if not self.exit_first_profit and self.take_profit_multiplier == 0:
                    if df.loc[df.index[i-1], 'MA_Turn_Down']:
                        exit_trigger = df.loc[df.index[i-1], 'Low']
                    
                    if df.loc[df.index[i-1], 'MA_Turn_Up'] and exit_trigger:
                        exit_trigger = None
                    
                    if exit_trigger and current_low <= exit_trigger:
                        self.close_position(current_idx, exit_trigger, 'Gatilho MA')
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
            self.close_position(df.index[-1], last_price, 'Fim do Backtest')
    
    def close_position(self, exit_date, exit_price, reason):
        """Fecha a posição atual"""
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
        """Calcula métricas de performance com significância estatística"""
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
        
        df_trades['duration'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.total_seconds() / 3600  # horas
        
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
        """Imprime resultados do backtest"""
        if show_full:
            duration_label = "horas" if self.timeframe == '4h' else "dias"
            
            print("\n" + "="*70)
            print("RESULTADOS DO BACKTEST")
            print("="*70)
            print(f"Capital Inicial:      ${metrics['initial_capital']:,.2f}")
            print(f"Capital Final:        ${metrics['final_capital']:,.2f}")
            print(f"Retorno Total:        {metrics['total_return']:.2f}%")
            print(f"Expectância/Trade:    ${metrics['expectancy']:,.2f} ({metrics['expectancy_pct']:.2f}%)")
            
            print(f"\n{'TRADES':-^70}")
            print(f"Total de Trades:      {metrics['total_trades']}")
            print(f"Trades Vencedores:    {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
            print(f"Trades Perdedores:    {metrics['losing_trades']} ({100-metrics['win_rate']:.1f}%)")
            print(f"Win Rate IC 95%:      [{metrics['win_rate_ci_lower']:.1f}%, {metrics['win_rate_ci_upper']:.1f}%]")
            
            if metrics['statistically_significant']:
                print(f"Significância:        ✅ SIM (p={metrics['p_value']:.4f})")
            else:
                print(f"Significância:        ⚠️ NÃO (p={metrics['p_value']:.4f}) - Amostra pequena")
            
            print(f"Duração Média:        {metrics['avg_trade_duration']:.1f} {duration_label}")
            print(f"Tempo em Mercado:     {metrics['time_in_market']:.1f}%")
            
            print(f"\n{'GANHOS/PERDAS':-^70}")
            print(f"Ganho Médio:          ${metrics['avg_win']:,.2f} ({metrics['avg_win_pct']:.2f}%)")
            print(f"Perda Média:          ${metrics['avg_loss']:,.2f} ({metrics['avg_loss_pct']:.2f}%)")
            print(f"Maior Ganho:          ${metrics['largest_win']:,.2f} ({metrics['largest_win_pct']:.2f}%)")
            print(f"Maior Perda:          ${metrics['largest_loss']:,.2f} ({metrics['largest_loss_pct']:.2f}%)")
            
            print(f"\n{'MÉTRICAS DE RISCO':-^70}")
            print(f"Max Drawdown:         {metrics['max_drawdown']:.2f}%")
            print(f"Profit Factor:        {metrics['profit_factor']:.2f}")
            print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
            
            print("="*70)
        else:
            sig = "✅" if metrics['statistically_significant'] else "⚠️"
            print(f"T:{metrics['total_trades']:3d} | WR:{metrics['win_rate']:5.1f}% {sig} | "
                  f"Exp:{metrics['expectancy_pct']:6.2f}% | PF:{metrics['profit_factor']:5.2f}")
    
    def plot_results(self, df, df_equity, df_trades, timeframe_name):
        """Gera gráficos dos resultados"""
        output_dir = f'results/{timeframe_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=(f'Preço BTC ({self.timeframe}) e Média Móvel 8', 'Equity Curve', 'Drawdown'),
                           row_heights=[0.5, 0.3, 0.2])
        
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='BTC'),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MA8'],
                                mode='lines',
                                name='MA8',
                                line=dict(color='orange', width=2)),
                     row=1, col=1)
        
        if not df_trades.empty:
            entries = df_trades[['entry_date', 'entry_price']].copy()
            fig.add_trace(go.Scatter(x=entries['entry_date'],
                                    y=entries['entry_price'],
                                    mode='markers',
                                    name='Entrada',
                                    marker=dict(color='green', size=10, symbol='triangle-up')),
                         row=1, col=1)
            
            exits = df_trades[['exit_date', 'exit_price']].copy()
            fig.add_trace(go.Scatter(x=exits['exit_date'],
                                    y=exits['exit_price'],
                                    mode='markers',
                                    name='Saída',
                                    marker=dict(color='red', size=10, symbol='triangle-down')),
                         row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_equity['date'],
                                y=df_equity['equity'],
                                mode='lines',
                                name='Equity',
                                line=dict(color='blue', width=2),
                                fill='tozeroy'),
                     row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df_equity['date'],
                                y=df_equity['drawdown'],
                                mode='lines',
                                name='Drawdown',
                                line=dict(color='red', width=1),
                                fill='tozeroy'),
                     row=3, col=1)
        
        fig.update_layout(
            title=f'Backtest BTC - Estratégia MA8 ({self.timeframe})',
            xaxis_title='Data',
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=2, col=1)
        
        fig.write_html(f'{output_dir}/backtest_chart.html')
        
        if df_trades.empty:
            return
        
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        ax1.hist(df_trades['pnl_pct'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_title('Distribuição de Retornos (%)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Retorno (%)')
        ax1.set_ylabel('Frequência')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.axvline(x=df_trades['pnl_pct'].mean(), color='g', linestyle='--', linewidth=2, 
                   label=f"Média: {df_trades['pnl_pct'].mean():.1f}%")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        colors = ['green' if x > 0 else 'red' for x in df_trades['pnl_pct']]
        ax2.bar(range(len(df_trades)), df_trades['pnl_pct'], color=colors, alpha=0.7)
        ax2.set_title('Retorno por Trade (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Trade #')
        ax2.set_ylabel('Retorno (%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(df_trades.index, df_trades['capital'], linewidth=2, color='blue', marker='o', markersize=3)
        ax3.set_title('Evolução do Capital', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trade #')
        ax3.set_ylabel('Capital ($)')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, which='both')
        ax3.fill_between(df_trades.index, df_trades['capital'], alpha=0.3)
        
        ax4.scatter(df_trades['duration'], df_trades['pnl_pct'], 
                   c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        ax4.set_title('Duração vs Retorno', fontsize=12, fontweight='bold')
        duration_label = 'horas' if self.timeframe == '4h' else 'dias'
        ax4.set_xlabel(f'Duração ({duration_label})')
        ax4.set_ylabel('Retorno (%)')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_trades_csv(self, df_trades, timeframe_name):
        """Salva trades em CSV"""
        output_dir = f'results/{timeframe_name}'
        os.makedirs(output_dir, exist_ok=True)
        df_trades.to_csv(f'{output_dir}/trades.csv', index=False)
    
    def save_summary(self, metrics, timeframe_name, filter_config=None):
        """Salva resumo em arquivo de texto"""
        output_dir = f'results/{timeframe_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/summary.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"BACKTEST BTC - ESTRATÉGIA MA8 ({self.timeframe})\n")
            f.write("="*70 + "\n\n")
            
            if filter_config:
                f.write("CONFIGURAÇÃO:\n")
                f.write(f"  Body % mínimo:           {filter_config['body_pct']}%\n")
                f.write(f"  Close Position mínimo:   {filter_config['close_pos']}%\n")
                f.write(f"  Candle Size multiplier:  {filter_config['size_mult']}x\n")
                f.write(f"  Take Profit multiplier:  {filter_config['tp_mult']}x\n")
                f.write(f"  Exit First Profit:       {filter_config['exit_fp']}\n\n")
            
            f.write(f"Capital Inicial:      ${metrics['initial_capital']:,.2f}\n")
            f.write(f"Capital Final:        ${metrics['final_capital']:,.2f}\n")
            f.write(f"Retorno Total:        {metrics['total_return']:.2f}%\n")
            f.write(f"Expectância/Trade:    ${metrics['expectancy']:,.2f} ({metrics['expectancy_pct']:.2f}%)\n\n")
            
            f.write(f"{'TRADES':-^70}\n")
            f.write(f"Total de Trades:      {metrics['total_trades']}\n")
            f.write(f"Trades Vencedores:    {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)\n")
            f.write(f"Trades Perdedores:    {metrics['losing_trades']} ({100-metrics['win_rate']:.1f}%)\n")
            
            duration_label = "horas" if self.timeframe == '4h' else "dias"
            f.write(f"Duração Média:        {metrics['avg_trade_duration']:.1f} {duration_label}\n")
            f.write(f"Tempo em Mercado:     {metrics['time_in_market']:.1f}%\n\n")
            
            f.write(f"{'GANHOS/PERDAS':-^70}\n")
            f.write(f"Ganho Médio:          ${metrics['avg_win']:,.2f} ({metrics['avg_win_pct']:.2f}%)\n")
            f.write(f"Perda Média:          ${metrics['avg_loss']:,.2f} ({metrics['avg_loss_pct']:.2f}%)\n")
            f.write(f"Maior Ganho:          ${metrics['largest_win']:,.2f} ({metrics['largest_win_pct']:.2f}%)\n")
            f.write(f"Maior Perda:          ${metrics['largest_loss']:,.2f} ({metrics['largest_loss_pct']:.2f}%)\n\n")
            
            f.write(f"{'MÉTRICAS DE RISCO':-^70}\n")
            f.write(f"Max Drawdown:         {metrics['max_drawdown']:.2f}%\n")
            f.write(f"Profit Factor:        {metrics['profit_factor']:.2f}\n")
            f.write(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}\n")
            f.write("="*70 + "\n")


def test_4h_timeframe(df_data_4h):
    """Testa configuração vencedora em 4H"""
    print("\n" + "="*70)
    print("🕐 TESTE EM 4 HORAS - CONFIGURAÇÃO VENCEDORA DO DIÁRIO")
    print("="*70)
    print("\nAplicando config que deu 83.7% WR no diário:")
    print("  Body: 0%, Candle Size: 1.5x, Exit: FirstProfit\n")
    
    bt_4h = BTCBacktest(
        timeframe='4h',
        ma_period=8,
        initial_capital=10000,
        body_pct_min=0,
        close_position_min=0,
        candle_size_multiplier=1.5,
        exit_first_profit=True
    )
    
    df_test = df_data_4h.copy()
    df_test = bt_4h.calculate_candle_metrics(df_test)
    df_test = bt_4h.calculate_ma(df_test)
    bt_4h.run_backtest(df_test)
    
    metrics, trades, equity = bt_4h.calculate_metrics()
    bt_4h.print_results(metrics, show_full=True)
    bt_4h.plot_results(df_test, equity, trades, '4h_test')
    bt_4h.save_trades_csv(trades, '4h_test')
    bt_4h.save_summary(metrics, '4h_test', {
        'body_pct': 0,
        'close_pos': 0,
        'size_mult': 1.5,
        'tp_mult': 0,
        'exit_fp': True
    })
    
    return metrics


def optimize_4h(df_data_4h):
    """Otimiza especificamente para 4H"""
    print("\n" + "="*70)
    print("🔧 OTIMIZAÇÃO ESPECÍFICA PARA 4 HORAS")
    print("="*70)
    print("\nTestando configurações adaptadas para 4H...\n")
    
    # Valores para 4H
    body_pct_values = [0, 30, 40]
    close_position_values = [0]
    candle_size_values = [0, 1.0, 1.2, 1.5, 1.8]
    take_profit_values = [0, 1.5, 2.0]
    exit_first_profit_values = [False, True]
    
    results = []
    
    for body_pct, close_pos, size_mult, tp_mult, exit_fp in product(
        body_pct_values, close_position_values, candle_size_values, 
        take_profit_values, exit_first_profit_values):
        
        if exit_fp and tp_mult > 0:
            continue
        
        bt = BTCBacktest(
            timeframe='4h',
            ma_period=8,
            initial_capital=10000,
            body_pct_min=body_pct,
            close_position_min=close_pos,
            candle_size_multiplier=size_mult,
            take_profit_multiplier=tp_mult,
            exit_first_profit=exit_fp
        )
        
        df_test = df_data_4h.copy()
        df_test = bt.calculate_candle_metrics(df_test)
        df_test = bt.calculate_ma(df_test)
        bt.run_backtest(df_test)
        
        metrics, trades, equity = bt.calculate_metrics()
        
        if metrics['total_trades'] >= 30:
            exit_strategy = "FirstProfit" if exit_fp else (f"TP{tp_mult}x" if tp_mult > 0 else "MA")
            
            results.append({
                'body_pct': body_pct,
                'candle_size_mult': size_mult,
                'take_profit_mult': tp_mult,
                'exit_first_profit': exit_fp,
                'exit_strategy': exit_strategy,
                **metrics
            })
            
            bt.print_results(metrics, show_full=False)
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("\n⚠️ Nenhuma configuração com 30+ trades encontrada!")
        return None
    
    # Salvar
    os.makedirs('results/optimization', exist_ok=True)
    df_results.to_csv('results/optimization/4h_optimization.csv', index=False)
    
    # Top 10
    print("\n" + "="*90)
    print("🏆 TOP 10 - TIMEFRAME 4H")
    print("="*90)
    
    df_by_wr = df_results.sort_values('win_rate', ascending=False)
    print(f"{'#':<3} {'Body%':<6} {'Size':<5} {'Exit':<11} {'Trades':<7} {'WR%':<7} "
          f"{'Exp%':<7} {'Ret%':<10} {'PF':<5} {'Sig':<4}")
    print("-"*90)
    
    for idx, row in df_by_wr.head(10).iterrows():
        sig = "✅" if row['statistically_significant'] else "⚠️"
        print(f"{df_by_wr.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<6.0f} "
              f"{row['candle_size_mult']:<5.1f} "
              f"{row['exit_strategy']:<11} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<7.1f} "
              f"{row['expectancy_pct']:<7.2f} "
              f"{row['total_return']:<10.2f} "
              f"{row['profit_factor']:<5.2f} "
              f"{sig:<4}")
    
    best = df_by_wr.iloc[0]
    
    print("\n" + "="*70)
    print("🎯 MELHOR CONFIGURAÇÃO 4H")
    print("="*70)
    print(f"Body % mínimo:          {best['body_pct']:.0f}%")
    print(f"Candle Size multiplier: {best['candle_size_mult']:.1f}x")
    print(f"Estratégia de Saída:    {best['exit_strategy']}")
    print(f"\nWin Rate:               {best['win_rate']:.2f}%")
    print(f"Total Trades:           {best['total_trades']:.0f}")
    print(f"Expectância:            {best['expectancy_pct']:.2f}%")
    print(f"Retorno Total:          {best['total_return']:.2f}%")
    print(f"Profit Factor:          {best['profit_factor']:.2f}")
    print("="*70)
    
    return best


def main():
    print("="*70)
    print("📊 BACKTEST BTC - TIMEFRAME 4 HORAS 📊")
    print("="*70)
    
    # Baixar dados 4H
    print("\nBaixando dados 4H...")
    bt_temp = BTCBacktest(timeframe='4h', ma_period=8, initial_capital=10000)
    
    try:
        df_4h = bt_temp.download_data(years=5)  # 5 anos de 4H = ~11k candles
        print(f"✅ Dados 4H carregados com sucesso!\n")
    except Exception as e:
        print(f"❌ Erro ao baixar dados 4H: {str(e)}")
        return
    
    # Teste com config vencedora do diário
    metrics_test = test_4h_timeframe(df_4h)
    
    # Otimização específica para 4H
    best_4h = optimize_4h(df_4h)
    
    if best_4h is not None:
        print("\n" + "="*70)
        print("🏆 EXECUTANDO BACKTEST FINAL 4H")
        print("="*70)
        
        bt_best = BTCBacktest(
            timeframe='4h',
            ma_period=8,
            initial_capital=10000,
            body_pct_min=best_4h['body_pct'],
            close_position_min=0,
            candle_size_multiplier=best_4h['candle_size_mult'],
            take_profit_multiplier=best_4h['take_profit_mult'],
            exit_first_profit=best_4h['exit_first_profit']
        )
        
        df_4h_best = df_4h.copy()
        df_4h_best = bt_best.calculate_candle_metrics(df_4h_best)
        df_4h_best = bt_best.calculate_ma(df_4h_best)
        bt_best.run_backtest(df_4h_best)
        
        metrics_best, trades_best, equity_best = bt_best.calculate_metrics()
        bt_best.print_results(metrics_best, show_full=True)
        bt_best.plot_results(df_4h_best, equity_best, trades_best, '4h_optimized')
        bt_best.save_trades_csv(trades_best, '4h_optimized')
        bt_best.save_summary(metrics_best, '4h_optimized', {
            'body_pct': best_4h['body_pct'],
            'close_pos': 0,
            'size_mult': best_4h['candle_size_mult'],
            'tp_mult': best_4h['take_profit_mult'],
            'exit_fp': best_4h['exit_first_profit']
        })
    
    print("\n" + "="*70)
    print("✅ BACKTEST 4H CONCLUÍDO!")
    print("="*70)
    print("\nResultados salvos em:")
    print("  • results/4h_test/       (config diário aplicada em 4H)")
    print("  • results/4h_optimized/  (config otimizada para 4H)")
    print("  • results/optimization/4h_optimization.csv")


if __name__ == "__main__":
    main()
