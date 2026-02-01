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

class BTCBacktest:
    def __init__(self, timeframe='1d', ma_period=8, initial_capital=10000, 
                 body_pct_min=0, close_position_min=0, candle_size_multiplier=0,
                 take_profit_multiplier=0):
        self.timeframe = timeframe
        self.ma_period = ma_period
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        
        # Filtros adicionais
        self.body_pct_min = body_pct_min
        self.close_position_min = close_position_min
        self.candle_size_multiplier = candle_size_multiplier
        self.take_profit_multiplier = take_profit_multiplier  # NOVO: alvo fixo
        
    def download_from_cryptocompare(self, days):
        """Baixa dados do CryptoCompare API"""
        print(f"Baixando dados do CryptoCompare...")
        
        if self.timeframe == '1d':
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
            limit = min(days, 2000)
        elif self.timeframe == '1wk':
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
            limit = min(days // 7, 2000)
        else:
            endpoint = 'https://min-api.cryptocompare.com/data/v2/histoday'
            limit = min(days, 2000)
        
        all_data = []
        to_timestamp = int(time.time())
        
        iterations = (days // limit) + 1
        
        for i in range(iterations):
            params = {
                'fsym': 'BTC',
                'tsym': 'USD',
                'limit': limit,
                'toTs': to_timestamp
            }
            
            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data['Response'] == 'Success':
                    df_chunk = pd.DataFrame(data['Data']['Data'])
                    all_data.append(df_chunk)
                    
                    if len(df_chunk) < limit:
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
        
        if self.timeframe == '1wk':
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
        
        if self.timeframe == '1d':
            interval = '1d'
        elif self.timeframe == '1wk':
            interval = '1w'
        else:
            interval = '1d'
        
        endpoint = 'https://api.binance.com/api/v3/klines'
        
        all_data = []
        limit = 1000
        end_time = int(time.time() * 1000)
        
        iterations = (days // limit) + 1
        
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
        """Baixa dados históricos do BTC"""
        days = years * 365
        df = None
        
        try:
            df = self.download_from_binance(days)
            if df is not None and len(df) > 100:
                return df
        except Exception as e:
            print(f"  Binance falhou: {str(e)}")
        
        try:
            df = self.download_from_cryptocompare(days)
            if df is not None and len(df) > 100:
                return df
        except Exception as e:
            print(f"  CryptoCompare falhou: {str(e)}")
        
        try:
            import yfinance as yf
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            for ticker in ["BTC-USD", "BTCUSD=X"]:
                try:
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        interval=self.timeframe,
                        progress=False,
                        auto_adjust=True
                    )
                    
                    if not data.empty and len(data) > 100:
                        df = data
                        break
                except:
                    continue
        except:
            pass
        
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
            
            if self.position is None:
                if df.loc[df.index[i-1], 'MA_Turn_Up']:
                    if self.check_entry_filters(df, df.index[i-1]):
                        entry_trigger = df.loc[df.index[i-1], 'High']
                        stop_loss = df.loc[df.index[i-1], 'Low']
                
                if entry_trigger and current_price >= entry_trigger:
                    quantity = self.capital / entry_trigger
                    
                    # Calcula take profit se configurado
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
                # Verifica take profit primeiro
                if self.position['take_profit'] and current_price >= self.position['take_profit']:
                    exit_price = self.position['take_profit']
                    self.close_position(current_idx, exit_price, 'Take Profit')
                    exit_trigger = None
                    continue
                
                # Verifica stop loss
                if current_low <= self.position['stop_loss']:
                    exit_price = self.position['stop_loss']
                    self.close_position(current_idx, exit_price, 'Stop Loss')
                    exit_trigger = None
                    continue
                
                # Detecta virada da MA para baixo (apenas se não tem TP fixo)
                if self.take_profit_multiplier == 0:
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
        """Calcula métricas de performance"""
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
        
        df_trades['duration'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days
        
        returns = df_equity['equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        total_days = (df_equity['date'].max() - df_equity['date'].min()).days
        days_in_market = df_trades['duration'].sum()
        time_in_market = (days_in_market / total_days * 100) if total_days > 0 else 0
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(self.trades)) * 100 if self.trades else 0,
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
            print(f"Duração Média:        {metrics['avg_trade_duration']:.1f} dias")
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
            print(f"Trades: {metrics['total_trades']:3d} | WinRate: {metrics['win_rate']:5.1f}% | "
                  f"Return: {metrics['total_return']:10.2f}% | PF: {metrics['profit_factor']:5.2f} | "
                  f"Exp: {metrics['expectancy_pct']:6.2f}% | DD: {metrics['max_drawdown']:6.2f}%")
    
    def plot_results(self, df, df_equity, df_trades, timeframe_name):
        """Gera gráficos dos resultados"""
        output_dir = f'results/{timeframe_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=('Preço BTC e Média Móvel 8', 'Equity Curve', 'Drawdown'),
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
        ax4.set_xlabel('Duração (dias)')
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
                f.write(f"  Take Profit multiplier:  {filter_config['tp_mult']}x\n\n")
            
            f.write(f"Capital Inicial:      ${metrics['initial_capital']:,.2f}\n")
            f.write(f"Capital Final:        ${metrics['final_capital']:,.2f}\n")
            f.write(f"Retorno Total:        {metrics['total_return']:.2f}%\n")
            f.write(f"Expectância/Trade:    ${metrics['expectancy']:,.2f} ({metrics['expectancy_pct']:.2f}%)\n\n")
            
            f.write(f"{'TRADES':-^70}\n")
            f.write(f"Total de Trades:      {metrics['total_trades']}\n")
            f.write(f"Trades Vencedores:    {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)\n")
            f.write(f"Trades Perdedores:    {metrics['losing_trades']} ({100-metrics['win_rate']:.1f}%)\n")
            f.write(f"Duração Média:        {metrics['avg_trade_duration']:.1f} dias\n")
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


def optimize_filters(df_data):
    """Otimiza os filtros incluindo take profit"""
    print("\n" + "="*70)
    print("OTIMIZAÇÃO COMPLETA - INCLUINDO TAKE PROFIT FIXO")
    print("="*70)
    print("\nTestando combinações de filtros...\n")
    
    # Valores base (melhores da otimização anterior)
    body_pct_values = [0, 45, 50]
    close_position_values = [0]
    candle_size_values = [1.5, 2.0]
    
    # NOVO: Take Profit multiplier (0 = usar MA para sair)
    take_profit_values = [0, 1.1, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]
    
    results = []
    total_combinations = (len(body_pct_values) * len(close_position_values) * 
                         len(candle_size_values) * len(take_profit_values))
    current = 0
    
    print(f"Total de combinações: {total_combinations}\n")
    
    for body_pct, close_pos, size_mult, tp_mult in product(
        body_pct_values, close_position_values, candle_size_values, take_profit_values):
        
        current += 1
        
        tp_label = f"{tp_mult}x" if tp_mult > 0 else "MA"
        print(f"[{current}/{total_combinations}] Body:{body_pct}% | Size:{size_mult}x | TP:{tp_label:4} ... ", end='')
        
        bt = BTCBacktest(
            timeframe='1d',
            ma_period=8,
            initial_capital=10000,
            body_pct_min=body_pct,
            close_position_min=close_pos,
            candle_size_multiplier=size_mult,
            take_profit_multiplier=tp_mult
        )
        
        df_test = df_data.copy()
        df_test = bt.calculate_candle_metrics(df_test)
        df_test = bt.calculate_ma(df_test)
        bt.run_backtest(df_test)
        
        metrics, trades, equity = bt.calculate_metrics()
        
        if metrics['total_trades'] >= 10:
            results.append({
                'body_pct': body_pct,
                'close_position': close_pos,
                'candle_size_mult': size_mult,
                'take_profit_mult': tp_mult,
                **metrics
            })
            bt.print_results(metrics, show_full=False)
        else:
            print(f"Poucos trades ({metrics['total_trades']}) - Descartado")
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("\nNenhuma combinação válida encontrada!")
        return None
    
    # Salvar resultados completos
    os.makedirs('results/optimization', exist_ok=True)
    df_results.to_csv('results/optimization/full_optimization.csv', index=False)
    
    # Criar múltiplas visualizações
    print("\n" + "="*70)
    print("TOP 10 CONFIGURAÇÕES (por Win Rate)")
    print("="*70)
    
    df_by_winrate = df_results.sort_values('win_rate', ascending=False)
    print(f"{'#':<3} {'Body%':<7} {'Size':<6} {'TP':<6} {'Trades':<7} {'WinRate':<9} "
          f"{'Exp%':<8} {'Return%':<12} {'PF':<6} {'DD%':<7}")
    print("-"*70)
    
    for idx, row in df_by_winrate.head(10).iterrows():
        tp_label = f"{row['take_profit_mult']:.1f}x" if row['take_profit_mult'] > 0 else "MA"
        print(f"{df_by_winrate.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<7.0f} "
              f"{row['candle_size_mult']:<6.1f} "
              f"{tp_label:<6} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<9.1f} "
              f"{row['expectancy_pct']:<8.2f} "
              f"{row['total_return']:<12.2f} "
              f"{row['profit_factor']:<6.2f} "
              f"{row['max_drawdown']:<7.2f}")
    
    print("\n" + "="*70)
    print("TOP 10 CONFIGURAÇÕES (por Expectância %)")
    print("="*70)
    
    df_by_exp = df_results.sort_values('expectancy_pct', ascending=False)
    print(f"{'#':<3} {'Body%':<7} {'Size':<6} {'TP':<6} {'Trades':<7} {'WinRate':<9} "
          f"{'Exp%':<8} {'Return%':<12} {'PF':<6} {'DD%':<7}")
    print("-"*70)
    
    for idx, row in df_by_exp.head(10).iterrows():
        tp_label = f"{row['take_profit_mult']:.1f}x" if row['take_profit_mult'] > 0 else "MA"
        print(f"{df_by_exp.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<7.0f} "
              f"{row['candle_size_mult']:<6.1f} "
              f"{tp_label:<6} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<9.1f} "
              f"{row['expectancy_pct']:<8.2f} "
              f"{row['total_return']:<12.2f} "
              f"{row['profit_factor']:<6.2f} "
              f"{row['max_drawdown']:<7.2f}")
    
    print("\n" + "="*70)
    print("TOP 10 CONFIGURAÇÕES (por Profit Factor)")
    print("="*70)
    
    df_by_pf = df_results.sort_values('profit_factor', ascending=False)
    print(f"{'#':<3} {'Body%':<7} {'Size':<6} {'TP':<6} {'Trades':<7} {'WinRate':<9} "
          f"{'Exp%':<8} {'Return%':<12} {'PF':<6} {'DD%':<7}")
    print("-"*70)
    
    for idx, row in df_by_pf.head(10).iterrows():
        tp_label = f"{row['take_profit_mult']:.1f}x" if row['take_profit_mult'] > 0 else "MA"
        print(f"{df_by_pf.index.get_loc(idx)+1:<3} "
              f"{row['body_pct']:<7.0f} "
              f"{row['candle_size_mult']:<6.1f} "
              f"{tp_label:<6} "
              f"{row['total_trades']:<7.0f} "
              f"{row['win_rate']:<9.1f} "
              f"{row['expectancy_pct']:<8.2f} "
              f"{row['total_return']:<12.2f} "
              f"{row['profit_factor']:<6.2f} "
              f"{row['max_drawdown']:<7.2f}")
    
    # Melhor configuração (por expectância, mais confiável que win rate puro)
    best = df_by_exp.iloc[0]
    
    print("\n" + "="*70)
    print("MELHOR CONFIGURAÇÃO (por Expectância)")
    print("="*70)
    print(f"Body % mínimo:          {best['body_pct']:.0f}%")
    print(f"Candle Size multiplier: {best['candle_size_mult']:.1f}x")
    
    if best['take_profit_mult'] > 0:
        print(f"Take Profit:            {best['take_profit_mult']:.1f}x o risco")
    else:
        print(f"Take Profit:            Seguir MA (sem alvo fixo)")
    
    print(f"\nWin Rate:               {best['win_rate']:.2f}%")
    print(f"Expectância:            {best['expectancy_pct']:.2f}% por trade")
    print(f"Total Trades:           {best['total_trades']:.0f}")
    print(f"Retorno Total:          {best['total_return']:.2f}%")
    print(f"Profit Factor:          {best['profit_factor']:.2f}")
    print(f"Max Drawdown:           {best['max_drawdown']:.2f}%")
    print("="*70)
    
    return best


def main():
    print("="*70)
    print("BACKTEST BTC - OTIMIZAÇÃO COMPLETA COM TAKE PROFIT")
    print("="*70)
    
    print("\nBaixando dados para otimização...")
    bt_temp = BTCBacktest(timeframe='1d', ma_period=8, initial_capital=10000)
    df_daily = bt_temp.download_data(years=15)
    print(f"✓ Dados baixados: {len(df_daily)} candles")
    
    # Otimizar filtros
    best_config = optimize_filters(df_daily)
    
    if best_config is not None:
        print("\n" + "="*70)
        print("EXECUTANDO BACKTEST COM MELHOR CONFIGURAÇÃO")
        print("="*70)
        
        bt_best = BTCBacktest(
            timeframe='1d',
            ma_period=8,
            initial_capital=10000,
            body_pct_min=best_config['body_pct'],
            close_position_min=best_config['close_position'],
            candle_size_multiplier=best_config['candle_size_mult'],
            take_profit_multiplier=best_config['take_profit_mult']
        )
        
        df_daily_best = df_daily.copy()
        df_daily_best = bt_best.calculate_candle_metrics(df_daily_best)
        df_daily_best = bt_best.calculate_ma(df_daily_best)
        bt_best.run_backtest(df_daily_best)
        
        metrics_best, trades_best, equity_best = bt_best.calculate_metrics()
        bt_best.print_results(metrics_best, show_full=True)
        bt_best.plot_results(df_daily_best, equity_best, trades_best, 'daily_optimized')
        bt_best.save_trades_csv(trades_best, 'daily_optimized')
        bt_best.save_summary(metrics_best, 'daily_optimized', {
            'body_pct': best_config['body_pct'],
            'close_pos': best_config['close_position'],
            'size_mult': best_config['candle_size_mult'],
            'tp_mult': best_config['take_profit_mult']
        })
    
    # Backtest semanal
    print("\n\n" + "="*70)
    print("TIMEFRAME SEMANAL (W1) - SEM FILTROS")
    print("="*70)
    
    bt_w1 = BTCBacktest(timeframe='1wk', ma_period=8, initial_capital=10000)
    df_w1 = bt_w1.download_data(years=15)
    df_w1 = bt_w1.calculate_ma(df_w1)
    bt_w1.run_backtest(df_w1)
    metrics_w1, trades_w1, equity_w1 = bt_w1.calculate_metrics()
    bt_w1.print_results(metrics_w1)
    bt_w1.plot_results(df_w1, equity_w1, trades_w1, 'weekly')
    bt_w1.save_trades_csv(trades_w1, 'weekly')
    bt_w1.save_summary(metrics_w1, 'weekly')
    
    print("\n" + "="*70)
    print("✓ BACKTEST CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print("\nResultados salvos em:")
    print("- results/daily_optimized/  (timeframe diário otimizado)")
    print("- results/weekly/           (timeframe semanal)")
    print("- results/optimization/     (análise completa de otimização)")


if __name__ == "__main__":
    main()
