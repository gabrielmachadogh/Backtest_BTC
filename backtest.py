import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import requests

class BTCBacktest:
    def __init__(self, timeframe='1d', ma_period=8, initial_capital=10000):
        self.timeframe = timeframe
        self.ma_period = ma_period
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    def download_from_cryptocompare(self, days):
        """Baixa dados do CryptoCompare API (gratuito, sem necessidade de API key)"""
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
                    print(f"  Erro na API: {data.get('Message', 'Unknown')}")
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
        """Baixa dados históricos do BTC com múltiplas fontes"""
        print(f"Baixando dados BTC ({self.timeframe}) dos últimos {years} anos...")
        
        days = years * 365
        df = None
        
        try:
            df = self.download_from_binance(days)
            if df is not None and len(df) > 100:
                print(f"✓ Dados baixados da Binance: {len(df)} candles")
                print(f"  Período: {df.index[0]} até {df.index[-1]}")
                return df
        except Exception as e:
            print(f"  Binance falhou: {str(e)}")
        
        try:
            df = self.download_from_cryptocompare(days)
            if df is not None and len(df) > 100:
                print(f"✓ Dados baixados do CryptoCompare: {len(df)} candles")
                print(f"  Período: {df.index[0]} até {df.index[-1]}")
                return df
        except Exception as e:
            print(f"  CryptoCompare falhou: {str(e)}")
        
        try:
            import yfinance as yf
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            for ticker in ["BTC-USD", "BTCUSD=X"]:
                try:
                    print(f"  Tentando yfinance com {ticker}...")
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
                        print(f"✓ Dados baixados do yfinance: {len(df)} candles")
                        break
                except Exception as e:
                    print(f"    {ticker} falhou: {str(e)}")
                    continue
        except Exception as e:
            print(f"  yfinance falhou: {str(e)}")
        
        if df is None or df.empty:
            raise Exception("Não foi possível baixar dados de nenhuma fonte")
        
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
    
    def run_backtest(self, df):
        """Executa o backtest com a estratégia"""
        print("\nExecutando backtest...")
        
        entry_trigger = None
        exit_trigger = None
        stop_loss = None
        
        for i in range(self.ma_period + 1, len(df)):
            current_idx = df.index[i]
            current_price = df.loc[current_idx, 'High']
            current_low = df.loc[current_idx, 'Low']
            
            if self.position is None:
                if df.loc[df.index[i-1], 'MA_Turn_Up']:
                    entry_trigger = df.loc[df.index[i-1], 'High']
                    stop_loss = df.loc[df.index[i-1], 'Low']
                
                if entry_trigger and current_price >= entry_trigger:
                    quantity = self.capital / entry_trigger
                    self.position = {
                        'entry_date': current_idx,
                        'entry_price': entry_trigger,
                        'quantity': quantity,
                        'stop_loss': stop_loss
                    }
                    entry_trigger = None
                    
            else:
                if current_low <= self.position['stop_loss']:
                    exit_price = self.position['stop_loss']
                    self.close_position(current_idx, exit_price, 'Stop Loss')
                    exit_trigger = None
                    continue
                
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
        
        print(f"\nBacktest finalizado! Total de trades: {len(self.trades)}")
    
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
        
        # Drawdown
        df_equity['cummax'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['cummax']) / df_equity['cummax'] * 100
        max_drawdown = df_equity['drawdown'].min()
        
        # Métricas adicionais
        df_trades['duration'] = (df_trades['exit_date'] - df_trades['entry_date']).dt.days
        
        # Sharpe Ratio (simplificado - usando retornos diários)
        returns = df_equity['equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        # Tempo em mercado
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
    
    def print_results(self, metrics):
        """Imprime resultados do backtest"""
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
    
    def plot_results(self, df, df_equity, df_trades, timeframe_name):
        """Gera gráficos dos resultados"""
        output_dir = f'results/{timeframe_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Gráfico interativo
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=('Preço BTC e Média Móvel 8', 'Equity Curve', 'Drawdown'),
                           row_heights=[0.5, 0.3, 0.2])
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='BTC'),
                     row=1, col=1)
        
        # Média Móvel
        fig.add_trace(go.Scatter(x=df.index, y=df['MA8'],
                                mode='lines',
                                name='MA8',
                                line=dict(color='orange', width=2)),
                     row=1, col=1)
        
        if not df_trades.empty:
            # Sinais de entrada
            entries = df_trades[['entry_date', 'entry_price']].copy()
            fig.add_trace(go.Scatter(x=entries['entry_date'],
                                    y=entries['entry_price'],
                                    mode='markers',
                                    name='Entrada',
                                    marker=dict(color='green', size=10, symbol='triangle-up')),
                         row=1, col=1)
            
            # Sinais de saída
            exits = df_trades[['exit_date', 'exit_price']].copy()
            fig.add_trace(go.Scatter(x=exits['exit_date'],
                                    y=exits['exit_price'],
                                    mode='markers',
                                    name='Saída',
                                    marker=dict(color='red', size=10, symbol='triangle-down')),
                         row=1, col=1)
        
        # Equity Curve
        fig.add_trace(go.Scatter(x=df_equity['date'],
                                y=df_equity['equity'],
                                mode='lines',
                                name='Equity',
                                line=dict(color='blue', width=2),
                                fill='tozeroy'),
                     row=2, col=1)
        
        # Drawdown
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
            yaxis_title='Preço (USD)',
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=2, col=1)
        
        fig.write_html(f'{output_dir}/backtest_chart.html')
        print(f"\nGráfico salvo em: {output_dir}/backtest_chart.html")
        
        if df_trades.empty:
            print("Nenhum trade executado - pulando gráficos de métricas")
            return
        
        # Gráfico de análise
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Distribuição de retornos percentuais
        ax1.hist(df_trades['pnl_pct'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_title('Distribuição de Retornos (%)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Retorno (%)')
        ax1.set_ylabel('Frequência')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.axvline(x=df_trades['pnl_pct'].mean(), color='g', linestyle='--', linewidth=2, label=f"Média: {df_trades['pnl_pct'].mean():.1f}%")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PnL por trade
        colors = ['green' if x > 0 else 'red' for x in df_trades['pnl_pct']]
        ax2.bar(range(len(df_trades)), df_trades['pnl_pct'], color=colors, alpha=0.7)
        ax2.set_title('Retorno por Trade (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Trade #')
        ax2.set_ylabel('Retorno (%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Evolução do Capital (escala log)
        ax3.plot(df_trades.index, df_trades['capital'], linewidth=2, color='blue', marker='o', markersize=3)
        ax3.set_title('Evolução do Capital', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trade #')
        ax3.set_ylabel('Capital ($)')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, which='both')
        ax3.fill_between(df_trades.index, df_trades['capital'], alpha=0.3)
        
        # Duração vs Retorno
        ax4.scatter(df_trades['duration'], df_trades['pnl_pct'], 
                   c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        ax4.set_title('Duração vs Retorno', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Duração (dias)')
        ax4.set_ylabel('Retorno (%)')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics.png', dpi=300, bbox_inches='tight')
        print(f"Gráfico de métricas salvo em: {output_dir}/metrics.png")
        plt.close()
        
    def save_trades_csv(self, df_trades, timeframe_name):
        """Salva trades em CSV"""
        output_dir = f'results/{timeframe_name}'
        os.makedirs(output_dir, exist_ok=True)
        df_trades.to_csv(f'{output_dir}/trades.csv', index=False)
        print(f"Trades salvos em: {output_dir}/trades.csv")
    
    def save_summary(self, metrics, timeframe_name):
        """Salva resumo em arquivo de texto"""
        output_dir = f'results/{timeframe_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/summary.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"BACKTEST BTC - ESTRATÉGIA MA8 ({self.timeframe})\n")
            f.write("="*70 + "\n\n")
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
        
        print(f"Resumo salvo em: {output_dir}/summary.txt")


def main():
    print("="*70)
    print("BACKTEST BTC - ESTRATÉGIA MÉDIA MÓVEL 8")
    print("="*70)
    
    # Backtest D1
    print("\n### TIMEFRAME DIÁRIO (D1) ###")
    bt_d1 = BTCBacktest(timeframe='1d', ma_period=8, initial_capital=10000)
    df_d1 = bt_d1.download_data(years=15)
    df_d1 = bt_d1.calculate_ma(df_d1)
    bt_d1.run_backtest(df_d1)
    metrics_d1, trades_d1, equity_d1 = bt_d1.calculate_metrics()
    bt_d1.print_results(metrics_d1)
    bt_d1.plot_results(df_d1, equity_d1, trades_d1, 'daily')
    bt_d1.save_trades_csv(trades_d1, 'daily')
    bt_d1.save_summary(metrics_d1, 'daily')
    
    # Backtest W1
    print("\n\n### TIMEFRAME SEMANAL (W1) ###")
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
    print("- results/daily/  (timeframe diário)")
    print("- results/weekly/ (timeframe semanal)")
    print("\nArquivos gerados:")
    print("  • backtest_chart.html  (gráfico interativo)")
    print("  • metrics.png          (análise visual)")
    print("  • trades.csv           (lista de trades)")
    print("  • summary.txt          (resumo textual)")


if __name__ == "__main__":
    main()
