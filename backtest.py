import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

class BTCBacktest:
    def __init__(self, timeframe='1d', ma_period=8, initial_capital=10000):
        self.timeframe = timeframe
        self.ma_period = ma_period
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    def download_data(self, years=15, max_retries=3):
        """Baixa dados históricos do BTC com retry"""
        print(f"Baixando dados BTC ({self.timeframe}) dos últimos {years} anos...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        # Lista de tickers alternativos
        tickers = ["BTC-USD", "BTCUSD=X"]
        
        df = None
        for ticker in tickers:
            for attempt in range(max_retries):
                try:
                    print(f"Tentativa {attempt + 1}/{max_retries} para {ticker}...")
                    
                    # Download com configurações específicas
                    data = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date, 
                        interval=self.timeframe,
                        progress=False,
                        auto_adjust=True,
                        prepost=False,
                        threads=True,
                        proxy=None
                    )
                    
                    if not data.empty and len(data) > 100:
                        df = data
                        print(f"✓ Dados baixados com sucesso usando {ticker}")
                        print(f"  {len(df)} candles de {df.index[0]} até {df.index[-1]}")
                        break
                    else:
                        print(f"  Dados insuficientes ou vazios")
                        
                except Exception as e:
                    print(f"  Erro: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"  Aguardando {wait_time}s antes de tentar novamente...")
                        time.sleep(wait_time)
            
            if df is not None and not df.empty:
                break
        
        if df is None or df.empty:
            raise Exception("Não foi possível baixar dados após todas as tentativas")
        
        # Renomear colunas se necessário
        df.columns = df.columns.str.strip()
        column_mapping = {
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }
        
        # Se as colunas já estão corretas
        if 'Close' in df.columns:
            return df
        
        # Se vieram em minúsculas ou com o ticker
        for col in df.columns:
            col_clean = col.split()[0] if ' ' in col else col
            for std_name in column_mapping.keys():
                if std_name.lower() in col.lower():
                    df.rename(columns={col: std_name}, inplace=True)
                    break
        
        return df
    
    def calculate_ma(self, df):
        """Calcula média móvel de 8 períodos"""
        df['MA8'] = df['Close'].rolling(window=self.ma_period).mean()
        
        # Detecta quando a MA vira para cima ou para baixo
        df['MA_Direction'] = 0
        df.loc[df['MA8'] > df['MA8'].shift(1), 'MA_Direction'] = 1  # Virando pra cima
        df.loc[df['MA8'] < df['MA8'].shift(1), 'MA_Direction'] = -1  # Virando pra baixo
        
        # Marca candle que virou
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
            
            # Verifica se não está em posição
            if self.position is None:
                # Detecta virada da MA para cima
                if df.loc[df.index[i-1], 'MA_Turn_Up']:
                    entry_trigger = df.loc[df.index[i-1], 'High']
                    stop_loss = df.loc[df.index[i-1], 'Low']
                    print(f"\n[{current_idx.strftime('%Y-%m-%d')}] MA virou para CIMA - Gatilho: ${entry_trigger:.2f}, Stop: ${stop_loss:.2f}")
                
                # Tenta entrar se romper o gatilho
                if entry_trigger and current_price >= entry_trigger:
                    quantity = self.capital / entry_trigger
                    self.position = {
                        'entry_date': current_idx,
                        'entry_price': entry_trigger,
                        'quantity': quantity,
                        'stop_loss': stop_loss
                    }
                    print(f"[{current_idx.strftime('%Y-%m-%d')}] ENTRADA ${entry_trigger:.2f} | Qtd: {quantity:.6f} BTC")
                    entry_trigger = None
                    
            # Verifica se está em posição
            else:
                # Verifica stop loss
                if current_low <= self.position['stop_loss']:
                    exit_price = self.position['stop_loss']
                    self.close_position(current_idx, exit_price, 'Stop Loss')
                    exit_trigger = None
                    continue
                
                # Detecta virada da MA para baixo
                if df.loc[df.index[i-1], 'MA_Turn_Down']:
                    exit_trigger = df.loc[df.index[i-1], 'Low']
                    print(f"\n[{current_idx.strftime('%Y-%m-%d')}] MA virou para BAIXO - Gatilho saída: ${exit_trigger:.2f}")
                
                # Desarma saída se MA virar para cima novamente
                if df.loc[df.index[i-1], 'MA_Turn_Up'] and exit_trigger:
                    print(f"[{current_idx.strftime('%Y-%m-%d')}] MA virou CIMA - DESARMANDO saída")
                    exit_trigger = None
                
                # Tenta sair se atingir o gatilho
                if exit_trigger and current_low <= exit_trigger:
                    self.close_position(current_idx, exit_trigger, 'Gatilho MA')
                    exit_trigger = None
            
            # Registra equity
            if self.position:
                current_equity = self.position['quantity'] * df.loc[current_idx, 'Close']
            else:
                current_equity = self.capital
            
            self.equity_curve.append({
                'date': current_idx,
                'equity': current_equity
            })
        
        # Fecha posição final se ainda estiver aberta
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
        print(f"[{exit_date.strftime('%Y-%m-%d')}] SAÍDA ${exit_price:.2f} | {reason} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
        
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
            'largest_win': df_trades['pnl'].max(),
            'largest_loss': df_trades['pnl'].min(),
            'max_drawdown': max_drawdown,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        }
        
        return metrics, df_trades, df_equity
    
    def print_results(self, metrics):
        """Imprime resultados do backtest"""
        print("\n" + "="*60)
        print("RESULTADOS DO BACKTEST")
        print("="*60)
        print(f"Capital Inicial: ${metrics['initial_capital']:,.2f}")
        print(f"Capital Final: ${metrics['final_capital']:,.2f}")
        print(f"Retorno Total: {metrics['total_return']:.2f}%")
        print(f"\nTotal de Trades: {metrics['total_trades']}")
        print(f"Trades Vencedores: {metrics['winning_trades']}")
        print(f"Trades Perdedores: {metrics['losing_trades']}")
        print(f"Taxa de Acerto: {metrics['win_rate']:.2f}%")
        print(f"\nGanho Médio: ${metrics['avg_win']:,.2f}")
        print(f"Perda Média: ${metrics['avg_loss']:,.2f}")
        print(f"Maior Ganho: ${metrics['largest_win']:,.2f}")
        print(f"Maior Perda: ${metrics['largest_loss']:,.2f}")
        print(f"\nMax Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print("="*60)
    
    def plot_results(self, df, df_equity, df_trades, timeframe_name):
        """Gera gráficos dos resultados"""
        output_dir = f'results/{timeframe_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Gráfico 1: Preço e MA com sinais
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=('Preço BTC e Média Móvel 8', 'Equity Curve'),
                           row_heights=[0.7, 0.3])
        
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
                                line=dict(color='blue', width=2)),
                     row=2, col=1)
        
        fig.update_layout(
            title=f'Backtest BTC - Estratégia MA8 ({self.timeframe})',
            xaxis_title='Data',
            yaxis_title='Preço (USD)',
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.write_html(f'{output_dir}/backtest_chart.html')
        print(f"\nGráfico salvo em: {output_dir}/backtest_chart.html")
        
        if df_trades.empty:
            print("Nenhum trade executado - pulando gráficos de métricas")
            return
        
        # Gráfico 2: Distribuição de trades
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribuição de PnL
        ax1.hist(df_trades['pnl'], bins=30, edgecolor='black', alpha=0.7)
        ax1.set_title('Distribuição de PnL')
        ax1.set_xlabel('PnL ($)')
        ax1.set_ylabel('Frequência')
        ax1.axvline(x=0, color='r', linestyle='--')
        
        # PnL por trade
        colors = ['green' if x > 0 else 'red' for x in df_trades['pnl']]
        ax2.bar(range(len(df_trades)), df_trades['pnl'], color=colors, alpha=0.7)
        ax2.set_title('PnL por Trade')
        ax2.set_xlabel('Trade #')
        ax2.set_ylabel('PnL ($)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Capital acumulado
        ax3.plot(df_trades['capital'], linewidth=2, color='blue')
        ax3.set_title('Evolução do Capital')
        ax3.set_xlabel('Trade #')
        ax3.set_ylabel('Capital ($)')
        ax3.grid(True, alpha=0.3)
        
        # Drawdown
        ax4.fill_between(range(len(df_equity)), df_equity['drawdown'], 0, 
                         color='red', alpha=0.3)
        ax4.set_title('Drawdown')
        ax4.set_xlabel('Período')
        ax4.set_ylabel('Drawdown (%)')
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


def main():
    print("="*60)
    print("BACKTEST BTC - ESTRATÉGIA MÉDIA MÓVEL 8")
    print("="*60)
    
    # Backtest D1 (diário)
    print("\n### TIMEFRAME DIÁRIO (D1) ###")
    bt_d1 = BTCBacktest(timeframe='1d', ma_period=8, initial_capital=10000)
    df_d1 = bt_d1.download_data(years=15)
    df_d1 = bt_d1.calculate_ma(df_d1)
    bt_d1.run_backtest(df_d1)
    metrics_d1, trades_d1, equity_d1 = bt_d1.calculate_metrics()
    bt_d1.print_results(metrics_d1)
    bt_d1.plot_results(df_d1, equity_d1, trades_d1, 'daily')
    bt_d1.save_trades_csv(trades_d1, 'daily')
    
    # Backtest W1 (semanal)
    print("\n\n### TIMEFRAME SEMANAL (W1) ###")
    bt_w1 = BTCBacktest(timeframe='1wk', ma_period=8, initial_capital=10000)
    df_w1 = bt_w1.download_data(years=15)
    df_w1 = bt_w1.calculate_ma(df_w1)
    bt_w1.run_backtest(df_w1)
    metrics_w1, trades_w1, equity_w1 = bt_w1.calculate_metrics()
    bt_w1.print_results(metrics_w1)
    bt_w1.plot_results(df_w1, equity_w1, trades_w1, 'weekly')
    bt_w1.save_trades_csv(trades_w1, 'weekly')
    
    print("\n" + "="*60)
    print("BACKTEST CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print("\nResultados salvos em:")
    print("- results/daily/ (timeframe diário)")
    print("- results/weekly/ (timeframe semanal)")


if __name__ == "__main__":
    main()
    
