import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class BTCBacktest:
    def __init__(self, timeframe='1d', ma_period=8, initial_capital=10000):
        self.timeframe = timeframe
        self.ma_period = ma_period
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    def download_data(self, years=15):
        """Baixa dados históricos do BTC"""
        print(f"Baixando dados BTC ({self.timeframe}) dos últimos {years} anos...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        ticker = "BTC-USD"
        df = yf.download(ticker, start=start_date, end=end_date, interval=self.timeframe, progress=False)
        
        if df.empty:
            raise Exception("Erro ao baixar dados")
        
        print(f"Dados baixados: {len(df)} candles de {df.index[0]} até {df.index[-1]}")
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
                    print(f"\n[{current_idx}] MA virou para CIMA - Gatilho de entrada: ${entry_trigger:.2f}, Stop: ${stop_loss:.2f}")
                
                # Tenta entrar se romper o gatilho
                if entry_trigger and current_price >= entry_trigger:
                    quantity = self.capital / entry_trigger
                    self.position = {
                        'entry_date': current_idx,
                        'entry_price': entry_trigger,
                        'quantity': quantity,
                        'stop_loss': stop_loss
                    }
                    print(f"[{current_idx}] ENTRADA em ${entry_trigger:.2f} | Stop: ${stop_loss:.2f} | Quantidade: {quantity:.6f} BTC")
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
                    print(f"\n[{current_idx}] MA virou para BAIXO - Gatilho de saída: ${exit_trigger:.2f}")
                
                # Desarma saída se MA virar para cima novamente
                if df.loc[df.index[i-1], 'MA_Turn_Up'] and exit_trigger:
                    print(f"[{current_idx}] MA virou para CIMA novamente - DESARMANDO saída")
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
        print(f"[{exit_date}] SAÍDA em ${exit_price:.2f} | Motivo: {reason} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%) | Capital: ${self.capital:.2f}")
        
        self.position = None
    
    def calculate_metrics(self):
        """Calcula métricas de performance"""
        if not self.trades:
            return {}
        
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
    
    def plot_results(self, df, df_equity, df_trades):
        """Gera gráficos dos resultados"""
        os.makedirs('results', exist_ok=True)
        
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
        
        fig.write_html('results/backtest_chart.html')
        print("\nGráfico salvo em: results/backtest_chart.html")
        
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
        plt.savefig('results/metrics.png', dpi=300, bbox_inches='tight')
        print("Gráfico de métricas salvo em: results/metrics.png")
        
    def save_trades_csv(self, df_trades):
        """Salva trades em CSV"""
        df_trades.to_csv('results/trades.csv', index=False)
        print("Trades salvos em: results/trades.csv")


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
    bt_d1.plot_results(df_d1, equity_d1, trades_d1)
    bt_d1.save_trades_csv(trades_d1)
    
    # Backtest W1 (semanal)
    print("\n\n### TIMEFRAME SEMANAL (W1) ###")
    bt_w1 = BTCBacktest(timeframe='1wk', ma_period=8, initial_capital=10000)
    df_w1 = bt_w1.download_data(years=15)
    df_w1 = bt_w1.calculate_ma(df_w1)
    bt_w1.run_backtest(df_w1)
    metrics_w1, trades_w1, equity_w1 = bt_w1.calculate_metrics()
    bt_w1.print_results(metrics_w1)
    
    # Salvar resultados W1 separadamente
    os.makedirs('results/weekly', exist_ok=True)
    
    # Renomear arquivos D1
    os.rename('results/backtest_chart.html', 'results/daily/backtest_chart_D1.html')
    os.rename('results/metrics.png', 'results/daily/metrics_D1.png')
    os.rename('results/trades.csv', 'results/daily/trades_D1.csv')
    
    # Criar diretório daily
    os.makedirs('results/daily', exist_ok=True)
    
    # Mover arquivos D1
    import shutil
    if os.path.exists('results/backtest_chart.html'):
        shutil.move('results/backtest_chart.html', 'results/daily/backtest_chart_D1.html')
    if os.path.exists('results/metrics.png'):
        shutil.move('results/metrics.png', 'results/daily/metrics_D1.png')
    if os.path.exists('results/trades.csv'):
        shutil.move('results/trades.csv', 'results/daily/trades_D1.csv')
    
    # Gerar gráficos W1
    bt_w1.plot_results(df_w1, equity_w1, trades_w1)
    bt_w1.save_trades_csv(trades_w1)
    
    # Mover arquivos W1
    shutil.move('results/backtest_chart.html', 'results/weekly/backtest_chart_W1.html')
    shutil.move('results/metrics.png', 'results/weekly/metrics_W1.png')
    shutil.move('results/trades.csv', 'results/weekly/trades_W1.csv')
    
    print("\n" + "="*60)
    print("BACKTEST CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print("\nResultados salvos em:")
    print("- results/daily/ (timeframe diário)")
    print("- results/weekly/ (timeframe semanal)")


if __name__ == "__main__":
    main()
