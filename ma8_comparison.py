import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class MA8Backtest:
    def __init__(self, exit_mode='simple'):
        """
        exit_mode: 
            'simple' -> Sai no fechamento quando a média vira para baixo.
            'breakout' -> Sai apenas se romper a mínima do candle que virou para baixo.
        """
        self.ticker = 'BTC-USD'
        self.exit_mode = exit_mode
        self.capital = 10000
        self.initial_capital = 10000
        self.position = None
        self.trades = []
        
    def download_data(self):
        print(f"Baixando dados históricos de {self.ticker} via Yahoo Finance...")
        # Baixa o máximo de histórico disponível (BTC-USD no Yahoo vai até ~2014)
        df = yf.download(self.ticker, period="max", interval="1d", progress=False, auto_adjust=True)
        
        if len(df) == 0:
            raise Exception("Erro ao baixar dados.")
            
        print(f"Dados obtidos: {len(df)} dias (De {df.index[0].date()} até {df.index[-1].date()})")
        return df[['Open', 'High', 'Low', 'Close']]

    def prepare_indicators(self, df):
        df['MA8'] = df['Close'].rolling(window=8).mean()
        
        # Detectar viradas
        # 1 = Virou pra Cima, -1 = Virou pra Baixo, 0 = Sem mudança
        df['MA_Dir'] = 0
        
        # Logica: MA atual > MA anterior (Cima)
        df.loc[df['MA8'] > df['MA8'].shift(1), 'MA_Dir'] = 1
        df.loc[df['MA8'] < df['MA8'].shift(1), 'MA_Dir'] = -1
        
        # Identificar o candle EXATO da virada
        # Virou pra Cima: Candle atual é 1, anterior não era 1
        df['Turn_Up'] = (df['MA_Dir'] == 1) & (df['MA_Dir'].shift(1) != 1)
        
        # Virou pra Baixo: Candle atual é -1, anterior não era -1
        df['Turn_Down'] = (df['MA_Dir'] == -1) & (df['MA_Dir'].shift(1) != -1)
        
        return df

    def run(self, df):
        self.trades = []
        self.position = None
        self.capital = self.initial_capital
        
        # Gatilhos pendentes
        buy_trigger = None   # Preço de entrada
        stop_loss = None     # Stop Loss inicial
        sell_trigger = None  # Para modo 'breakout'
        
        # Iterar linha a linha
        for i in range(10, len(df)):
            idx = df.index[i]
            prev_idx = df.index[i-1]
            
            curr_open = df['Open'].iloc[i]
            curr_high = df['High'].iloc[i]
            curr_low = df['Low'].iloc[i]
            curr_close = df['Close'].iloc[i]
            
            # Dados do candle ANTERIOR (onde o sinal acontece)
            prev_turn_up = df['Turn_Up'].iloc[i-1]
            prev_turn_down = df['Turn_Down'].iloc[i-1]
            prev_high = df['High'].iloc[i-1]
            prev_low = df['Low'].iloc[i-1]
            
            # ----------------------------------------
            # GESTÃO DE POSIÇÃO (SAÍDA)
            # ----------------------------------------
            if self.position:
                # 1. Verificar STOP LOSS (Prioridade máxima)
                # O Stop Loss foi definido na entrada (mínima do candle de sinal)
                if curr_low <= self.position['stop_price']:
                    # Assume slippage zero, sai no stop exato (ou abertura se abriu abaixo)
                    exit_price = min(curr_open, self.position['stop_price'])
                    self._close_trade(idx, exit_price, 'Stop Loss')
                    buy_trigger = None # Cancela qualquer reentrada imediata no mesmo candle
                    continue

                # 2. Verificar GATILHO DE SAÍDA (Modo Breakout)
                if self.exit_mode == 'breakout' and sell_trigger:
                    # Desarma se MA virou pra cima de novo
                    if prev_turn_up:
                        sell_trigger = None
                    # Executa venda se romper mínima
                    elif curr_low <= sell_trigger:
                        exit_price = min(curr_open, sell_trigger)
                        self._close_trade(idx, exit_price, 'Exit Trigger')
                        continue

                # 3. Lógica de Sinal de Saída
                if prev_turn_down:
                    if self.exit_mode == 'simple':
                        # Sai no fechamento do candle atual que confirmou a virada para baixo no anterior? 
                        # NÃO. Se virou no anterior (prev_turn_down), a gente já sabe na abertura deste.
                        # Setup clássico Larry Williams/MA: Virou pra baixo, sai no FECHAMENTO ou ABERTURA?
                        # Pedido: "média simplesmente virar para baixo". Normalmente isso executa no Close do candle que virou 
                        # ou Open do seguinte. Vou usar Open do seguinte para ser realista (backtest não pode olhar futuro).
                        
                        self._close_trade(idx, curr_open, 'MA Turn Down')
                        
                    elif self.exit_mode == 'breakout':
                        # Arma gatilho na mínima do candle que virou (prev_low)
                        sell_trigger = prev_low

            # ----------------------------------------
            # GESTÃO DE ENTRADA
            # ----------------------------------------
            else: # Não posicionado
                
                # Se tínhamos um gatilho de compra armado do candle anterior
                if buy_trigger:
                    # Verifica rompimento
                    if curr_high > buy_trigger:
                        # Compra
                        entry_price = max(curr_open, buy_trigger)
                        self._open_trade(idx, entry_price, stop_loss)
                        buy_trigger = None # Consumiu o gatilho
                    else:
                        # Se não rompeu, verifica se o setup ainda é válido.
                        # Normalmente setups de gatilho valem apenas para o candle seguinte.
                        # Se não acionou, cancela.
                        buy_trigger = None
                        stop_loss = None

                # Verifica se gerou NOVO sinal de entrada (para o próximo candle)
                if prev_turn_up:
                    buy_trigger = prev_high
                    stop_loss = prev_low

    def _open_trade(self, date, price, stop):
        qty = self.capital / price
        self.position = {
            'entry_date': date,
            'entry_price': price,
            'qty': qty,
            'stop_price': stop
        }

    def _close_trade(self, date, price, reason):
        pnl = (price - self.position['entry_price']) * self.position['qty']
        pnl_pct = (price / self.position['entry_price']) - 1
        duration = (date - self.position['entry_date']).days
        
        self.capital += pnl
        self.trades.append({
            'entry_date': self.position['entry_date'],
            'exit_date': date,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'reason': reason,
            'duration': duration
        })
        self.position = None

    def get_metrics(self):
        if not self.trades: return None
        
        df_t = pd.DataFrame(self.trades)
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        
        return {
            'Total Trades': len(df_t),
            'Win Rate': (len(wins) / len(df_t)) * 100,
            'Avg Duration': df_t['duration'].mean(),
            'Avg Win %': wins['pnl_pct'].mean() if not wins.empty else 0,
            'Avg Loss %': losses['pnl_pct'].mean() if not losses.empty else 0,
            'Total Return %': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'Profit Factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0
        }

def main():
    print("="*80)
    print("COMPARATIVO: SETUP MA8 (DIÁRIO) - DADOS DESDE 2014")
    print("="*80)
    
    # 1. Carregar dados uma vez
    bt_loader = MA8Backtest()
    df = bt_loader.download_data()
    df = bt_loader.prepare_indicators(df)
    
    # 2. Teste Saída 1 (Simples)
    bt_simple = MA8Backtest(exit_mode='simple')
    bt_simple.run(df)
    res_simple = bt_simple.get_metrics()
    
    # 3. Teste Saída 2 (Breakout)
    bt_breakout = MA8Backtest(exit_mode='breakout')
    bt_breakout.run(df)
    res_breakout = bt_breakout.get_metrics()
    
    # 4. Exibir Comparativo
    print("\n" + "="*80)
    print(f"{'MÉTRICA':<25} | {'SAÍDA 1 (Simples)':<20} | {'SAÍDA 2 (Rompimento)':<20}")
    print("-" * 80)
    
    metrics = [
        ('Total Trades', 'Total Trades', '{:.0f}'),
        ('Taxa de Acerto', 'Win Rate', '{:.2f}%'),
        ('Tempo Médio (dias)', 'Avg Duration', '{:.1f} dias'),
        ('Média de Ganho', 'Avg Win %', '{:.2f}%'),
        ('Média de Perda', 'Avg Loss %', '{:.2f}%'),
        ('Retorno Total', 'Total Return %', '{:.2f}%'),
        ('Profit Factor', 'Profit Factor', '{:.2f}')
    ]
    
    for label, key, fmt in metrics:
        val1 = res_simple[key]
        val2 = res_breakout[key]
        print(f"{label:<25} | {fmt.format(val1):<20} | {fmt.format(val2):<20}")
    
    print("="*80)
    print("\nLEGENDA:")
    print("Saída 1 (Simples): Vende na abertura assim que a MA8 vira para baixo.")
    print("Saída 2 (Rompimento): Arma gatilho na mínima do candle que virou. Só vende se perder essa mínima.")

if __name__ == "__main__":
    main()
