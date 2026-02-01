# BTC Backtest - Estratégia Média Móvel 8

Backtest automatizado de estratégia long-only em Bitcoin usando média móvel de 8 períodos.

## Estratégia

### Entrada
- Quando a MA8 vira para cima, arma gatilho na máxima do candle que virou
- Entra ao romper a máxima
- Stop loss na mínima do candle de entrada

### Saída
- Quando a MA8 vira para baixo, arma gatilho na mínima do candle que virou
- Sai ao romper a mínima
- Desarma a saída se a MA8 virar para cima novamente
- Stop loss sempre ativo

## Timeframes
- D1 (Diário)
- W1 (Semanal)

## Como usar

### Local
```bash
pip install -r requirements.txt
python backtest.py
