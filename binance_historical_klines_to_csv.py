from binance_data.client import DataClient
#store_data = DataClient().kline_data(['ETHUSDT'], '4h', storage=['csv','/home/crypto/Desktop/pivot_reversal_strategy/'],progress_statements=True)
store_data = DataClient().kline_data(['BTCUSDT'], '4h', storage=['csv','/home/crypto/Desktop/pivot_reversal_strategy/'],progress_statements=True)
