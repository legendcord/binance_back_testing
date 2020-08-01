from binance_data.client import DataClient
store_data = DataClient().kline_data(['ETHUSDT'], '1m', storage=['csv','/home/haha/Desktop/back_test/binance_back_testing/'],progress_statements=True)
#store_data = DataClient().kline_data(['BTCUSDT'], '1m', storage=['csv','/home/haha/Desktop/back_test/binance_back_testing/'],progress_statements=True)
