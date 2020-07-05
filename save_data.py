import os
import csv
import pytz
from datetime import datetime
import pandas as pd
import time

from catalyst.api import record, symbol, symbols, get_datetime
from catalyst.utils.run_algo import run_algorithm

symbol_str = 'eth_usdt'

def initialize(context):
    # Portfolio assets list
    context.asset = symbol(symbol_str)

    # Create an empty DataFrame to store results
    #context.pricing_data_1m = pd.DataFrame()
    #context.pricing_data_4h = pd.DataFrame()
    context.pricing_data_1h = pd.DataFrame()

    context.start_time = time.time()

def handle_data(context, data):
    # Variables to record for a given asset: price and volume
    # Other options include 'open', 'high', 'open', 'close'
    # Please note that 'price' equals 'close'
    context.current_1m = data.history(context.asset, ['open', 'high', 'low', 'close'], 1, '1T')    
    current_datetime = get_datetime()    
#    if current_datetime.hour % 4 == 0 and current_datetime.minute == 0 and current_datetime.second == 0:
#        context.current_4h = data.history(context.asset, ['open', 'high', 'low', 'close'], 1, '4H')
    if current_datetime.hour % 1 == 0 and current_datetime.minute == 0 and current_datetime.second == 0:
        context.current_1h = data.history(context.asset, ['open', 'high', 'low', 'close'], 1, '1H')
    if current_datetime.hour == 0 and current_datetime.minute == 0 and current_datetime.second == 0:        
        # Store some information daily
        print('\nCurrent date is ' + str(get_datetime().date()))
        print('elapsed time(minute): ' + str((time.time() - context.start_time) / 60))
        
    #context.pricing_data_1m = context.pricing_data_1m.append(context.current_1m)
    context.pricing_data_1h = context.pricing_data_1h.append(context.current_1h)

    '''
    print('====================================================================')
    print(context.current_1m)
    print(context.current_4h)
    '''

    

def analyze(context=None, results=None):
    print('saving data ...')
    # Save pricing data to a CSV file
    #filename = symbol_str + '_2020_1m'
    #context.pricing_data_1m.to_csv(filename + '.csv')
    filename = symbol_str + '_2020_1h'
    context.pricing_data_1h.to_csv(filename + '.csv')
    context.end_time = time.time()
    elapsed_min = (context.end_time - context.start_time) / 60
    print('elapsed time(minute): ' + str(elapsed_min))

run_algorithm(
    capital_base = 1,
    data_frequency = 'minute',
    initialize = initialize,
    handle_data = handle_data,
    analyze = analyze,
    #exchange_name = 'bitfinex',
    exchange_name = 'binance',
    quote_currency = 'usdt',
    start = pd.to_datetime('2020-1-1', utc = True),
    end = pd.to_datetime('2020-3-9', utc = True))
