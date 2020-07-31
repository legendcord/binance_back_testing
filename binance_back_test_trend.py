from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data from file
#data_1m = pd.read_csv('1m_data/ETHUSDT/ETHUSDT2020.csv')
#data_hour = pd.read_csv('4h_data/ETHUSDT/ETHUSDT2020.csv')
data_1m = pd.read_csv('1m_data/ETHUSDT/ETHUSDT.csv')
hour_num = 4
hour_str = str(hour_num)+'h'
data_hour = pd.read_csv(hour_str+'_data/ETHUSDT/ETHUSDT.csv')

#data_1m = pd.read_csv('1m_data/BTCUSDT/BTCUSDT2020.csv')
#data_hour = pd.read_csv('4h_data/BTCUSDT/BTCUSDT2020.csv')
#data_1m = pd.read_csv('1m_data/BTCUSDT/BTCUSDT.csv')
#data_hour = pd.read_csv('4h_data/BTCUSDT/BTCUSDT.csv')

data_1m = data_1m.drop_duplicates()
# data_hour may contain duplicated rows
data_hour = data_hour.drop_duplicates()

[rows, columns] = data_1m.shape

#pv_enum = ['l6r1', 'l4r2', 'l2r6']
pv_enum = ['l4r2']

start_year = 2017

risk_control = False # risk control
adjust_leverage = False
P_val = 1/100
leverage = 3

current_pivot_high = None
current_pivot_low = None
PVH_updated = False
PVL_updated = False
pv_highs = {}
pv_lows = {}
pv_status = {}
previous_order_sizes = {}
last_short_prices = {}
last_long_prices = {}
for i in range(len(pv_enum)):
    index = pv_enum[i]
    pv_highs[index] = None
    pv_lows[index] = None
    pv_status[index] = 0
    previous_order_sizes[index] = 0
    last_short_prices[index] = None
    last_long_prices[index] = None

portfolio_value = 1
max_port = 1
leverage_adjust_co = 1
cumulative_return = 1
portfolio_curve = []
portfolio_curve.append(cumulative_return)
base_curve = []
base_curve.append(1)

def get_datetime(datetime_str):
    datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    return datetime_obj    

def is_pivot_high(high_val, adjacent_highs):
    for val in adjacent_highs:
        if high_val < val:
            return False
    return True

def is_pivot_low(low_val, adjacent_lows):
    for val in adjacent_lows:
        if low_val > val:
            return False
    return True

def get_current_pivot_points(recent_kline, left_bars, right_bars):
    global current_pivot_high, current_pivot_low, PVH_updated, PVL_updated
    
    PVH_updated = False
    PVL_updated = False
    
    past_highs = []
    past_lows = []
    bar_count = len(recent_kline)
    for i in range(bar_count):
        #past_highs.append(recent_kline[i]['High'])
        #past_lows.append(recent_kline[i]['Low'])
        past_highs.append(recent_kline[i][2]) # high
        past_lows.append(recent_kline[i][3]) # low
    # reverse list so most recent data comes first
    past_highs.reverse()
    past_lows.reverse()
    
    # looking for pivot high
    for index, high_val in enumerate(past_highs):
        # exclue boundary points
        if index >= right_bars and index <= (bar_count-1-left_bars):
            # only need to look at neighboring points
            # note python includes left boundary and exludes right boundary when indexing list
            adjacent_highs = past_highs[index-right_bars:index+left_bars+1]
            # found pivot high
            if is_pivot_high(high_val, adjacent_highs):
                current_pivot_high = high_val
                PVH_updated = True
                #print('\npivot high updated - current pivot high: ' + str(current_pivot_high))
                break
    # looking for pivot low
    for index, low_val in enumerate(past_lows):
        # exclue boundary points
        if index >= right_bars and index <= (bar_count-1-left_bars):
            # only need to look at neighboring points
            # note python includes left boundary and exludes right boundary when indexing list
            adjacent_lows = past_lows[index-right_bars:index+left_bars+1]
            # found pivot low
            if is_pivot_low(low_val, adjacent_lows):
                current_pivot_low = low_val
                PVL_updated = True
                #print('\npivot low updated - current pivot low: ' + str(current_pivot_low))
                break
    if current_pivot_high is not None and current_pivot_low is not None and current_pivot_high < current_pivot_low:
        if PVL_updated and PVH_updated:
            if current_pivot_high * 1.01 < current_pivot_low:
                raise ValueError('pivot high should be higher than pivot low')
            else:
                print('warning: pivot high and pivot low are too close')
        else:
            print('pivot high is lower than pivot low, this is good!')
            
def get_pv_index_str(left_bars, right_bars):
    pv_index_str = 'l'+str(left_bars)+'r'+str(right_bars)
    return pv_index_str
            
def get_and_update_pivot_points(recent_kline, left_bars, right_bars):
    global PVH_updated, PVL_updated, current_pivot_high, current_pivot_low, pv_highs, pv_lows
    
    get_current_pivot_points(recent_kline, left_bars, right_bars)
    pv_index_str = get_pv_index_str(left_bars, right_bars)
    if PVH_updated:
        pv_highs[pv_index_str] = current_pivot_high
    if PVL_updated:
        pv_lows[pv_index_str] = current_pivot_low
            
def get_order_size(pv_enum_str):
    global pv_highs, pv_lows, portfolio_value, leverage, max_port, adjust_leverage, P_val, leverage_adjust_co

    # adjust leverage
    if adjust_leverage:
        leverage_adjust_co = portfolio_value / max_port
        if leverage_adjust_co < 0.2:
            leverage_adjust_co = 0.2
        
    pvh = pv_highs[pv_enum_str]
    pvl = pv_lows[pv_enum_str]    
    if (pvh/pvl) > (1+P_val) and risk_control:
        order_size = portfolio_value * P_val / ((pvh/pvl) - 1) * leverage * leverage_adjust_co
    else:
        order_size = portfolio_value * leverage * leverage_adjust_co
    return order_size

def update_cumulative_return(orderDirection, pv_enum_str):
    global last_short_prices, last_long_prices, previous_order_sizes, cumulative_return, portfolio_value, max_port

    if last_short_prices[pv_enum_str] is None:
        last_short_price = None
    else:
        last_short_price = last_short_prices[pv_enum_str] * 0.9985 # friction
    if last_long_prices[pv_enum_str] is None:
        last_long_price = None
    else:
        last_long_price = last_long_prices[pv_enum_str] * 1.0015 # friction
    previous_order_size = previous_order_sizes[pv_enum_str]
    if last_short_price is None or last_long_price is None:
        raise ValueError('last_short_price and last_long_price should both be valid')        
    # long order (first sell then buy)
    print('pv_enum_str is: ' + pv_enum_str)
    if orderDirection == 1:        
        #print('\nSold at ' + str(last_short_price) + ' and Bought at ' + str(last_long_price))
        cumulative_return *= 1 + (last_short_price - last_long_price) / last_short_price * previous_order_size / portfolio_value
    # short order (first buy then sell)
    if orderDirection == -1:
        #print('\nBought at ' + str(last_long_price) + ' and Sold at ' + str(last_short_price))
        cumulative_return *= 1 + (last_short_price - last_long_price) / last_long_price * previous_order_size / portfolio_value
    portfolio_value = cumulative_return
    if max_port < portfolio_value:
        max_port = portfolio_value
    portfolio_curve.append(portfolio_value)
    print('\033[1m'+'Current cumulative_return is: ' + str(cumulative_return) + '\033[0m')
    return

def place_order_based_on_state(pv_enum_str, last_low, last_high, dt_str, current_close):
    global base_curve, base_price, pv_status, pv_highs, pv_lows, previous_order_sizes, last_short_prices, last_long_prices

    order_status = pv_status[pv_enum_str]
    pvl = pv_lows[pv_enum_str]
    pvh = pv_highs[pv_enum_str]
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@ payyyyyyyyyyy attentionnnnnnnnnnn tooooooooo thisssssssss caseeeeeeeee example l2r6 binance ethusdt Dec10-2019 ~ Dec19-2019
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if pvl > pvh:
        return
    
    # case 1 - no position
    if order_status == 0:        
        order_size = get_order_size(pv_enum_str)
        if last_low < pvl:
            print('\nCurrent date is ' + dt_str)
            print('zero ' + pv_enum_str + ' status => go short')
            print('last_low is ' + str(last_low))
            print('current ' + pv_enum_str + ' pivot low is ' + str(pvl))
            pv_status[pv_enum_str] = -1
            previous_order_sizes[pv_enum_str] = order_size
            last_short_prices[pv_enum_str] = pvl
        elif last_high > pvh:
            print('\nCurrent date is ' + dt_str)
            print('zero ' + pv_enum_str + ' status => go long')
            print('last_high is ' + str(last_high))
            print('current ' + pv_enum_str + ' pivot high is ' + str(pvh))
            pv_status[pv_enum_str] = 1
            previous_order_sizes[pv_enum_str] = order_size
            last_long_prices[pv_enum_str] = pvh            
    # case 2 - positive position
    elif order_status > 0:
        if last_low < pvl:
            #flip order
            order_size = get_order_size(pv_enum_str)
            print('\nCurrent date is ' + dt_str)
            print('positive ' + pv_enum_str + ' position => go short')
            print('last_low is ' + str(last_low))
            print('current ' + pv_enum_str + ' pivot low is ' + str(pvl))
            pv_status[pv_enum_str] = -1
            last_short_prices[pv_enum_str] = pvl
            update_cumulative_return(-1, pv_enum_str)
            base_curve.append(current_close/base_price)
            # this need to happen after update_cumulative_return
            previous_order_sizes[pv_enum_str] = order_size
    # case 3 - negative position
    else:
        if last_high > pvh:
            #flip order
            order_size = get_order_size(pv_enum_str)
            print('\nCurrent date is ' + dt_str)
            print('negative ' + pv_enum_str + ' position => go long')
            print('last_high is ' + str(last_high))
            print('current ' + pv_enum_str + ' pivot high is ' + str(pvh))
            pv_status[pv_enum_str] = 1
            last_long_prices[pv_enum_str] = pvh
            update_cumulative_return(1, pv_enum_str)
            base_curve.append(current_close/base_price)
            # this need to happen after update_cumulative_return
            previous_order_sizes[pv_enum_str] = order_size            

kline_hour = []
data_hour = data_hour.values.tolist()
current_position = 0
base_price = None
j = 0
max_j = len(data_hour)
for i in range(rows):
    dt_str = data_1m['Opened'][i]
    dt_obj = get_datetime(dt_str)
    # update 4hour pivot points
    if dt_obj.hour % hour_num == 0 and dt_obj.minute == 0 and dt_obj.second == 0:
        dt_obj_hour = get_datetime(data_hour[j][0])
        while dt_obj_hour < dt_obj:
            kline_hour.append(data_hour[j])
            j = j + 1
            if j >= max_j:
                break
            dt_obj_hour = get_datetime(data_hour[j][0])            
        if dt_obj != dt_obj_hour and hour_num != 1:
            print('\ninconsistency ...')
            print(dt_obj)
            print(dt_obj_hour)
            raise ValueError('inconsistency ...')
        if j >= max_j:
            break        
        if len(kline_hour) < 20:
            continue
        for index in range(len(pv_enum)):
            tmp_str = pv_enum[index].split('l')
            [left_bars, right_bars] = tmp_str[1].split('r')
            left_bars = int(left_bars)
            right_bars = int(right_bars)
            num_points = left_bars + right_bars + 1
            get_and_update_pivot_points(kline_hour[-num_points:], left_bars, right_bars)
        
    last_low = data_1m['Low'][i]
    last_high = data_1m['High'][i]
    current_close = data_1m['Close'][i]
    if dt_obj.year < start_year:
        continue
    if base_price is None:
        base_price = data_1m['Close'][i]
        print('base_price is: ' + str(base_price))
    for ordIdx in range(len(pv_enum)):
        if pv_highs[pv_enum[ordIdx]] is None or pv_lows[pv_enum[ordIdx]] is None:
            continue
        place_order_based_on_state(pv_enum[ordIdx], last_low, last_high, dt_str, current_close)

print('cumulative return is: ' + str(cumulative_return))
f = plt.figure(figsize = (7.2, 7.2))

ax1 = f.add_subplot(111)
ax1.plot(portfolio_curve, 'blue', linewidth=0.5)
ax1.plot(base_curve, 'black', linewidth=0.5)
ax1.set_title('portfolio curve')
ax1.set_xlabel('index')
ax1.set_ylabel('portfolio value')

plt.show()
