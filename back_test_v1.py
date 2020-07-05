from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data from file
data_2017_1m = pd.read_csv('data/eth_usdt_2017_1m.csv')
data_2017_4h = pd.read_csv('data/eth_usdt_2017_4h.csv')
data_2018_1m = pd.read_csv('data/eth_usdt_2018_1m.csv')
data_2018_4h = pd.read_csv('data/eth_usdt_2018_4h.csv')
data_2019_1m = pd.read_csv('data/eth_usdt_2019_1m.csv')
data_2019_4h = pd.read_csv('data/eth_usdt_2019_4h.csv')
data_2020_1m = pd.read_csv('data/eth_usdt_2020_1m.csv')
data_2020_4h = pd.read_csv('data/eth_usdt_2020_4h.csv')

#data_1m = pd.DataFrame(np.concatenate([data_2017_1m.values]), columns=data_2017_1m.columns)
#data_4h = pd.DataFrame(np.concatenate([data_2017_4h.values]), columns=data_2017_4h.columns)

#data_1m = pd.DataFrame(np.concatenate([data_2018_1m.values]), columns=data_2018_1m.columns)
#data_4h = pd.DataFrame(np.concatenate([data_2018_4h.values]), columns=data_2018_4h.columns)

#data_1m = pd.DataFrame(np.concatenate([data_2019_1m.values]), columns=data_2019_1m.columns)
#data_4h = pd.DataFrame(np.concatenate([data_2019_4h.values]), columns=data_2019_4h.columns)

#data_1m = pd.DataFrame(np.concatenate([data_2020_1m.values]), columns=data_2020_1m.columns)
#data_4h = pd.DataFrame(np.concatenate([data_2020_4h.values]), columns=data_2020_4h.columns)

data_1m = pd.DataFrame(np.concatenate([data_2019_1m.values, data_2020_1m.values]), columns=data_2019_1m.columns)
data_4h = pd.DataFrame(np.concatenate([data_2019_4h.values, data_2020_4h.values]), columns=data_2019_4h.columns)

#data_1m = pd.DataFrame(np.concatenate([data_2018_1m.values, data_2019_1m.values, data_2020_1m.values]), columns=data_2019_1m.columns)
#data_4h = pd.DataFrame(np.concatenate([data_2018_4h.values, data_2019_4h.values, data_2020_4h.values]), columns=data_2019_4h.columns)

#data_1m = pd.DataFrame(np.concatenate([data_2017_1m.values, data_2018_1m.values, data_2019_1m.values, data_2020_1m.values]), columns=data_2019_1m.columns)
#data_4h = pd.DataFrame(np.concatenate([data_2017_4h.values, data_2018_4h.values, data_2019_4h.values, data_2020_4h.values]), columns=data_2019_4h.columns)

[rows, columns] = data_1m.shape

# try L6R1, L4R2, L2R6 with P_val = 1/100, adjust_leverage = True, leverage = 8, leverage_adjust_co = 0.4

#L6R1, L4R2, L2R6, L6R6, (L4R6)
risk_control = True
# 10 percent
P_val = 1/100
adjust_leverage = False
leverage_adjust_co = 1
leverage = 12

max_port = 1
portfolio_value = 1
current_pivot_high = None
current_pivot_low = None
pvh_l6r1 = None
pvl_l6r1 = None
pvh_l4r2 = None
pvl_l4r2 = None
pvh_l4r6 = None
pvl_l4r6 = None
pvh_l6r6 = None
pvl_l6r6 = None
pvh_l2r6 = None
pvl_l2r6 = None
PVH_updated = False
PVL_updated = False
l6r1_status = 0
l4r2_status = 0
l4r6_status = 0
l6r6_status = 0
l2r6_status = 0
previous_order_size_l6r1 = 0
previous_order_size_l4r2 = 0
previous_order_size_l4r6 = 0
previous_order_size_l6r6 = 0
previous_order_size_l2r6 = 0
last_short_price_l6r1 = None
last_long_price_l6r1 = None
last_short_price_l4r2 = None
last_long_price_l4r2 = None
last_short_price_l4r6 = None
last_long_price_l4r6 = None
last_short_price_l6r6 = None
last_long_price_l6r6 = None
last_short_price_l2r6 = None
last_long_price_l2r6 = None

cumulative_return = 1
portfolio_curve = []
portfolio_curve.append(cumulative_return)
base_curve = []
base_curve.append(1)

def get_datetime(datetime_str):
    datetime_str = datetime_str.split('+')[0]
    datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    #print(type(datetime_obj))
    #print(datetime_obj)
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

def get_current_pivot_points(recent_kline_4h, left_bars, right_bars):
    global current_pivot_high, current_pivot_low, PVH_updated, PVL_updated
    
    PVH_updated = False
    PVL_updated = False
    
    past_highs = []
    past_lows = []
    bar_count = len(recent_kline_4h)
    for i in range(bar_count):
        past_highs.append(recent_kline_4h[i]['high'])
        past_lows.append(recent_kline_4h[i]['low'])    
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
            raise ValueError('pivot high should be higher than pivot low')
        else:
            print('pivot high is lower than pivot low, this is good!')

def get_and_update_pivot_points(recent_kline_4h, left_bars, right_bars):
    global PVH_updated, PVL_updated, current_pivot_high, current_pivot_low, pvh_l6r1, pvl_l6r1, pvh_l4r2, pvl_l4r2, pvh_l4r6, pvl_l4r6, pvh_l6r6, pvl_l6r6, pvh_l2r6, pvl_l2r6
    get_current_pivot_points(recent_kline_4h, left_bars, right_bars)    
    if left_bars == 6 and right_bars == 1:
        if PVH_updated:
            pvh_l6r1 = current_pivot_high
        if PVL_updated:
            pvl_l6r1 = current_pivot_low
    if left_bars == 4 and right_bars == 2:
        if PVH_updated:
            pvh_l4r2 = current_pivot_high
        if PVL_updated:
            pvl_l4r2 = current_pivot_low
    if left_bars == 4 and right_bars == 6:
        if PVH_updated:
            pvh_l4r6 = current_pivot_high
        if PVL_updated:
            pvl_l4r6 = current_pivot_low
    if left_bars == 6 and right_bars == 6:
        if PVH_updated:
            pvh_l6r6 = current_pivot_high
        if PVL_updated:
            pvl_l6r6 = current_pivot_low
    if left_bars == 2 and right_bars == 6:
        if PVH_updated:
            pvh_l2r6 = current_pivot_high
        if PVL_updated:
            pvl_l2r6 = current_pivot_low                        
            
def get_order_size(pv_enum_str):
    global pvh_l6r1, pvl_l6r1, pvh_l4r2, pvl_l4r2, pvh_l4r6, pvl_l4r6, pvh_l6r6, pvl_l6r6, pvh_l2r6, pvl_l2r6, portfolio_value, leverage, P_val, adjust_leverage, max_port, leverage_adjust_co

    # adjust leverage
    if adjust_leverage:
        leverage_adjust_co = portfolio_value / max_port
        if leverage_adjust_co < 0.5:
            leverage_adjust_co = 0.5
    
    if pv_enum_str == 'l6r1':
        pvh = pvh_l6r1
        pvl = pvl_l6r1
    elif pv_enum_str == 'l4r2':
        pvh = pvh_l4r2
        pvl = pvl_l4r2
    elif pv_enum_str == 'l4r6':
        pvh = pvh_l4r6
        pvl = pvl_l4r6
    elif pv_enum_str == 'l6r6':
        pvh = pvh_l6r6
        pvl = pvl_l6r6
    elif pv_enum_str == 'l2r6':
        pvh = pvh_l2r6
        pvl = pvl_l2r6        
    else:
        raise ValueError('unsupported pv_enum_str')
    if (pvh/pvl) > (1+P_val) and risk_control:
        order_size = portfolio_value * P_val / ((pvh/pvl) - 1) * leverage * leverage_adjust_co
    else:
        order_size = portfolio_value * leverage * leverage_adjust_co
    return order_size

def update_cumulative_return(orderDirection, pv_enum_str):
    global previous_order_size_l6r1, previous_order_size_l4r2, previous_order_size_l4r6, previous_order_size_l6r6, previous_order_size_l2r6, last_short_price_l6r1, last_long_price_l6r1, last_short_price_l4r2, last_long_price_l4r2, last_short_price_l4r6, last_long_price_l4r6, last_short_price_l6r6, last_long_price_l6r6, last_short_price_l2r6, last_long_price_l2r6, cumulative_return, portfolio_value, max_port
    if pv_enum_str == 'l6r1':
        last_short_price = last_short_price_l6r1
        last_long_price = last_long_price_l6r1
        previous_order_size = previous_order_size_l6r1
    elif pv_enum_str == 'l4r2':
        last_short_price = last_short_price_l4r2
        last_long_price = last_long_price_l4r2
        previous_order_size = previous_order_size_l4r2
    elif pv_enum_str == 'l4r6':
        last_short_price = last_short_price_l4r6
        last_long_price = last_long_price_l4r6
        previous_order_size = previous_order_size_l4r6
    elif pv_enum_str == 'l6r6':
        last_short_price = last_short_price_l6r6
        last_long_price = last_long_price_l6r6
        previous_order_size = previous_order_size_l6r6
    elif pv_enum_str == 'l2r6':
        last_short_price = last_short_price_l2r6
        last_long_price = last_long_price_l2r6
        previous_order_size = previous_order_size_l2r6
    else:
        raise ValueError('unsupported pv_enum_str')
    
    if last_short_price is None or last_long_price is None:
        raise ValueError('last_short_price and last_long_price should both be valid')        
    # long order
    print('pv_enum_str is: ' + pv_enum_str)
    if orderDirection == 1:
        print('\nSold at ' + str(last_short_price) + ' and Bought at ' + str(last_long_price))
        #cumulative_return *= 1 + (last_short_price / last_long_price * 0.996 - 1) * previous_order_size / portfolio_value
        cumulative_return *= 1 + (last_short_price - last_long_price) / last_short_price * 0.996 * previous_order_size / portfolio_value
    # short order
    if orderDirection == -1:
        print('\nBought at ' + str(last_long_price) + ' and Sold at ' + str(last_short_price))
        cumulative_return *= 1 + (last_short_price / last_long_price * 0.996 - 1) * previous_order_size / portfolio_value

    portfolio_value = cumulative_return
    if max_port < portfolio_value:
        max_port = portfolio_value
    portfolio_curve.append(portfolio_value)
    print('\033[1m'+'Current cumulative_return is: ' + str(cumulative_return) + '\033[0m')
    return

def place_order_based_on_state(pv_enum_str, last_low, last_high, dt_str, current_close):
    global l6r1_status, l4r2_status, l4r6_status, l6r6_status, l2r6_status, pvh_l6r1, pvl_l6r1, pvh_l4r2, pvl_l4r2, pvh_l4r6, pvl_l4r6, pvh_l6r6, pvl_l6r6, pvh_l2r6, pvl_l2r6, previous_order_size_l6r1, previous_order_size_l4r2, previous_order_size_l4r6, previous_order_size_l6r6, previous_order_size_l2r6, last_short_price_l6r1, last_long_price_l6r1, last_short_price_l4r2, last_long_price_l4r2, last_short_price_l4r6, last_long_price_l4r6, last_short_price_l6r6, last_long_price_l6r6, last_short_price_l2r6, last_long_price_l2r6, base_curve, base_price
    if pv_enum_str == 'l6r1':
        order_status = l6r1_status
        pvl = pvl_l6r1
        pvh = pvh_l6r1
    elif pv_enum_str == 'l4r2':
        order_status = l4r2_status
        pvl = pvl_l4r2
        pvh = pvh_l4r2
    elif pv_enum_str == 'l4r6':
        order_status = l4r6_status
        pvl = pvl_l4r6
        pvh = pvh_l4r6
    elif pv_enum_str == 'l6r6':
        order_status = l6r6_status
        pvl = pvl_l6r6
        pvh = pvh_l6r6
    elif pv_enum_str == 'l2r6':
        order_status = l2r6_status
        pvl = pvl_l2r6
        pvh = pvh_l2r6                
    else:
        raise ValueError('unsupported pv_enum_str')
    
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
            if pv_enum_str == 'l6r1':
                l6r1_status = -1
                previous_order_size_l6r1 = order_size
                last_short_price_l6r1 = pvl
            elif pv_enum_str == 'l4r2':
                l4r2_status = -1
                previous_order_size_l4r2 = order_size
                last_short_price_l4r2 = pvl
            elif pv_enum_str == 'l4r6':
                l4r6_status = -1
                previous_order_size_l4r6 = order_size
                last_short_price_l4r6 = pvl
            elif pv_enum_str == 'l6r6':
                l6r6_status = -1
                previous_order_size_l6r6 = order_size
                last_short_price_l6r6 = pvl
            elif pv_enum_str == 'l2r6':
                l2r6_status = -1
                previous_order_size_l2r6 = order_size
                last_short_price_l2r6 = pvl                                
            else:
                raise ValueError('unsupported pv_enum_str')            
        if last_high > pvh:
            print('\nCurrent date is ' + dt_str)
            print('zero ' + pv_enum_str + ' status => go long')
            print('last_high is ' + str(last_high))
            print('current ' + pv_enum_str + ' pivot high is ' + str(pvh))
            if pv_enum_str == 'l6r1':
                l6r1_status = 1
                previous_order_size_l6r1 = order_size
                last_long_price_l6r1 = pvh
            elif pv_enum_str == 'l4r2':
                l4r2_status = 1
                previous_order_size_l4r2 = order_size
                last_long_price_l4r2 = pvh
            elif pv_enum_str == 'l4r6':
                l4r6_status = 1
                previous_order_size_l4r6 = order_size
                last_long_price_l4r6 = pvh
            elif pv_enum_str == 'l6r6':
                l6r6_status = 1
                previous_order_size_l6r6 = order_size
                last_long_price_l6r6 = pvh
            elif pv_enum_str == 'l2r6':
                l2r6_status = 1
                previous_order_size_l2r6 = order_size
                last_long_price_l2r6 = pvh                                
            else:
                raise ValueError('unsupported pv_enum_str')                        
    # case 2 - positive position
    elif order_status > 0:
        if last_low < pvl:
            #flip order
            order_size = get_order_size(pv_enum_str)
            print('\nCurrent date is ' + dt_str)
            print('positive ' + pv_enum_str + ' position => go short')
            print('last_low is ' + str(last_low))
            print('current ' + pv_enum_str + ' pivot low is ' + str(pvl))            
            if pv_enum_str == 'l6r1':
                l6r1_status = -1
                last_short_price_l6r1 = pvl
                update_cumulative_return(-1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_size_l6r1 = order_size
            elif pv_enum_str == 'l4r2':
                l4r2_status = -1
                last_short_price_l4r2 = pvl
                update_cumulative_return(-1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_size_l4r2 = order_size
            elif pv_enum_str == 'l4r6':
                l4r6_status = -1
                last_short_price_l4r6 = pvl
                update_cumulative_return(-1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_size_l4r6 = order_size
            elif pv_enum_str == 'l6r6':
                l6r6_status = -1
                last_short_price_l6r6 = pvl
                update_cumulative_return(-1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_size_l6r6 = order_size
            elif pv_enum_str == 'l2r6':
                l2r6_status = -1
                last_short_price_l2r6 = pvl
                update_cumulative_return(-1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_size_l2r6 = order_size                                
            else:
                raise ValueError('unsupported pv_enum_str')
    # case 3 - negative position
    else:
        if last_high > pvh:
            #flip order
            order_size = get_order_size(pv_enum_str)
            print('\nCurrent date is ' + dt_str)
            print('negative ' + pv_enum_str + ' position => go long')
            print('last_high is ' + str(last_high))
            print('current ' + pv_enum_str + ' pivot high is ' + str(pvh))
            if pv_enum_str == 'l6r1':
                l6r1_status = 1
                last_long_price_l6r1 = pvh
                update_cumulative_return(1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_percent_l6r1 = order_size
            elif pv_enum_str == 'l4r2':
                l4r2_status = 1
                last_long_price_l4r2 = pvh
                update_cumulative_return(1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_percent_l4r2 = order_size
            elif pv_enum_str == 'l4r6':
                l4r6_status = 1
                last_long_price_l4r6 = pvh
                update_cumulative_return(1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_percent_l4r6 = order_size
            elif pv_enum_str == 'l6r6':
                l6r6_status = 1
                last_long_price_l6r6 = pvh
                update_cumulative_return(1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_percent_l6r6 = order_size
            elif pv_enum_str == 'l2r6':
                l2r6_status = 1
                last_long_price_l2r6 = pvh
                update_cumulative_return(1, pv_enum_str)
                base_curve.append(current_close/base_price)
                # this need to happen after update_cumulative_return
                previous_order_percent_l2r6 = order_size                            
            else:
                raise ValueError('unsupported pv_enum_str')

kline_4h = []
current_position = 0
base_price = 0
for i in range(rows):
    if i == 0:
        base_price = data_1m['close'][i]
    dt_str = data_1m['datetime'][i]
    dt_obj = get_datetime(dt_str)
    # update 4hour pivot points
    if dt_obj.hour % 4 == 0 and dt_obj.minute == 0 and dt_obj.second == 0:
        kline_4h.append(data_4h.loc[i])
        if len(kline_4h) < 13:
            continue
        left_bars = 6
        right_bars = 1
        num_points = left_bars + right_bars + 1
        get_and_update_pivot_points(kline_4h[-num_points:], left_bars, right_bars)
        left_bars = 4
        right_bars = 2
        num_points = left_bars + right_bars + 1
        get_and_update_pivot_points(kline_4h[-num_points:], left_bars, right_bars)
        left_bars = 4
        right_bars = 6
        num_points = left_bars + right_bars + 1
        get_and_update_pivot_points(kline_4h[-num_points:], left_bars, right_bars)
        left_bars = 6
        right_bars = 6
        num_points = left_bars + right_bars + 1
        get_and_update_pivot_points(kline_4h[-num_points:], left_bars, right_bars)
        left_bars = 2
        right_bars = 6
        num_points = left_bars + right_bars + 1
        get_and_update_pivot_points(kline_4h[-num_points:], left_bars, right_bars)                
    if pvh_l6r1 is None or pvl_l6r1 is None or pvh_l4r2 is None or pvl_l4r2 is None or pvh_l4r6 is None or pvl_l4r6 is None or pvh_l6r6 is None or pvl_l6r6 is None or pvh_l2r6 is None or pvl_l2r6 is None:
        continue
    last_low = data_1m['low'][i]
    last_high = data_1m['high'][i]
    current_close = data_1m['close'][i]
    place_order_based_on_state('l6r1', last_low, last_high, dt_str, current_close)
    #place_order_based_on_state('l2r6', last_low, last_high, dt_str, current_close)        
    #place_order_based_on_state('l4r2', last_low, last_high, dt_str, current_close)
    ##place_order_based_on_state('l4r6', last_low, last_high, dt_str, current_close)
    ##place_order_based_on_state('l6r6', last_low, last_high, dt_str, current_close)
            
print('cumulative return is: ' + str(cumulative_return))
f = plt.figure(figsize = (7.2, 7.2))

ax1 = f.add_subplot(111)
ax1.plot(portfolio_curve, 'blue', linewidth=0.5)
ax1.plot(base_curve, 'black', linewidth=0.5)
ax1.set_title('portfolio curve')
ax1.set_xlabel('index')
ax1.set_ylabel('portfolio value')

plt.show()

