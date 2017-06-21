import requests
import numpy as np

def test_data_sin (size):
    lin = [np.sin(i) for i in np.linspace(size,.5)]
    norm_lin = [x/size for x in lin]
    volume = np.ones(size)

    data = [[l,l,l,l,v,l] for l,v in zip(lin, volume)]
    norm_data = [[l,l,l,l,v,l] for l,v in zip(norm_lin, volume)]

    return data[1:], norm_data[1:] # Remove 0s

def test_data_lin (size):
    lin = np.arange(size)
    norm_lin = [x/size for x in lin]
    volume = np.ones(size)

    data = [[l,l,l,l,v,l] for l,v in zip(lin, volume)]
    norm_data = [[l,l,l,l,v,l] for l,v in zip(norm_lin, volume)]

    return data[1:], norm_data[1:] # Remove 0s

def get_data (url):

    if '.npy' in url:
        data = omit( np.load(url) )[1:]
    else:
        data = omit( raw_data(url) )[1:]

    return [[ \
        x['close'], x['open'], \
        x['high'], x['low'], \
        x['volume'], \
        x['weightedAverage']] for x in data]

def get_norm_data (url):
    if '.npy' in url:
        data = normalize( omit( np.load(url) )[1:] )
    else:
        data = normalize( omit( raw_data(url) )[1:] )

    return [[ \
        x['close'], x['open'], \
        x['high'], x['low'], \
        x['volume'], \
        x['weightedAverage']] for x in data]

def raw_data (url):
    return requests.get(url).json()

def omit (data):
    for x in data:
        del( x['date'] )
        del( x['quoteVolume'] )

    return data

def normalize (data):
    # Normalize volume
    v = [x['volume'] for x in data]
    v_max = np.max(v)
    for x in data:
        x['volume'] = x['volume'] / v_max

    # Normalize prices to percent change
    for i,x in reversed(list(enumerate(data))):
        # Close
        c = x['close']
        x['close'] = (c - data[i-1]['close']) / c
        # Open
        c = x['open']
        x['open'] = (c - data[i-1]['open']) / c
        # High
        c = x['high']
        x['high'] = (c - data[i-1]['high']) / c
        # Low
        c = x['low']
        x['low'] = (c - data[i-1]['low']) / c
        # Weight avg
        c = x['weightedAverage']
        x['weightedAverage'] = (c - data[i-1]['weightedAverage']) / c

    return data

if __name__ == '__main__':
    data = get_data('https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1435699200&end=9999999999&period=14400')
    print(np.max( [x['weightedAverage'] for x in data] ))
