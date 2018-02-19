import ccxt
import pandas as pd

usdt_exchanges = {
    'binance': ccxt.binance(),
    'hitbtc': ccxt.hitbtc(),
    # 'huobi': ccxt.huobi(),
    'poloniex': ccxt.poloniex(),
    'bittrex': ccxt.bittrex(),
    'gateio': ccxt.gateio(),
    'kucoin': ccxt.kucoin(),
    'tidex': ccxt.tidex(),
    'cryptopia': ccxt.cryptopia(),
    'okex': ccxt.okex()

}


def get_clean_ticker(ex, pair):
    try:
        d = ex.fetch_ticker(pair)
        del (d['info'])
        return d
    except:
        return {}


def get_fee_structure(ex, pair):
    try:
        d = ex.load_markets()[pair]
        return {'ask': d['maker'], 'bid': d['taker']}
    except:
        return {}


coin = 'ETH'

coin_data = pd.DataFrame(
    {key: get_clean_ticker(val, '%s/USDT' % coin) for key, val in usdt_exchanges.items()}).transpose()

ask_ser = coin_data['ask']
# ask_ser.columns = ['value']
bid_ser = coin_data['bid']
# bid_ser.columns = ['value']

# This creates a matrix that shows the difference between the bid and ask in pct terms (dodgy math)
ones_df = pd.DataFrame(1, index=ask_ser.index, columns=bid_ser.index)
arb_df = (ones_df.multiply(bid_ser, axis='index') / ask_ser) - 1

relevant_arbs_df = arb_df[arb_df > 0.01]

fee_df = pd.DataFrame({key: get_fee_structure(ex,'%s/USDT' % coin) for key, ex in usdt_exchanges.items()}).transpose()

#Some fees are shown as negative thus the application of abs over each series
fee_matrix_df = ones_df.multiply(fee_df['ask'].apply(abs), axis='index') + fee_df['bid'].apply(abs)

opportunity_df = relevant_arbs_df - fee_matrix_df