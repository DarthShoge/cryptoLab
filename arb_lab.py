from functools import reduce

import ccxt
import pandas as pd
import math

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


flatten = lambda lst: reduce(lambda l, i: l + flatten(i) if isinstance(i, (list, tuple)) else l + [i], lst, [])


def get_exchange_pairs(arb_df):
    exchanges = arb_df.columns.values
    for col in exchanges:
        for row in exchanges:
            if not math.isnan(arb_df.loc[row, col]):
                yield row, col


def get_arb_depth(ex_pairs, ask_series, bid_series, order_books):
    for bid_ex, ask_ex in ex_pairs:
        bids = list(filter(lambda x: x[0] <= ask_series[ask_ex], order_books[bid_ex]['bids']))
        offers = list(filter(lambda x: x[0] >= bid_series[bid_ex], order_books[bid_ex]['asks']))
        bid_nominal = sum([x[1] for x in bids])
        ask_nominal = sum([x[1] for x in offers])
        executable_amount = min(bid_nominal, ask_nominal)
        yield bid_ex, ask_ex, executable_amount


coin = 'LTC'
pair = '%s/USDT' % coin

coin_data = pd.DataFrame(
    {key: get_clean_ticker(val, pair) for key, val in usdt_exchanges.items()}).transpose()

fee_df = pd.DataFrame({key: get_fee_structure(ex, pair) for key, ex in usdt_exchanges.items()}).transpose()
fee_df = fee_df.apply(abs)

ask_ser = coin_data['ask'] * (1 + fee_df['ask'])
bid_ser = coin_data['bid'] * (1 - fee_df['bid'])
ones_df = pd.DataFrame(1, index=ask_ser.index, columns=bid_ser.index)
arb_df = (ones_df.multiply(bid_ser, axis='index') / ask_ser) - 1

relevant_arbs_df = arb_df[arb_df > 0]

exchange_pairs = list(get_exchange_pairs(relevant_arbs_df))

distinct_arb_exchanges = set(flatten(exchange_pairs))

arb_ex_order_book_dict = {x: usdt_exchanges[x].fetch_order_book(pair) for x in distinct_arb_exchanges}

executable_amounts = {(buy_ex, sell_ex): val for buy_ex, sell_ex, val in
                      get_arb_depth(exchange_pairs, ask_ser, bid_ser, arb_ex_order_book_dict)}
