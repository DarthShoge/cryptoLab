import math

import numpy as np
import pandas as pd


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


def get_exchange_pairs(arb_df):
    exchanges = arb_df.columns.values
    for col in exchanges:
        for row in exchanges:
            if not math.isnan(arb_df.loc[row, col]):
                yield row, col


def get_arb_depth(ex_pairs, ask_series, bid_series, order_books):
    for bid_ex, ask_ex in ex_pairs:
        bids = list(filter(lambda x: x[0] >= ask_series[ask_ex], order_books[bid_ex]['bids']))
        offers = list(filter(lambda x: x[0] <= bid_series[bid_ex], order_books[ask_ex]['asks']))
        bid_nominal = sum([x[1] for x in bids])
        ask_nominal = sum([x[1] for x in offers])
        executable_amount = min(bid_nominal, ask_nominal)
        yield bid_ex, ask_ex, executable_amount


def order_minimiser(market_df, target_nominal):
    running_nominal = target_nominal
    runner = []
    for x in range(len(market_df)):

        x_ = market_df.iloc[x]
        vol = x_['vol']
        # If the current allocatable vol is less than the remaining total
        # use the volume and deduct this from the total else
        if vol < running_nominal:
            running_nominal -= vol
            runner.append(x_['price_inc_fee'] * vol)
        else: # the total cannot fully consume the volume so use the  volume and break out
            runner.append(x_['price_inc_fee'] * running_nominal)
            break
    return sum(runner) / target_nominal


def get_arb_result_series(ex_pairs, ask_series, bid_series, order_books):
    for bid_ex, ask_ex in ex_pairs:
        target_ask = ask_series[ask_ex]
        target_bid = bid_series[bid_ex]
        bid_ex_bids = order_books[bid_ex]['bid']
        ask_ex_asks = order_books[ask_ex]['ask']
        bids = bid_ex_bids[bid_ex_bids['raw_price'] >= target_ask]
        offers = ask_ex_asks[ask_ex_asks['raw_price'] <= target_bid]
        bid_nominal = bids['vol'].sum()
        ask_nominal = offers['vol'].sum()
        executable_amount = min(bid_nominal, ask_nominal)
        avg_bid = order_minimiser(bids, executable_amount)
        avg_ask = order_minimiser(offers, executable_amount)
        nominal_return = (avg_bid*executable_amount) - (avg_ask*executable_amount)
        pct_return = np.log(avg_bid/avg_ask)
        yield pd.Series({'avg_bid': avg_bid,
                         'avg_ask': avg_ask,
                         'executable_amount': executable_amount,
                         'bid_nominal': bid_nominal,
                         'ask_nominal': ask_nominal,
                         'outlay': avg_ask * executable_amount,
                         'nominal_return': nominal_return,
                         'pct_return': pct_return}, name='{} vs {}'.format(bid_ex, ask_ex))


def orderBook_fee_deduction(order_books, fees):
    ask_fees = fees['ask']
    bid_fees = fees['bid']
    for x in order_books.keys():
        ask_fee = ask_fees[x]
        bid_fee = bid_fees[x]
        ask_orders = order_books[x]['asks']
        bid_orders = order_books[x]['bids']
        for n in range(0, len(ask_orders)):
            ask_orders[n][0] = ask_orders[n][0] * (1 + ask_fee)
        for n in range(0, len(bid_orders)):
            bid_orders[n][0] = bid_orders[n][0] * (1 - bid_fee)
        order_books[x]['asks'] = ask_orders
        order_books[x]['bids'] = bid_orders
    return order_books


def deduct_order_book_fees(order_books_dict, fee_df):
    ask_fees = fee_df['ask']
    bid_fees = fee_df['bid']
    results = {}
    for exchange, order_book in order_books_dict.items():
        ask_fee = ask_fees[exchange]
        bid_fee = bid_fees[exchange]
        ask_dict = {"price_inc_fee": np.array([x[0] for x in order_book['asks']]) * (1 + ask_fee),
                    'raw_price': np.array([x[0] for x in order_book['asks']]),
                    "vol": np.array([x[1] for x in order_book['asks']])}
        bid_dict = {'raw_price': np.array([x[0] for x in order_book['bids']]),
                    'price_inc_fee': np.array([x[0] for x in order_book['bids']]) * (1 - bid_fee),
                    "vol": np.array([x[1] for x in order_book['bids']])}
        results[exchange] = {'ask': pd.DataFrame(ask_dict), 'bid': pd.DataFrame(bid_dict)}
    return results