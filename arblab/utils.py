import math
from functools import reduce
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
        else:  # the total cannot fully consume the volume so use the  volume and break out
            runner.append(x_['price_inc_fee'] * running_nominal)
            break
    return sum(runner) / target_nominal


def order_book_minimiser(market_df, target_nominal):
    running_nominal = target_nominal
    runner = []
    for x in range(len(market_df)):

        x_ = market_df.iloc[x].copy()
        vol = x_['vol']
        # If the current allocatable vol is less than the remaining total
        # use the volume and deduct this from the total else
        if vol < running_nominal:
            running_nominal -= vol
            x_['executable'] = x_['price_inc_fee'] * vol
            runner.append(x_)
        else:  # the total cannot fully consume the volume so use the  volume and break out
            x_['executable'] = x_['price_inc_fee'] * running_nominal
            runner.append(x_)
            break
    return pd.DataFrame(runner)


def get_arb_result_series(ex_pairs, ask_series, bid_series, order_books, coin_pair, accounts_dict) -> pd.Series:
    base = coin_pair.split('/')[0]
    quote = coin_pair.split('/')[1]
    for bid_ex, ask_ex in ex_pairs:
        target_ask = ask_series[ask_ex]
        target_bid = bid_series[bid_ex]
        bid_ex_bids = order_books[bid_ex]['bid']
        ask_ex_asks = order_books[ask_ex]['ask']
        bids = bid_ex_bids[bid_ex_bids['raw_price'] >= target_ask]
        offers = ask_ex_asks[ask_ex_asks['raw_price'] <= target_bid]
        bid_maximum_depth = bids['vol'].sum()
        ask_maximum_depth = offers['vol'].sum()
        bid_ex_acc_min = accounts_dict[bid_ex][quote] if bid_ex in accounts_dict and quote in accounts_dict[
            bid_ex] else float('inf')
        ask_ex_acc_min = get_opposing_leg_nominal(accounts_dict, ask_ex, base, offers)
        executable_amount = min(bid_maximum_depth, ask_maximum_depth, bid_ex_acc_min, ask_ex_acc_min)
        executable_amount_in_base = 1 / executable_amount if executable_amount > 1 else np.nan
        actual_ask_ob = order_book_minimiser(offers, executable_amount_in_base)
        actual_bid_ob = order_book_minimiser(bids, executable_amount_in_base)
        avg_bid = order_minimiser(bids, executable_amount_in_base)
        avg_ask = order_minimiser(offers, executable_amount_in_base)
        nominal_return = (avg_bid * executable_amount_in_base) - (avg_ask * executable_amount_in_base)
        pct_return = np.log(avg_bid / avg_ask)
        yield pd.Series({'avg_bid': avg_bid,
                         'avg_ask': avg_ask,
                         'executable_amount': executable_amount,
                         'executable_amount_in_base': executable_amount_in_base,
                         'maximum_bid_depth': bid_maximum_depth,
                         'maximum_ask_depth': ask_maximum_depth,
                         'nominal_return': nominal_return,
                         'pct_return': pct_return}, name='{}/{}'.format(bid_ex, ask_ex))


def get_opposing_leg_nominal(accounts_dict, exchange, leg, orders):
    if exchange in accounts_dict and leg in accounts_dict[exchange]:
        ask_ex_acc_min = accounts_dict[exchange][leg]
        return order_minimiser(orders, ask_ex_acc_min) * ask_ex_acc_min
    else:
        return float('inf')


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


flatten = lambda lst: reduce(lambda l, i: l + flatten(i) if isinstance(i, (list, tuple)) else l + [i], lst, [])
