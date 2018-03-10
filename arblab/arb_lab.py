import ccxt
import pandas as pd


from arblab.utils import get_clean_ticker, get_fee_structure, get_exchange_pairs, get_arb_result_series, \
    deduct_order_book_fees, flatten



def find_arb_opportunities(exchanges, pair):
    coin_data = pd.DataFrame(
        {key: get_clean_ticker(val, pair) for key, val in exchanges.items()}).transpose()

    # retrieve exchange-specific transaction fees
    fee_df = pd.DataFrame({key: get_fee_structure(ex, pair) for key, ex in exchanges.items()}).transpose()
    fee_df = fee_df.apply(abs)

    # bid/ask series without fee deduction
    ask_ser_nf = coin_data['ask'] * (1 + fee_df['ask'])
    bid_ser_nf = coin_data['bid'] * (1 - fee_df['bid'])

    # bid/ask series with fee deduction
    ask_ser = coin_data['ask'] * (1 + fee_df['ask'])
    bid_ser = coin_data['bid'] * (1 - fee_df['bid'])

    # matrix of arbitrage opportunities after deducting for exchange fees
    ones_df = pd.DataFrame(1, index=ask_ser.index, columns=bid_ser.index)
    arb_df = (ones_df.multiply(bid_ser, axis='index') / ask_ser) - 1
    relevant_arbs_df = arb_df[arb_df > 0]

    exchange_pairs = list(get_exchange_pairs(relevant_arbs_df))
    distinct_arb_exchanges = set(flatten(exchange_pairs))

    # get bid ask order books for each excahnge
    arb_ex_order_book_dict = {x: exchanges[x].fetch_order_book(pair) for x in distinct_arb_exchanges}
    # deduct exchange-specific fees from bid/ask prices in order books
    order_book_df_dict = deduct_order_book_fees(arb_ex_order_book_dict, fee_df)

    executable_amounts = pd.DataFrame(
        [val for val in get_arb_result_series(exchange_pairs, ask_ser, bid_ser, order_book_df_dict)])

    return executable_amounts


def get_mkt_participants(pair):
    subclasses = {}
    children = ccxt.Exchange.__subclasses__()
    for child in children:
        if child not in subclasses:
            ex = child()
            try :
                market = ex.fetch_ticker(pair)
                subclasses[ex.name] = ex
            except Exception as e:
                pass

    return subclasses