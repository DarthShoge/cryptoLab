from typing import Dict

import ccxt
import pandas as pd
import json
from arblab.account import AccountFactory, ExchangeAccount
from arblab.arb_lab import find_arb_opportunities, get_mkt_participants

base = 'LTC'
quote = 'USDT'
pair = '%s/%s' % (base, quote)


exchanges = get_mkt_participants(pair)
arbs = find_arb_opportunities(exchanges, pair)
arbs = arbs[arbs['pct_return'] > 0]


# exchanges = {'OKCoin USD': ccxt.okcoinusd(),
#              'BitBay': ccxt.bitbay(),
#              'Bitfinex': ccxt.bitfinex(),
#              'Binance': ccxt.binance({'verbose': True}),
#              'Bitstamp': ccxt.bitstamp(),
#              'BTCTurk': ccxt.btcturk(),
#              'C-CEX': ccxt.ccex(),
#              'CoinMarketCap': ccxt.coinmarketcap(),
#              'Coinsecure': ccxt.coinsecure(),
#              'EXMO': ccxt.exmo(),
#              'FYB-SE': ccxt.fybse(),
#              'Gatecoin': ccxt.gatecoin()fac  factory k
#              'HitBtc': ccxt.hitbtc(),
#              'GDAX': ccxt.gdax(),
#              'Kraken': ccxt.kraken(),
#              'LiveCoin': ccxt.livecoin(),
#              'Lykke': ccxt.lykke(),
#              'SouthXchange': ccxt.southxchange(),
#              'Vaultoro': ccxt.vaultoro()}

# filtered_exchanges = {e: v for e, v in exchanges.items() if e not in ['CoinEx']}

# account_balances = {'HitBTC': {"LTC":2.38, "USDT":350.07},
#             'Binance': {"LTC": 1.09, "USDT": 140.07}}


def perform_arb(accounts_dict, account_balances_dict, arb_ser):
    ex_pair = tuple(arb_ser.name.split('/'))
    (sell_ex, buy_ex) = ex_pair
    so = accounts_dict[sell_ex].sell(pair, account_balances_dict[sell_ex][base])
    bo = accounts_dict[buy_ex].buy(pair, account_balances_dict[buy_ex][quote])
    rb = accounts_dict[buy_ex].rebalance_buy(accounts_dict[sell_ex].deposit_address)
    rs = accounts_dict[sell_ex].rebalance_sell(accounts_dict[buy_ex].deposit_address)
    return (so, bo, rb, rs)


config = json.load(open('config.json'))


def run_real_arb():
    factory = AccountFactory(config)
    accounts: Dict[str, ExchangeAccount] = {k: factory.create(k, base, quote) for k, v in config['Accounts'].items()}
    account_balances = {k: v.get_balances() for k, v in accounts.items()}
    exchanges = {k: v.exchange for k, v in accounts.items()}
    arbs = find_arb_opportunities(exchanges, pair, account_balances)
    arb_pairings_dict = [perform_arb(accounts, account_balances, arbs.iloc[x]) for x in
                         range(len(arbs[arbs.pct_return > 0]))]
    return arb_pairings_dict


run_real_arb()

t = 1