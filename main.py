import ccxt
import pandas as pd

from arblab.arb_lab import find_arb_opportunities, get_mkt_participants

pair = 'LTC/USDT'

# exchanges = get_mkt_participants(pair)

exchanges = {'Binance': ccxt.binance(),
             'HitBTC' : ccxt.hitbtc(),
             'Bittrex': ccxt.bittrex()}

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
#              'Gatecoin': ccxt.gatecoin(),
#              'HitBtc': ccxt.hitbtc(),
#              'GDAX': ccxt.gdax(),
#              'Kraken': ccxt.kraken(),
#              'LiveCoin': ccxt.livecoin(),
#              'Lykke': ccxt.lykke(),
#              'SouthXchange': ccxt.southxchange(),
#              'Vaultoro': ccxt.vaultoro()}

filtered_exchanges = { e:v for e,v in exchanges.items() if e not in ['CoinEx']}

accounts = {'HitBTC': {"LTC":2.38, "USDT":350.07},
            'Binance': {"LTC": 1.09, "USDT": 140.07}}

v = find_arb_opportunities(filtered_exchanges, pair, accounts)

hit_btc : ccxt.Exchange = exchanges['HitBTC']
hit_btc.apiKey = ''
hit_btc.secret = ''
hit_btc_balances = hit_btc.fetch_balance()

binance = exchanges['Binance']
binance.apiKey = ''
binance.secret = ''
binance_balances = binance.fetch_balance()

# hl = hit_btc.create_limit_sell_order(pair, 0.1,150.00 )
# ht = hit_btc.create_market_buy_order(pair, 0.1 )

# hit_btc.payment_post_transfer_to_main ({
#    'amount': 0.02,
#    'currency_code': 'LTC',
# })
#
# wh = hit_btc.withdraw('LTC',0.02, 'LRW15TRkMvy5B4T1KivpWfKWq5e87UtP5E')
hit_btc.has()
