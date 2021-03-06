from abc import ABCMeta

import ccxt
from ccxt import Exchange
from numpy import nan


class ExchangeAccount(metaclass=ABCMeta):

    def __init__(self, deposit_address, buy_coin, sell_coin, exchange: Exchange):
        self.sell_coin = sell_coin
        self.buy_coin = buy_coin
        self.deposit_address = deposit_address
        self.exchange = exchange
        # if not exchange.has['withdraw']:
        #     raise ccxt.ExchangeError('Exchange cannot withdraw')

    def buy(self, pair, amount=nan, **kwargs):
        return self.exchange.create_market_buy_order(pair, amount, params=kwargs)

    def sell(self, pair, amount=nan, **kwargs):
        return self.exchange.create_market_sell_order(pair, amount, params=kwargs)

    def rebalance(self, coin, amount, to):
        return self.exchange.withdraw(coin, amount, to)

    def rebalance_sell(self, to):
        balances = self.get_balances()
        rebalance_value = balances[self.sell_coin] / 2
        return self.rebalance(self.sell_coin, rebalance_value, to)

    def rebalance_buy(self, to):
        balances = self.get_balances()
        rebalance_value = balances[self.buy_coin] / 2
        return self.rebalance(self.buy_coin, rebalance_value, to)

    def get_balances(self, **kwargs):
        balances = {k: v for k, v in self.exchange.fetch_free_balance(kwargs).items() if v > 0}
        return balances


class HitBtcAccount(ExchangeAccount):
    def __init__(self, deposit_address, buy_coin, sell_coin, exchange: ccxt.hitbtc = None):
        ExchangeAccount.__init__(self, deposit_address, buy_coin,
                                 sell_coin, exchange if exchange is not None else ccxt.hitbtc())

    def buy(self, pair, amount=nan, **kwargs):
        balance = self.get_balances()
        if balance[self.sell_coin] < 0.005:
            balances = self.transfer_trade_to_main(self.sell_coin)
        super(HitBtcAccount, self).buy(pair, balance[self.sell_coin])

    def sell(self, pair, amount=nan, **kwargs):
        balance = self.get_balances()
        if balance[self.buy_coin] < 0.005:
            balances = self.transfer_trade_to_main(self.buy_coin)
        super(HitBtcAccount, self).sell(pair, balance[self.buy_coin])

    def rebalance(self, coin, amount, to):
        self.exchange.payment_post_transfer_to_main ({
           'amount': amount,
           'currency_code': coin,
        })
        return self.exchange.withdraw(coin, amount, to)

    def transfer_trade_to_main(self, coin):
        balances = self.get_balances(type='main')
        if coin not in balances and balances[coin] < 0.0001:
            self.exchange.payment_post_transfer_to_trading({
                'amount': balances[coin],
                'currency_code': coin,
            })

            balances = self.get_balances(type='trade')
        return balances


class AccountFactory:

    def __init__(self, config):
        self.config = config['Accounts']

    proxies = {'HitBTC':HitBtcAccount}

    def create(self, ex, base, quote):
        (e,) = (x for x in ccxt.Exchange.__subclasses__() if x().name == ex)
        exchange = e()
        exchange.apiKey = self.config[ex]['apikey']
        exchange.secret = self.config[ex]['secret']
        deposit = self.config[ex]['deposit'][base]
        if ex in AccountFactory.proxies:
            return AccountFactory.proxies[ex](deposit, base, quote, exchange)
        else:
            return ExchangeAccount(deposit, base, quote, exchange)
