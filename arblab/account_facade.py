from abc import ABCMeta

from ccxt import Exchange


class ExchangeAccount(metaclass=ABCMeta):

    def __init__(self, exchange : Exchange, deposit_address, buy_coin, sell_coin):
        self.sell_coin = sell_coin
        self.buy_coin = buy_coin
        self.deposit_address = deposit_address
        self.exchange = exchange

    def buy(self, pair,amount, **kwargs):
        return self.exchange.create_market_buy_order(pair,amount, params=kwargs)

    def sell(self, pair,amount, **kwargs):
        return self.exchange.create_market_sell_order(pair,amount, params=kwargs)

    def rebalance(self):
        self.exchange.withdraw('LTC', 0.02, 'LRW15TRkMvy5B4T1KivpWfKWq5e87UtP5E')