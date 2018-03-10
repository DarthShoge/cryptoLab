import ccxt
import pandas as pd

from arblab.arb_lab import find_arb_opportunities, get_mkt_participants

pair = 'ETH/USD'

exchanges = get_mkt_participants(pair)

v = find_arb_opportunities(exchanges, pair)