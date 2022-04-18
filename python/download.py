#!/usr/bin/env python3

import sys
import pandas as pd
import alpaca_trade_api as alpaca
from datetime import datetime

api_key_id = open("ALPACA_API_KEY_ID").readline()[:-1]
api_secret = open("ALPACA_SECRET_KEY").readline()[:-1]
base_url = "https://paper-api.alpaca.markets"
api = alpaca.REST(key_id=api_key_id, secret_key=api_secret, base_url=alpaca.rest.URL(base_url))

# --- #

def historical(ticker:str):
    df = pd.DataFrame(api.get_bars(ticker, alpaca.rest.TimeFrame.Minute, "2000-01-01", datetime.today().strftime("%Y-%m-%d")).df)
    # save close value to ./temp/dataset

#def real_time(ticker:str):

# --- #

if __name__ == "__main__":
    data_type = sys.argv[1] # "historical" or "realtime" (1 minute bars)
    ticker    = sys.argv[2]

    if data_type == "historical":
        historical(ticker)
    elif data_type == "realtime":
        pass
