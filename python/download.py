#!/usr/bin/env python3


import alpaca_trade_api as alpaca
import pandas as pd

api_key_id = os.environ.get("ALPACA_API_KEY_ID")
api_secret = os.environ.get("ALPACA_API_SECRET_KEY")
base_url = "https://paper-api.alpaca.markets"

api = alpaca.REST(key_id=api_key_id, secret_key=api_secret, base_url=alpaca.rest.URL(base_url))

df = pd.DataFrame(api.get_bars("MSFT", alpaca.rest.TimeFrame.Minute, "1990-01-01", "2022-02-18").df)

