import traceback
import datetime
import re
import time
from lib.model import ticker_data
from lib import bittrex_api, mysql_helper

"""
Main perpose of this class is parsing ticker(trade) data for 24 hours from bitcoin exchange API service

such as bittrex and poloniex.

I used bittrex to parse ticker data but if somebody want to parse from other API service,

then simply you can add the code and replace from ticker_api variable.
"""

class GetTradeData():

    start_time = time.time()

    def start(self):

        self.session = mysql_helper.MysqlHelper.session
        ticker_api = bittrex_api.BittrexApi()

        while True:
            start_parsing_time = time.time()

            tickers_raw = ticker_api.parse_ticker()
            tickers = self.parse_ticker_info(tickers_raw)
            self.save_ticker_info(tickers)

            end_parsing_time = time.time()

            self.wait_one_minute(start_parsing_time, end_parsing_time)

            if end_parsing_time - self.start_time > 1440: # parsing for 24 hours only.
                break

    def parse_ticker_info(self, tickers_raw):
        update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ticker_infos = []
        for ticker_raw in tickers_raw:
            ticker_info_raw = ticker_raw.get("Summary",{})
            ticker_info = self.get_snake_name(ticker_info_raw)
            ticker_info['updated_at'] = update_time
            ticker_infos.append(ticker_info)

        return ticker_infos


    def save_ticker_info(self, tickers):

        for ticker in tickers:
            try:
                ad = ticker_data(**ticker)
                self.session.add(ad)
                self.session.commit()
            except:
                print(traceback.format_exc())
                pass

    def wait_one_minute(self, start_time, end_time):
        wait_time = 60 - (end_time - start_time)
        print("waiting time : " , wait_time)
        if wait_time > 0:
            time.sleep(wait_time)

    def get_snake_name(self, tickerInfoRaw):
        tickerInfoNew = {}
        for key, value in tickerInfoRaw.items():
            tickerInfoNew[self.camel_to_snake(key)] = value
        return tickerInfoNew

    def camel_to_snake(self, name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

if __name__ == "__main__":
    GetTradeData().start()