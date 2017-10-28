import requests
import json
import datetime

class BittrexApi():

    def parse_ticker(self):

        url = "https://bittrex.com/api/v2.0/pub/Markets/GetMarketSummaries"

        headers = {
            'cache-control': "no-cache",
            }

        responseRaw = requests.request("GET", url, headers=headers)
        response = json.loads(responseRaw.text)
        if response.get("success"):
            print(datetime.datetime.now(), ": success")
        else:
            raise ValueError('parsing was not success')

        return response.get('result',[])