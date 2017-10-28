import unittest
from altcoin_max_price_prediction import get_trade_data

class GetTradeDataTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.main_class = get_trade_data.GetTradeData()
        cls.sample_data = [{'Market': {'MarketCurrency': '1ST', 'BaseCurrency': 'BTC',
                                       'MarketCurrencyLong': 'FirstBlood', 'BaseCurrencyLong': 'Bitcoin',
                                       'MinTradeSize': 1e-08, 'MarketName': 'BTC-1ST', 'IsActive': True,
                                       'Created': '2017-06-06T01:22:35.727', 'Notice': None, 'IsSponsored': None,
                                       'LogoUrl': 'https://bittrexblobstorage.blob.core.windows.net/public/5685a7be-1edf-4ba0-a313-b5309bb204f8.png'},
                            'Summary': {'MarketName': 'BTC-1ST', 'High': 5.498e-05, 'Low': 4.39e-05,
                                        'Volume': 998942.19871364, 'Last': 5.489e-05, 'BaseVolume': 49.55246742,
                                        'TimeStamp': '2017-10-27T14:35:01.16', 'Bid': 5.382e-05, 'Ask': 5.45e-05,
                                        'OpenBuyOrders': 229, 'OpenSellOrders': 5857, 'PrevDay': 4.631e-05,
                                        'Created': '2017-06-06T01:22:35.727'}, 'IsVerified': False}]

    def test_parse_ticker(self):
        self.assertIsNotNone(self.main_class.parse_ticker()[0].get("Market").get("MarketCurrency"))

    def test_parse_ticker_info(self):
        self.assertEqual(self.main_class.parse_ticker_info(self.sample_data)[0].get("market_name"), "BTC-1ST")

    def test_get_snake_name(self):
        self.assertEqual(self.main_class.get_snake_name(self.sample_data[0].get("Summary")).get("market_name"), "BTC-1ST")

    def test_camel_to_snake(self):
        self.assertEqual(self.main_class.camel_to_snake("marketName"), "market_name")

if __name__ == '__main__':
    unittest.main()
