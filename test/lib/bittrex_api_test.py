import json
import unittest
from lib import bittrex_api

class BittrexApiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.main_class = bittrex_api.BittrexApi()

    def test_parse_ticker(self):
        self.assertIsNotNone(self.main_class.parse_ticker()[0].get("Market").get("MarketCurrency"))


if __name__ == '__main__':
    unittest.main()
