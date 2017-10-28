import unittest
import pandas as pd
from altcoin_max_price_prediction import preprocess_trade_data

class PreprocessTradeDataTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.main_class = preprocess_trade_data.PreprocessTradeData()
        cls.main_class.history_size = 3
        cls.main_class.future_size = 3
        cls.sample_data_usd = [
            {'id': 253, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3142.91636975,
             'last': 5709.62, 'base_volume': 17860090.630093, 'time_stamp': '2017-10-16 23:04:02',
             'bid': 5709.62, 'ask': 5719.9, 'open_buy_orders': 7744, 'open_sell_orders': 3574, 'prev_day': 5745.0,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:04:08'},
            {'id': 515, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3134.80686624,
             'last': 5709.62, 'base_volume': 17813438.7573089, 'time_stamp': '2017-10-16 23:05:02',
             'bid': 5705.00000002, 'ask': 5719.9, 'open_buy_orders': 7746, 'open_sell_orders': 3574, 'prev_day': 5748.0,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:05:08'},
            {'id': 777, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3129.40685073,
             'last': 5710.0, 'base_volume': 17782400.6018975, 'time_stamp': '2017-10-16 23:06:03',
             'bid': 5713.0, 'ask': 5719.9, 'open_buy_orders': 7747, 'open_sell_orders': 3581, 'prev_day': 5722.03723499,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:06:08'},
            {'id': 1039, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3126.59574809,
             'last': 5719.0, 'base_volume': 17766282.7100516, 'time_stamp': '2017-10-16 23:07:03',
             'bid': 5713.0, 'ask': 5719.0, 'open_buy_orders': 7752, 'open_sell_orders': 3583, 'prev_day': 5725.0,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:07:08'},
            {'id': 1301, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3130.87751397,
             'last': 5723.9, 'base_volume': 17790785.9523331, 'time_stamp': '2017-10-16 23:07:56',
             'bid': 5713.0, 'ask': 5723.9, 'open_buy_orders': 7746, 'open_sell_orders': 3583, 'prev_day': 5722.00742303,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:08:08'},
            {'id': 1563, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3131.2529282,
             'last': 5723.9, 'base_volume': 17792925.8450579, 'time_stamp': '2017-10-16 23:08:56',
             'bid': 5710.00000003, 'ask': 5723.9, 'open_buy_orders': 7748, 'open_sell_orders': 3586, 'prev_day': 5740.0,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:09:08'},
            {'id': 1825, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3133.34108944,
             'last': 5722.9, 'base_volume': 17804873.9598486, 'time_stamp': '2017-10-16 23:10:04',
             'bid': 5715.0, 'ask': 5722.9, 'open_buy_orders': 7750, 'open_sell_orders': 3589, 'prev_day': 5722.00000001,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:10:08'},
            {'id': 2087, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3136.80773744,
             'last': 5723.9, 'base_volume': 17824697.2302864, 'time_stamp': '2017-10-16 23:11:05',
             'bid': 5722.7, 'ask': 5723.9, 'open_buy_orders': 7750, 'open_sell_orders': 3585, 'prev_day': 5725.01,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:11:08'},
            {'id': 2349, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3144.62472261,
             'last': 5716.16200005, 'base_volume': 17869370.704901, 'time_stamp': '2017-10-16 23:12:08',
             'bid': 5716.16200005, 'ask': 5725.0, 'open_buy_orders': 7751, 'open_sell_orders': 3573, 'prev_day': 5735.0,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:12:08'},
            {'id': 2611, 'market_name': 'USDT-BTC', 'high': 5756.0, 'low': 5565.25000001, 'volume': 3142.805971,
             'last': 5735.0, 'base_volume': 17858922.0652518, 'time_stamp': '2017-10-16 23:13:07',
             'bid': 5727.00000001, 'ask': 5735.0, 'open_buy_orders': 7753, 'open_sell_orders': 3561, 'prev_day': 5740.0,
             'created': '2015-12-11 06:31:40', 'updated_at': '2017-10-17 01:13:08'}]

        cls.sample_data = [
            {'id': 106, 'market_name': 'BTC-MUSIC', 'high': 3.74e-06, 'low': 3.21e-06, 'volume': 7769724.20671862,
             'last': 3.32e-06, 'base_volume': 26.12908177, 'time_stamp': '2017-10-16 23:04:00',
             'bid': 3.32e-06, 'ask': 3.33e-06, 'open_buy_orders': 257, 'open_sell_orders': 8848, 'prev_day': 3.74e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:04:08'},
            {'id': 368, 'market_name': 'BTC-MUSIC', 'high': 3.71e-06, 'low': 3.21e-06, 'volume': 7767876.50128578,
             'last': 3.34e-06, 'base_volume': 26.1216032, 'time_stamp': '2017-10-16 23:04:58',
             'bid': 3.32e-06, 'ask': 3.34e-06, 'open_buy_orders': 257, 'open_sell_orders': 8824, 'prev_day': 3.71e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:05:08'},
            {'id': 630, 'market_name': 'BTC-MUSIC', 'high': 3.71e-06, 'low': 3.21e-06, 'volume': 7761950.97742083,
             'last': 3.38e-06, 'base_volume': 26.0989871, 'time_stamp': '2017-10-16 23:06:08',
             'bid': 3.32e-06, 'ask': 3.38e-06, 'open_buy_orders': 258, 'open_sell_orders': 8845, 'prev_day': 3.71e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:06:08'},
            {'id': 892, 'market_name': 'BTC-MUSIC', 'high': 3.71e-06, 'low': 3.21e-06, 'volume': 7761950.97742083,
             'last': 3.38e-06, 'base_volume': 26.0989871, 'time_stamp': '2017-10-16 23:06:08',
             'bid': 3.32e-06, 'ask': 3.38e-06, 'open_buy_orders': 258, 'open_sell_orders': 8845, 'prev_day': 3.71e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:07:08'},
            {'id': 1154, 'market_name': 'BTC-MUSIC', 'high': 3.71e-06, 'low': 3.21e-06, 'volume': 7761950.97742083,
             'last': 3.38e-06, 'base_volume': 26.0989871, 'time_stamp': '2017-10-16 23:07:47',
             'bid': 3.33e-06, 'ask': 3.38e-06, 'open_buy_orders': 258, 'open_sell_orders': 8845, 'prev_day': 3.71e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:08:08'},
            {'id': 1416, 'market_name': 'BTC-MUSIC', 'high': 3.71e-06, 'low': 3.21e-06, 'volume': 7761950.97742083,
             'last': 3.38e-06, 'base_volume': 26.0989871, 'time_stamp': '2017-10-16 23:07:47',
             'bid': 3.33e-06, 'ask': 3.38e-06, 'open_buy_orders': 258, 'open_sell_orders': 8845, 'prev_day': 3.71e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:09:08'},
            {'id': 1678, 'market_name': 'BTC-MUSIC', 'high': 3.71e-06, 'low': 3.21e-06, 'volume': 7761950.97742083,
             'last': 3.38e-06, 'base_volume': 26.0989871, 'time_stamp': '2017-10-16 23:07:47',
             'bid': 3.33e-06, 'ask': 3.38e-06, 'open_buy_orders': 258, 'open_sell_orders': 8845, 'prev_day': 3.71e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:10:08'},
            {'id': 1940, 'market_name': 'BTC-MUSIC', 'high': 3.71e-06, 'low': 3.21e-06, 'volume': 7761950.97742083,
             'last': 3.38e-06, 'base_volume': 26.0989871, 'time_stamp': '2017-10-16 23:07:47',
             'bid': 3.33e-06, 'ask': 3.38e-06, 'open_buy_orders': 258, 'open_sell_orders': 8845, 'prev_day': 3.71e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:11:08'},
            {'id': 2202, 'market_name': 'BTC-MUSIC', 'high': 3.69e-06, 'low': 3.21e-06, 'volume': 7746010.66199945,
             'last': 3.38e-06, 'base_volume': 26.04065161, 'time_stamp': '2017-10-16 23:11:26',
             'bid': 3.33e-06, 'ask': 3.38e-06, 'open_buy_orders': 265, 'open_sell_orders': 8855, 'prev_day': 3.59e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:12:08'},
            {'id': 2464, 'market_name': 'BTC-MUSIC', 'high': 3.69e-06, 'low': 3.21e-06, 'volume': 7746180.1648616,
             'last': 3.33e-06, 'base_volume': 26.04121605, 'time_stamp': '2017-10-16 23:12:44',
             'bid': 3.33e-06, 'ask': 3.37e-06, 'open_buy_orders': 263, 'open_sell_orders': 8856, 'prev_day': 3.59e-06,
             'created': '2017-03-27 19:59:13', 'updated_at': '2017-10-17 01:13:08'}]



    def test_get_old(self):
        sample_df = pd.DataFrame(self.sample_data_usd)
        sample_df['usdt_btc_last'] = (1 / sample_df['last']).round(8)
        sample_df_with_old = self.main_class.get_old(sample_df,['usdt_btc_last'])
        self.assertEqual(sample_df_with_old['usdt_btc_last_old_1'].iloc[0], 1)

    def test_get_new(self):
        sample_df = pd.DataFrame(self.sample_data)
        sample_df_with_new = self.main_class.get_new(sample_df)
        self.assertEqual(sample_df_with_new['last_max'].iloc[0], 0.01807)

    def test_remove_no_calculated_old_and_new_data(self):
        sample_df = pd.DataFrame(self.sample_data)
        sample_df_with_new = self.main_class.remove_no_calculated_old_and_new_data(sample_df)
        self.assertEqual(sample_df_with_new.shape[0], 3)

    def test_ticker_merge_with_usd(self):
        sample_df_usd = pd.DataFrame(self.sample_data_usd)
        sample_df = pd.DataFrame(self.sample_data)
        sample_df_merge = self.main_class.ticker_merge_with_usd(sample_df, sample_df_usd)
        self.assertEqual(sample_df_merge.shape[0], 10)

if __name__ == '__main__':
    unittest.main()
