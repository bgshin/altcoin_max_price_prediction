import traceback
import pandas as pd
import numpy as np
from lib.model import ticker_data
from lib import mysql_helper
from tools.market_name import MarketName
import os

"""
    ###################################################################################
    ###################################################################################
    ### if you want to execute this in terminal, you should put this command first. ###
    export PYTHONPATH=..
    ###################################################################################
    ###################################################################################

    Main purpose of this class is preprocessing trade data.

"""
class PreprocessTradeData():

    ticker_data_folder = './tools/ticker_data'
    training_data_folder = f'{ticker_data_folder}_training/'
    history_size = 720 # 12 hours
    future_size = 360 # 6 hours
    split_column_names = ['last']
    usd_column_name = 'usdt_btc_last'

    def start(self):

        try:
            self.mysql_session = mysql_helper.MysqlHelper.session

            market_names = MarketName.market_name

            self.make_data_folder()

            ticker_infos_usd = self.get_ticker_infos_usd()

            for market_name in market_names:
                try:
                    print(market_name)

                    ticker_info = self.get_ticker_infos_alt(market_name, ticker_infos_usd)

                    ticker_info.to_csv(f"{self.ticker_data_folder}/{market_name}.csv", index=False)
                except:
                    pass

        except:
            print(traceback.format_exc())
            pass

        self.ticker_data_mix()

    def make_data_folder(self):

        if not os.path.exists(self.ticker_data_folder):
            os.makedirs(self.ticker_data_folder)

        if not os.path.exists(self.training_data_folder):
            os.makedirs(self.training_data_folder)

    def get_ticker_infos_usd(self):

        ticker_infos_usd_raw = self.get_ticker_data("USDT-BTC")
        ticker_infos_usd_raw[self.usd_column_name] = (1 / ticker_infos_usd_raw['last']).round(8)  # BTC / USD
        ticker_infos_usd_raw_with_old = self.get_old(
            ticker_infos_usd_raw, [self.usd_column_name], reset=True)
        ticker_infos_usd_raw_total = pd.concat(
            [ticker_infos_usd_raw[self.usd_column_name], ticker_infos_usd_raw_with_old], axis=1)

        return ticker_infos_usd_raw_total

    def get_ticker_infos_alt(self, market_name, ticker_infos_usd):

        ticker_info_raw = self.get_ticker_data(market_name)

        ticker_info_with_old = self.get_old(
            ticker_info_raw, self.split_column_names)

        ticker_info_with_old_and_usd = self.ticker_merge_with_usd(
            ticker_info_with_old, ticker_infos_usd)

        ticker_info_total_raw = self.get_new(ticker_info_with_old_and_usd)

        ticker_info_total = self.remove_no_calculated_old_and_new_data(
            ticker_info_total_raw)

        return ticker_info_total

    def get_ticker_data(self, market_name):

        return pd.read_sql(self.mysql_session.query(ticker_data)
                           .filter(ticker_data.market_name == market_name)
                           .statement, self.mysql_session.bind)

    def get_old(
            self,
            data,
            parameters,
            reset=False,
            ):
        if reset:
            new_data = pd.DataFrame([])
        else:
            new_data = data

        for col in parameters:
            col_value = data[col].values

            for old_num in range(1, self.history_size + 1):
                old = self._get_old(col_value, old_num)
                col_name = col + "_old_" + str(old_num)
                new_data[col_name] = old.round(5)

        return new_data

    def _get_old(self, data, num):

        new_data = data
        old = np.append([0] * num, new_data)[:-num]
        dif = np.subtract(new_data, old)
        dif_percentage = dif / new_data

        return dif_percentage

    def get_new(self, data):

        new_data = pd.DataFrame([])
        new_data.append(data)

        col_value = data["last"].values

        for new_num in range(1, self.future_size + 1):
            new = self._get_new(col_value, new_num)
            new_data["last_new_" + str(new_num)] = new.round(5)

        output_data = new_data.apply(self.get_output, axis=1)[
            ["last_max_index", "last_max"]]

        return pd.concat([data, output_data], axis=1)

    def get_output(self, data):

        max = -100
        max_index = 0

        for new_num in range(1, self.future_size + 1):
            compare_last = data["last_new_" + str(new_num)]
            if compare_last > max:
                max = compare_last
                max_index = new_num

        max_index = (max_index - (self.future_size / 2)) / (self.future_size / 2)

        return pd.Series([max_index, max], index=["last_max_index", "last_max"])

    def _get_new(self, data, num):

        new_data = data
        new = np.append(new_data, [0] * num)[num:]
        dif = np.subtract(new, new_data)
        dif_percentage = dif / new_data
        return dif_percentage

    def remove_no_calculated_old_and_new_data(self, data):

        new_data = data
        return new_data.iloc[self.history_size + 1: -
                            self.future_size].reset_index(drop=True)

    def ticker_merge_with_usd(self, ticker_info, ticker_info_usd):

        if ticker_info_usd.shape[0] > ticker_info.shape[0]:
            row_count_dif = ticker_info_usd.shape[0] - ticker_info.shape[0]
            ticker_info_usd = ticker_info_usd.iloc[:-row_count_dif]
        elif ticker_info_usd.shape[0] < ticker_info.shape[0]:
            row_count_dif = ticker_info.shape[0] - ticker_info_usd.shape[0]
            ticker_info = ticker_info.iloc[:-row_count_dif]

        return pd.concat([ticker_info, ticker_info_usd], axis=1)
    
    def ticker_data_mix(self):

        offset = 0
        while True:

            data = pd.DataFrame([])
            data_for_training = pd.DataFrame([])

            print(offset)

            for file in os.listdir(self.ticker_data_folder):

                if not '.csv' in file:
                    continue

                readData = pd.read_csv(self.ticker_data_folder + "/" + file, nrows=((offset + 1) * 50))

                readData = readData.loc[(offset * 50):]

                data = pd.concat([data, readData], ignore_index=True, axis=0)
                data_last_only = data.ix[:, 5]
                data_old_indicators = data.ix[:, 15:]
                data_for_training = pd.concat([data_last_only, data_old_indicators], axis=1)

            if data.shape[0] == 0:
                break

            data_for_training.to_csv(f"{self.training_data_folder}{offset}.csv", index=False)

            offset += 1
            
if __name__ == "__main__":
    PreprocessTradeData().start()
