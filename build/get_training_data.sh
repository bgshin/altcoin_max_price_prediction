#!/bin/bash
sudo apt install virtualenv -y
virtualenv -p python3.6 p3.6
. ./p3.6/bin/activate
pip install -r requirements.txt -I
export PYTHONPATH=.
python altcoin_max_price_prediction/make_mysql_data_structure.py
python altcoin_max_price_prediction/get_trade_data.py
python altcoin_max_price_prediction/preprocess_trade_data.py