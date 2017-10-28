#!/bin/bash
. ./p3.6/bin/activate
export PYTHONPATH=.

python altcoin_max_price_prediction/altcoin_nn_model_trainer.py 0.02 1 10 0.2 32 1 relu sigmoid 3 1 g1 s1 predict index
# python altcoin_max_price_prediction/altcoin_nn_model_trainer.py 0.02 2500 10 0.2 32 300 relu sigmoid 360 1 g1 s1 predict index
# python altcoin_max_price_prediction/altcoin_nn_model_trainer.py 0.02 250 10 0.2 32 300 relu sigmoid 360 1 g1 s1 predict last_min
