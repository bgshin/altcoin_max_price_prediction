# altcoin_max_price_prediction
Predict altcoin's max price reach time and value using neural network based on bitcoin exchange market data.

## Requires

ubuntu 16.04 (I didn't test with other OS / Version.)

Python3.6

Mysql 5.7+

Python libs in requirements.txt

## Main purpose

The main purpose is making neural network model to predict the max price reach time from altcoin exchange market.

Let's see one of the altcoin price time graph.

![](docs/purpose.png?raw=true)

Based on 12 hours history data, the NN model will predict "max price reach time index" and "max price value" in future 6 hours.

If we can predict what time altcoin price can reach the max value during specific times and how much price will be then it means we can make a money based on this NN model.

## how to setup?

First, clone this repository.

```bash
git clone https://github.com/SkyHenryk/altcoin_max_price_prediction.git
```

Second, write mysql credentials information in "mysql_credentials.yaml".

```
host:
port:
id:
pw:
schema: quant_db
```

Third, get training data during 24 hours(1 day). It will get Bittrex trade data every minute.

After 24 hours, it will prepare training data based on trade data.

```bash
cd altcoin_max_price_prediction
bash build/get_training_data.sh
```

Fourth, start to training the neural network model based on prepared training data.

```bash
bash build/train_model.sh
```

Done! You can find the model result in "model_info" table in "quant_db" schema in Mysql.

Or, You can check the graph using tensorboard.

ex)

```
tensorboard --logdir ./tools/result/TrainerA_1/TrainerA_1_0.02_2500_10_0.2_32_300_relu_sigmoid_360_a1_last_max_index/graph
tensorboard --logdir ./tools/result/TrainerA_1/TrainerA_1_0.02_250_10_0.2_32_300_relu_sigmoid_360_a1_last_max/graph
```

![](docs/tb.png?raw=true)
