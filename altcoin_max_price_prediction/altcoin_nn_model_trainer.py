import shutil
import pandas as pd
import traceback
import datetime
import os
from lib.model import model_info, predict_simulation
from lib import mysql_helper
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import callbacks

"""
    ###################################################################################
    ###################################################################################
    ### if you want to execute this in terminal, you should put this command first. ###
    export PYTHONPATH=..
    ###################################################################################
    ###################################################################################

    Main purpose of this class is training the NN model.
"""

class Trainer():

    ticker_data_folder = './tools/ticker_data'
    training_data_folder = f'{ticker_data_folder}_training/'
    model_title_name = "TrainerA"
    input_without_split = ['last', 'usdt_btc_last']
    input_with_split = ['last', 'usdt_btc_last']

    input_ban = ['low', 'high', 'ask', 'bid', 'volume', 'base_volume', 'open_sell_orders', 'market_name', 'high',
                                       'open_buy_orders', 'time_stamp', 'prev_day', 'created']

    def __init__(self,
                 lr,
                 dense,
                 deep,
                 dropout,
                 batchsize,
                 epoch,
                 testing=False,
                 output_name=["last_max","last_max_index"],
                 history_size=180,
                 model_version ="a1",
                 model_group ="split",
                 input_group ="a1",
                 activation= 'relu',
                 last_activation='sigmoid',
                 prediction_result=False
                 ):

        if history_size is not None:
            self.history_size=history_size
        self.output_name = output_name
        self.output_columns_count = len(self.output_name)
        self.oneHotColumns = {}
        self.output_name_prediction = [
            x + "_prediction" for x in self.output_name
        ]
        self.start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.column_standard_values = {}
        self.input_column = []
        self.min_standard = []
        self.max_standard = []
        self.input_all = []
        self.input_training = []
        self.input_without_training = []
        self.model = None
        self.lr = lr
        self.dense = dense
        self.deep = deep
        self.dropout = dropout
        self.batchsize = batchsize
        self.epoch = epoch
        self.activation = activation
        self.last_activation = last_activation
        self.input_group = input_group
        self.model_group = model_group
        self.model_version = model_version
        self.testing = testing
        self.prediction_result = prediction_result
        output_name_str = ""
        for output_name_each in output_name:
            output_name_str = output_name_str + output_name_each
        self.model_name = f"{self.model_title_name}_{model_version}_{lr}_{dense}_{deep}_{dropout}_{batchsize}_{epoch}_{activation}_{last_activation}_{history_size}_{input_group}_{output_name_str}"
        self.root_folder_name = f"./tools/result"
        self.root_folder_name_output = f"{self.root_folder_name}/{self.model_title_name}_{model_version}/"
        self.output_folder_name = f"{self.root_folder_name_output}{self.model_name}/"

        if testing:
            if os.path.exists(self.output_folder_name):
                shutil.rmtree(self.output_folder_name)

        self.input_all = self.input_all + self.input_without_split

        for column_raw in self.input_with_split:
            for num in range(1, self.history_size+1):
                self.input_all.append(column_raw + "_old_" + str(num))

        self.input_training = list(set(self.input_all) - set(self.input_ban))
        self.input_training.sort()

    def start(self):
        try:
            print("start : ", self.start_datetime)
            print("start : ", self.model_name)

            self.session = mysql_helper.MysqlHelper.session

            self.create_folders()

            self.register_model_info()

            offset = 0

            try:
                while True:

                    data = pd.read_csv(self.training_data_folder + f"{str(offset)}.csv")

                    x, y = self.load(data)

                    training_x = x[self.input_training]
                    self.model, training_result, training_history = self.train(training_x, y, model=self.model)

                    if self.testing:
                        break

                    offset += 1

            except FileNotFoundError:
                pass

            self.save_model(self.model, training_result, training_history)

            if self.prediction_result:
                self.predict_for_validation(self.model)

            return

        except:

            print(traceback.format_exc())

            modelResult = {
                'status': 'error',
                'model_group': self.model_group,
                'comment': str(traceback.format_exc()),
                'learning_rate': self.lr,
                'dense': self.dense,
                'deep': self.deep,
                'dropout': self.dropout,
                'batchsize': self.batchsize,
                'epoch': self.epoch,
                'activation': self.activation,
                'last_activation': self.last_activation,
                'input_group': self.input_group,
                'output_name': str(self.output_name),
                'input_without_split': str(self.input_without_split),
                'input_with_split': str(self.input_with_split),
                'input_ban': str(self.input_ban),
                'model_name': self.model_name,
                'history_size' : self.history_size,
                'updated_at': self.start_datetime,
                'model_version' : self.model_version,
                'training_accuracy' : None,
                'validation_accuracy' : None,
                'training_loss' : None,
                'validation_loss' : None,
                'training_accuracy_last_5' : None,
                'validation_accuracy_last_5' : None,
                'training_loss_last_5' : None,
                'validation_loss_last_5' : None,
                'last_max_similar_acc' : None,
                'last_max_index_similar_acc' : None,
            }
            self.save_model_info(modelResult)
            pass

    def create_folders(self):

        if not os.path.exists(self.root_folder_name):
            os.makedirs(self.root_folder_name)

        if not os.path.exists(self.root_folder_name_output):
            os.makedirs(self.root_folder_name_output)

        if os.path.exists(self.output_folder_name):
            self.model = self.load_model()
        else:
            os.makedirs(self.output_folder_name)

    def register_model_info(self):

        model_info = {
            'id': None,
            'status': 'submitted',
            'model_group': self.model_group,
            'comment': None,
            'learning_rate': self.lr,
            'dense': self.dense,
            'deep': self.deep,
            'dropout': self.dropout,
            'batchsize': self.batchsize,
            'epoch': self.epoch,
            'activation': self.activation,
            'last_activation': self.last_activation,
            'input_group': self.input_group,
            'output_name': str(self.output_name),
            'input_without_split': str(self.input_without_split),
            'input_with_split': str(self.input_with_split),
            'input_ban': str(self.input_ban),
            'model_name': self.model_name,
            'history_size': self.history_size,
            'updated_at': self.start_datetime,
            'model_version': self.model_version,
            'training_accuracy': None,
            'validation_accuracy': None,
            'training_loss': None,
            'validation_loss': None,
            'training_accuracy_last_5': None,
            'validation_accuracy_last_5': None,
            'training_loss_last_5': None,
            'validation_loss_last_5': None,
            'last_max_similar_acc': None,
            'last_max_index_similar_acc': None,
        }
        self.save_model_info(model_info)

    def get_training_x(self, data):

        for columnRaw in self.input_with_split:
            oldData = data["old_" + columnRaw].split(",")
            data = pd.concat([data, oldData], axis=1)
        return data

    def load(self, allData):

        x = allData.iloc[:,:-2]
        y = allData[self.output_name]
        return x, y

    def train(self, X, y, model=None):

        if not model:

            model = Sequential()
            model.add(Dense(self.dense, input_dim=X.shape[1]))
            model.add(Activation(self.activation))
            for _ in range(0, self.deep):
                model.add(Dense(self.dense))
                model.add(Activation(self.activation))
                model.add(Dropout(self.dropout))
            model.add(Dense(self.output_columns_count))
            model.add(Activation(self.last_activation))

            sgd = SGD(lr=self.lr)
            model.compile(
                loss='binary_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

        tb_call_back = callbacks.TensorBoard(log_dir=self.output_folder_name + 'graph',
                                           write_graph=True,
                                           write_grads=True,
                                           )
        training_history = model.fit(
            X.values,
            y.values,
            batch_size=self.batchsize,
            nb_epoch=self.epoch,
            validation_split=0.2,
            callbacks=[tb_call_back])
        predict_y = pd.DataFrame(
            model.predict_proba(X.values), columns=self.output_name_prediction)
        training_result = pd.concat(
            (X, y, predict_y), axis=1)

        return model, training_result, training_history

    def predict_for_validation(self, model):

        data = pd.read_csv(self.ticker_data_folder + "/BTC-1ST.csv")
        predict_x_origin, y = self.load(data)
        one_hot_predict_x = predict_x_origin[self.input_training]
        prediction_raw = model.predict(one_hot_predict_x.values)
        prediction = pd.concat(
            [
                predict_x_origin,
                y,
                pd.DataFrame(
                    prediction_raw, columns=self.output_name_prediction)
            ],
            axis=1)

        self.save_prediction(prediction)

        return prediction

    def load_model(self):

        from keras.models import load_model
        model = load_model(self.output_folder_name + 'model.h5')
        model.load_weights(self.output_folder_name + 'model_weights.h5')

        return model

    def save_model(self, model, training_result, training_history):

        model.save(self.output_folder_name + 'model.h5')
        model.save_weights(self.output_folder_name + 'model_weights.h5', overwrite=True)

        training_history_dict = training_history.history
        val_loss = training_history_dict.get("val_loss", [None])
        val_acc = training_history_dict.get("val_acc", [None])
        acc = training_history_dict.get("acc", [None])
        loss = training_history_dict.get("loss", [None])
        val_loss_avg = sum(val_loss[-5:]) / 5 if len(val_loss) > 4 else val_loss[-1]
        val_acc_avg = sum(val_acc[-5:]) / 5 if len(val_acc) > 4 else val_acc[-1]
        acc_avg = sum(acc[-5:]) / 5 if len(acc) > 4 else acc[-1]
        loss_avg = sum(loss[-5:]) / 5 if len(loss) > 4 else loss[-1]

        model_info = {
            'status': 'finished',
            'model_group': self.model_group,
            'comment': None,
            'learning_rate': self.lr,
            'dense': self.dense,
            'deep': self.deep,
            'dropout': self.dropout,
            'batchsize': self.batchsize,
            'epoch': self.epoch,
            'activation': self.activation,
            'last_activation': self.last_activation,
            'input_group': self.input_group,
            'output_name': str(self.output_name),
            'input_without_split': str(self.input_without_split),
            'input_with_split': str(self.input_with_split),
            'input_ban': str(self.input_ban),
            'model_name': self.model_name,
            'history_size' : self.history_size,
            'updated_at': self.start_datetime,
            'training_accuracy': float(round(acc[-1], 4)),
            'validation_accuracy': float(round(val_acc[-1], 4)),
            'training_loss': float(round(loss[-1], 4)),
            'validation_loss': float(round(val_loss[-1], 4)),
            'training_accuracy_last_5': float(round(acc_avg, 4)),
            'validation_accuracy_last_5': float(round(val_acc_avg, 4)),
            'training_loss_last_5': float(round(loss_avg, 4)),
            'validation_loss_last_5': float(round(val_loss_avg, 4)),
            'model_version': self.model_version,
        }


        last_max_index_prediction_similar_count = 0
        last_max_index_prediction_not_similar_count = 1

        last_max_prediction_similar_count = 0
        last_max_prediction_not_similar_count = 1

        for index, row in training_result.iterrows():
            if "last_max" in self.output_name:
                if abs(row["last_max_prediction"] - row["last_max"]) < 0.01:
                    last_max_prediction_similar_count += 1
                else:
                    last_max_prediction_not_similar_count += 1
            if "last_max_index" in self.output_name:
                if abs(row["last_max_index_prediction"] - row["last_max_index"]) < 0.001:
                    last_max_index_prediction_similar_count += 1
                else:
                    last_max_index_prediction_not_similar_count += 1


        model_info['last_max_similar_acc'] = round(last_max_prediction_similar_count / (last_max_prediction_similar_count + last_max_prediction_not_similar_count), 4)
        model_info['last_max_index_similar_acc'] = round(last_max_index_prediction_similar_count / (last_max_index_prediction_similar_count + last_max_index_prediction_not_similar_count), 4)

        self.save_model_info(model_info)

    def save_model_info(self, data):

        try:
            ad = model_info(**data)
            self.session.add(ad)
            self.session.commit()
        except:
            print(traceback.format_exc())
            pass

    def save_prediction(self, predictions):

        for index, prediction in predictions.iterrows():
            data = prediction[list(set(predict_simulation.__table__.columns.keys()) - set(['id']))]

            if "last_max" in self.output_name:

                if abs(prediction["last_max_prediction"] - prediction["last_max"]) < 0.01:
                    data["last_max_prediction_similar"] = 1
                else:
                    data["last_max_prediction_similar"] = 0
            else:
                data["last_max_prediction_similar"] = None
                data["last_max_prediction"] = None
                data["last_max"] = None

            if "last_max_index" in self.output_name:

                if abs(prediction["last_max_index_prediction"] - prediction["last_max_index"]) < 0.001:
                    data["last_max_index_prediction_similar"] = 1
                else:
                    data["last_max_index_prediction_similar"] = 0
            else:
                data["last_max_index_prediction_similar"] = None
                data["last_max_index_prediction"] = None
                data["last_max_index"] = None

            data['model_name'] = self.model_name

            try:
                ad = predict_simulation(**data)
                self.session.add(ad)
                self.session.commit()
            except:
                print(traceback.format_exc())
                pass

if __name__ == "__main__":

    import sys

    args = sys.argv[1:]
    if len(args) > 1:
        lr, dense, deep, dropout, batchsize, epoch, activation, last_activation, history_size, model_version, inputGroup, model_group, prediction_result_raw, output_name_raw = args

        if "predict" in prediction_result_raw:
            prediction_result = True
        else:
            prediction_result = False

        print("output_name_raw : ", output_name_raw)
        output_name = []
        if "index" in output_name_raw:
            output_name.append("last_max_index")
        if "last_max" in output_name_raw:
            output_name.append("last_max")
        if len(output_name) == 0:
            output_name = ["last_max_index","last_max"]
        Trainer(float(lr), int(dense), int(deep), float(dropout), int(batchsize), int(epoch),
                model_version=model_version,
                model_group=model_group, output_name=output_name, history_size=int(history_size),
                activation=activation, last_activation=last_activation, prediction_result=prediction_result).start()

    # Trainer(0.02, 50, 10, 0.2, 32, 300, testing=True).start()
