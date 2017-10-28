from sqlalchemy import Column, Integer, VARCHAR, DateTime, DECIMAL, Text, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

import pymysql
pymysql.install_as_MySQLdb()
Base = declarative_base()


class ticker_data(Base):
    __tablename__ = 'ticker_data'

    id = Column(Integer, primary_key=True)
    market_name = Column(VARCHAR(30))
    high = Column(DECIMAL(16,8))
    low = Column(DECIMAL(16,8))
    volume = Column(DECIMAL(25,8))
    last = Column(DECIMAL(16,8))
    base_volume = Column(DECIMAL(30,8))
    time_stamp = Column(DateTime())
    bid = Column(DECIMAL(16,8))
    ask = Column(DECIMAL(16,8))
    open_buy_orders = Column(Integer())
    open_sell_orders = Column(Integer())
    prev_day = Column(DECIMAL(16,8))
    created = Column(DateTime())
    updated_at = Column(DateTime())


class model_info(Base):
    __tablename__ = 'model_info'
    id = Column(Integer, primary_key=True)
    updated_at = Column(DateTime())
    model_name = Column(VARCHAR(100))
    model_group = Column(VARCHAR(30))
    status = Column(VARCHAR(30))
    comment = Column(Text())
    learning_rate = Column(DECIMAL(16,8))
    dense = Column(Integer())
    deep = Column(Integer())
    dropout = Column(DECIMAL(16,8))
    batchsize = Column(Integer())
    epoch = Column(Integer())
    activation = Column(VARCHAR(30))
    last_activation = Column(VARCHAR(30))
    input_group = Column(VARCHAR(30))
    input_without_split = Column(Text())
    input_with_split = Column(Text())
    input_ban = Column(Text())
    output_name = Column(VARCHAR(30))
    training_accuracy = Column(DECIMAL(8,4))
    validation_accuracy = Column(DECIMAL(8,4))
    training_loss = Column(DECIMAL(8,4))
    validation_loss = Column(DECIMAL(8,4))
    training_accuracy_last_5 = Column(DECIMAL(8,4))
    validation_accuracy_last_5 = Column(DECIMAL(8,4))
    training_loss_last_5 = Column(DECIMAL(8,4))
    validation_loss_last_5 = Column(DECIMAL(8,4))
    last_max_similar_acc = Column(DECIMAL(8,4))
    last_max_index_similar_acc = Column(DECIMAL(8,4))
    history_size = Column(Integer())
    model_version = Column(VARCHAR(30))


class predict_simulation(Base):
    __tablename__ = 'predict_simulation'

    id = Column(Integer, primary_key=True)
    model_name = Column(VARCHAR(100))
    market_name = Column(VARCHAR(30))
    high= Column(DECIMAL(16,8))
    low= Column(DECIMAL(16,8))
    volume= Column(DECIMAL(25,8))
    last= Column(DECIMAL(16,8))
    base_volume= Column(DECIMAL(16,8))
    time_stamp = Column(DateTime())
    bid = Column(DECIMAL(16,8))
    ask= Column(DECIMAL(16,8))
    open_buy_orders = Column(Integer())
    open_sell_orders = Column(Integer())
    prev_day = Column(DECIMAL(16,8))
    created = Column(DateTime())
    last_max = Column(DECIMAL(16,8))
    last_max_index = Column(DECIMAL(16,8))
    last_max_prediction = Column(DECIMAL(16,8))
    last_max_index_prediction = Column(DECIMAL(16,8))
    last_max_prediction_similar = Column(Integer())
    last_max_index_prediction_similar = Column(Integer())
