
from lib.model import ticker_data, model_info, predict_simulation, Base
from lib import mysql_helper

"""
Main purpose of this is making mysql table to store trade data.
"""
class MakeMysqlTable():

    def start(self):
        engine = mysql_helper.MysqlHelper.engine
        Base.metadata.bind = engine
        ticker_data.__table__.create(bind = engine)
        model_info.__table__.create(bind = engine)
        predict_simulation.__table__.create(bind = engine)

if __name__ == "__main__":
    MakeMysqlTable().start()