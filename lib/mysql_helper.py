from sqlalchemy import create_engine
from sqlalchemy.schema import CreateSchema
from sqlalchemy.orm import sessionmaker
from lib.model import Base
import yaml

class MysqlHelper():

    with open("mysql_credentials.yaml", 'r') as stream:
        try:
            credentials = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    host = credentials['host']
    port = credentials['port']
    id = credentials['id']
    pw = credentials['pw']
    schema = credentials['schema']

    try:
        engine = create_engine(f'mysql://{id}:{pw}@{host}:{port}')
        engine.execute(CreateSchema(schema))
    except:
        pass

    engine = create_engine(f'mysql://{id}:{pw}@{host}:{port}/{schema}')
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()