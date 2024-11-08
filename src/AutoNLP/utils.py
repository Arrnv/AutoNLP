import os 
import sys
from src.AutoNLP.exception import CustomException
from src.AutoNLP.logging import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('db')

def read_sql_data():
    logging.info("Reading SQL Database")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info(f"Connection Established {mydb}")
        df=pd.read_sql_query('Select * from user_behavior_dataset',mydb)
        print(df.head())
        
        return df
    except Exception as e:
        raise CustomException(e,sys)