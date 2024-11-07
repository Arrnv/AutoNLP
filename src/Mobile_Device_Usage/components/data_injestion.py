import os 
import sys
from src.Mobile_Device_Usage.exception import CustomException
from src.Mobile_Device_Usage.logging import logging
import pandas as pd
from src.Mobile_Device_Usage.utils import read_sql_data
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataInjestionConfig:
    train_data_path:str = os.path.join('artifact','Train.csv')
    test_data_path:str = os.path.join('artifact','Test.csv')
    raw_data_path:str = os.path.join('artifact','Raw.csv')
    
class DataInjestion:
    def __init__(self):
        self.injestion_config = DataInjestionConfig()
    
    def Initiate_data_injestion(self):
        try:
            df = read_sql_data()
            logging.info("Reading data from mysql Completed")
            os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.injestion_config.raw_data_path, index=False, header=True)
            train_set, Test_set= train_test_split(df, test_size=0.2,random_state=42)
            train_set.to_csv(self.injestion_config.train_data_path, index=False, header=True)
            Test_set.to_csv(self.injestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data injestion is complete")
            
            return (
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)