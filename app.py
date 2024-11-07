from src.Mobile_Device_Usage.logging import logging
from src.Mobile_Device_Usage.exception import CustomException
from src.Mobile_Device_Usage.components.data_injestion import DataInjestion, DataInjestionConfig
import sys

if __name__ == "__main__":
    logging.info("The executation have started")
    
    try:
        # data_injestion_config = DataInjestionConfig()
        data_injestion = DataInjestion()
        data_injestion.Initiate_data_injestion()
        
    except Exception as e:
        logging.info("custom Exception")
        raise CustomException(e, sys)