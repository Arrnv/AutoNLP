import sys
from src.Mobile_Device_Usage.logging import logging


def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in Python Script name [{0}] Line No. [{1}] Error Message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_massage, error_details:sys):
        super().__init__(error_massage)
        self.error_massage = error_message_detail(error_massage, error_details)
    
    def __str__(self):
        return self.error_massage