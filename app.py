from src.AutoNLP.logging import logging
from src.AutoNLP.exception import CustomException
import sys
from src.AutoNLP.components.Preprocessing import Preprocessor
import pandas as pd
data = pd.DataFrame({
    'text_column': [
        'This is a test sentence.',
        'Another example of text!',
        'NLP preprocessing is essential.',
        'Testing the pipeline functions.'
    ],
    'target': [0, 1, 0, 1]
})

df = pd.read_csv('iphone.csv')

def run_pipeline(data, text_col:str, clean_text=True, tokenize=True):
            logging.info("Starting pipeline...")
            preprocessor = Preprocessor()
            data[text_col] = data[text_col].fillna('')
            if tokenize==True and clean_text==True:    
                data['processed_text'] = data[text_col].apply(preprocessor.clean_text)
                data['tockenized_data'] = data['processed_text'].apply(preprocessor.tokenize_and_lemmatize)
            elif tokenize==True and clean_text==False: 
                data['tockenized_data'] = data['processed_text'].apply(preprocessor.tokenize_and_lemmatize)
            elif tokenize==False and clean_text==True:    
                 data['processed_text'] = data[text_col].apply(preprocessor.clean_text)
            print(data.head())
            logging.info("Preprocessing completed.")

if __name__ == "__main__":
    # Taking input csv
    try:
        run_pipeline(df, text_col='reviewDescription', clean_text=True, tokenize=False)
        run_pipeline(df, text_col='processed_text', clean_text=False, tokenize=True)
        pass

    except Exception as e:
        logging.info("custom Exception")
        raise CustomException(e, sys)