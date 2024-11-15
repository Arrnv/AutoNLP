from src.AutoNLP.logging import logging
from src.AutoNLP.exception import CustomException
import sys
from src.AutoNLP.components.Preprocessing import Preprocessor
import pandas as pd
from src.AutoNLP.components.eda import EDA

df = pd.read_csv("iphone.csv")
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
            return data
            
            
      
             
if __name__ == "__main__":
    # Taking input csv
    try:
       data = run_pipeline(df,text_col='reviewDescription',clean_text=True, tokenize=True)
       EDA.Word_Cloud(df, 'processed_text')
       EDA.top_10_Tokens(df,'tockenized_data')
       EDA.Pos_wordcloud(df, 'tockenized_data')
       EDA.neg_wordcloud(df,'tockenized_data' )

    except Exception as e:
        logging.info("custom Exception")
        raise CustomException(e, sys)