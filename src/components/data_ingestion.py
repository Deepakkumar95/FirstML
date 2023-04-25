import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation



#Initialize the Data Ingestion Configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str= os.path.join("artifacts", "train.csv" )
    test_data_path:str= os.path.join("artifacts", "test.csv" )
    raw_data_path:str= os.path.join("artifacts", "raw.csv" )

##Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion method starts")
        try:
            df= pd.read_csv(os.path.join("src/notebook/data", "gemstone.csv"))
            logging.info("dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train Test Split")
            train_set, test_set= train_test_split(df, test_size=0.3, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header= True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header= True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )



        
        except Exception as e:
            logging.info("Error occured at Data Ingestion Stage")
            raise CustomException(e, sys)


#Run Data Ingestion
if __name__=="__main__":
    obj= DataIngestion()
    train_data_path, test_data_path= obj.initiate_data_ingestion()
    data_transformation= DataTransformation()
    train_arr, test_arr, _= data_transformation.initiate_data_transformation(train_data_path, test_data_path)

