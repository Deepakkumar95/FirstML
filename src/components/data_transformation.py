import os
import sys
from dataclasses import dataclass

from src.logger import logging 
from src.exception import CustomException
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass()
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join("artifacts", "preprecessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformation()

    def get_data_transformation_object(self):
        try:
            logging.info("data transformation initiated")
            df
        
        except Exception as e:
            logging.info("error occured at data transformation stage")
            raise CustomException(e, sys)

