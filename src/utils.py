import sys
import os

import pickle

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from src.exception import CustomException
from src.logger import logging


def save_obj(file_path):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)