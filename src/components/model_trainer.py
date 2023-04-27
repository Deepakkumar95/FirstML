import numpy
import pandas
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from src.utils import evaluate_model
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variable from train and test")
            X_train, y_train, X_test, y_test=(

                    train_array[:,:-1],
                    train_array[:,:-1],
                    test_array[:,:-1],
                    test_array[:,:-1]
            )
            ## Train multiple models

            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet()
            }

            models_report:dict= evaluate_model(X_train, y_train, X_test, y_test, models)
            print(models_report)
            print("\n========================================================================\n")
            logging.info(f"Model Report: {models_report}")

            #to get best model score from dictionary
            best_model_score= max(sorted(models_report.values()))

            #to get best model name from dictionary
            best_model_name= list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            best_model= models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            
            save_obj(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

        except Exception as e:
            logging.info("Error occured at Model Training")
            raise CustomException(e, sys)
