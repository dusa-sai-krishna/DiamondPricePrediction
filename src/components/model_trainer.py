#header file
import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'../..')))

#import libraries
import os,sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj,model_evaluator
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


#Create a ModelTrainerConfig class
@dataclass
class ModelTrainerConfig():
    model_file_path=os.path.join(os.getcwd(),"artifacts","model.pkl")


#create a ModelTransformer class
class ModelTrainer():
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()
        
    
    def initiateModelTrainer(self,clean_train_arr,clean_test_arr):
        logging.info('Initiating Model Trainer')
        
        try:
            
            #split the data
            X_train,y_train,X_test,y_test=(
                clean_train_arr[:,:-1],
                clean_train_arr[:,-1],
                clean_test_arr[:,:-1],
                clean_test_arr[:,-1],
            )
            logging.info('Data successfully splitted')
            
            #specify the models
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elastic_net':ElasticNet()
            }
            
            #get the best model
            model_report=model_evaluator(X_train,y_train,X_test,y_test,models)
            
            sorted_model_report=sorted(list(model_report.items()),key=lambda x:x[1]) # sorted based on scores
            
            logging.info(f'Sorted model report is {sorted_model_report}')
            best_model,best_score=sorted_model_report[-1]
            logging.info(f'Best model is {best_model} and its score is {best_score}')
            
            #save the model
            save_obj(self.trainer_config.model_file_path,best_model)
            
            logging.info('Model has been saved successfully')
        
        except CustomException as e:
            logging.info(f'Error occurred during training the model',e)
            print(e)
        