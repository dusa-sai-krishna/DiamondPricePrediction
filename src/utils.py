#header file
import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'..')))

from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

import pickle as pkl

def save_obj(file_path,obj):
    logging.info(f'initiating to save the file at {file_path}')
    
    try:
        
        #get the directory name
        dirname=os.path.dirname(file_path)
        logging.info(f'Obtained directory name to save the object,{dirname}')
        
        
        #create directory
        os.makedirs(dirname,exist_ok=True)
        logging.info('successfully created the directory')        
        #save the model
        
        with open(file_path,'wb') as f:
            pkl.dump(obj,f)
        f.close()
        
        logging.info('Successfully converted object to a pkl file')
        
    except CustomException as e:
        logging.info(f'Error while saving a obj, {e}')
        print(e)
        
        
def load_obj(file_path):
    logging.info('Process of loading object started')
    
    try:
        with open(file_path,'rb') as f:
            obj=pkl.load(f)
        f.close()
        logging.info(f'Object at {file_path} loaded successfully')
        return obj
        
    except CustomException as e:
        logging.info(f'Error occurred while loading object,{e}')
        print(e)






def model_evaluator(X_train,y_train,X_test,y_test,models):
    
    logging.info('Model evaluation has started')
    
    try:
        model_report={}
        #iterate through models
        for name,model in models.items():
            
            #fit the model
            model.fit(X_train,y_train)
            
            #get predictions
            y_pred=model.predict(X_test)
            
            #get r2_score
            score=r2_score(y_test,y_pred)
            
            #update in model report
            model_report[model]=score

            logging.info(f'Successfully evaluated {name} model')
        
        logging.info('Model evaluation completed')
        return model_report
    except CustomException as e:
        logging.info(f'Error occurred during model evaluation {e}')
        print(e)