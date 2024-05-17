#header imports
import sys,os
from os.path import dirname,join,abspath

from src.utils import load_obj
sys.path.insert(0,abspath(join(dirname(__file__),'../..')))

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

import pandas as pd

#create Prediction Pipeline config class
@dataclass
class PredictionPipelineConfig():
    
    preprocessor_file_path=os.path.join(os.getcwd(),"artifacts","preprocessor.pkl")
    model_file_path=os.path.join(os.getcwd(),"artifacts","model.pkl")
    

#initiate PredictionPipeline
class PredictionPipeline():
    
    def __init__(self) -> None:
        self.predictionPipeline_config=PredictionPipelineConfig()
    
    def predict(self,features):
        
        # load the pre-processor
        preprocessor=load_obj(self.predictionPipeline_config.preprocessor_file_path)
        logging.info('Preprocessor loaded successfully')
        
        #load the model
        model=load_obj(self.predictionPipeline_config.model_file_path)
        logging.info('Model loaded successfully')
        
        #preprocess the features
        cleaned_features=preprocessor.transform(features)
        logging.info('Features are cleaned')
        #get prediction
        prediction=model.predict(cleaned_features)
        logging.info(f'Prediction is successful')
        
        return prediction
    
    
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataFrame(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('dataFrame Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e,sys)


