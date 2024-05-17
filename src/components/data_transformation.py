#header file
import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'../..')))

#import libraries
import os,sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


#create a Data TransformationConfig Class
@dataclass
class DataTransformationConfig():
    preprocessor_file_path=os.path.join(os.getcwd(),"artifacts","preprocessor.pkl")
    clean_train_file_path=os.path.join(os.getcwd(),"artifacts","clean_train.csv")
    clean_test_file_path=os.path.join(os.getcwd(),"artifacts","clean_test.csv")
    

#create Data Transformation class
class DataTransformation():
    
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
        
    
    def getPreprocessorObject(self):
        
        # separate columns based on data types
        categorical_cols=['cut', 'color', 'clarity']
        numerical_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
        
        # Define the custom ranking for each ordinal variable
        cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
        color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
        clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'] 
        
        ## Numerical Pipeline
        num_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())

            ]

        )

        # Categorical Pipeline
        cat_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('ordinal_encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
            ('scaler',StandardScaler())
            ]

        )

        preprocessor=ColumnTransformer([
        ('num_pipeline',num_pipeline,numerical_cols),
        ('cat_pipeline',cat_pipeline,categorical_cols)
        ])
        logging.info('Preprocessor created successfully!')

        return preprocessor
    
    def initiateDataTransformation(self,train_path,test_path):
        logging.info('Data Transformation has started')
        try:
            
            
            #read test and train data
            train_df=pd.read_csv(train_path)
            logging.info('Train data read successfully')
            
            test_df=pd.read_csv(test_path)
            logging.info('Test data read successfully')
            
            #split dependent and independent features
            X_train,y_train=train_df.drop(['id','price'],axis=1),train_df['price']
            X_test,y_test=test_df.drop(['id','price'],axis=1),test_df['price']
            logging.info('Splitting of Dependent and Independent features is successful')
            
            # get preprocessor and pre-process the content
            preprocessor=self.getPreprocessorObject()
            X_train_arr=preprocessor.fit_transform(X_train)
            logging.info('X_train successfully pre-processed')
            
            X_test_arr=preprocessor.transform(X_test)
            logging.info('X_test successfully pre-processed')
            
            #combine X_train_arr with y_train and vice versa
            clean_train_arr=np.c_[X_train_arr,np.array(y_train)]
            clean_test_arr=np.c_[X_test_arr,np.array(y_test)]
            logging.info('Concatenation of  cleaned arrays is successful')
            
            #save the pre-processor 
            save_obj(self.transformation_config.preprocessor_file_path,preprocessor)
            logging.info('Pre-processor successfully saved')
            
            return(
                clean_train_arr,clean_test_arr
            )
            
            
                    
        except CustomException as e:
            logging.info(f'Exception occurred in Data Transformation,{e}')
            print(e)
