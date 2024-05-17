import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'../..')))

#import libraries
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion


if __name__=='__main__':
    ingestor=DataIngestion()
    train_path,test_path=ingestor.initiate_data_ingestion()
    print(train_path,test_path)