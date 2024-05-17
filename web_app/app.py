import sys,os
from os.path import dirname,join,abspath
sys.path.insert(0,abspath(join(dirname(__file__),'..')))



#import libraries
import streamlit as st
from src.pipelines.prediction_pipeline import PredictionPipeline,CustomData


import streamlit as st
import numpy as np

#setting page config
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon='💎'
)

#title
st.title("Diamond Price Prediction")

start=st.button("Let's go")
#Initializing session state
if 'start_state' not in st.session_state:
    st.session_state['start_state'] = False

if (start or st.session_state['start_state']):
    
    st.session_state.start_state=True
    
    #Take input
    carat=(st.text_input("Carat",key="carat"))
    depth=(st.text_input("Depth",key="depth"))
    table=(st.text_input("Table",key="table"))
    x=(st.text_input("x",key="x"))
    y=(st.text_input("y",key="y"))
    z=(st.text_input("z",key="z"))
    cut=st.selectbox('Select the cut type',['Fair', 'Good', 'Very Good','Premium','Ideal'])
    color=st.selectbox('Select the color',['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity=st.selectbox('Select the clarity',['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])
    
    submit=st.button('Predict')
    if submit:
        #Create data frame of the feature input
        
        obj= CustomData(float(carat),float(depth),float(table),float(x),float(y),float(z),cut,color,clarity)
        df=obj.get_data_as_dataFrame()
        
        #get prediction
        prediction_agent=PredictionPipeline()
        prediction=prediction_agent.predict(df)
        
        st.header(f'Price of the diamond is {prediction[0]}')