# Diamond Price Prediction

## Overview

This project aims to predict the price of diamonds based on various features using a regression model. The project is implemented in Python 3.7 and employs a modular code structure with pipelines to automate the workflow. Additionally, a Streamlit front end is provided for an interactive user experience.

## Dataset

The dataset used for this project is sourced from the Kaggle Playground Series - Season 3, Episode 8 competition. The goal is to predict the price of a given diamond based on the following features:

- `id`: Unique identifier of each diamond
- `carat`: Carat (ct.) refers to the unique unit of weight measurement used exclusively to weigh gemstones and diamonds.
- `cut`: Quality of Diamond Cut
- `color`: Color of Diamond
- `clarity`: Diamond clarity is a measure of the purity and rarity of the stone, graded by the visibility of these characteristics under 10-power magnification.
- `depth`: The depth of the diamond is its height (in millimeters) measured from the culet (bottom tip) to the table (flat, top surface).
- `table`: A diamond's table is the facet which can be seen when the stone is viewed face up.
- `x`: Diamond X dimension
- `y`: Diamond Y dimension
- `z`: Diamond Z dimension

**Target variable**:
- `price`: Price of the given diamond.

[Dataset Source Link](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

## Project Structure

The project follows a modular structure to ensure scalability and maintainability. Below is an overview of the project structure:

```
├── src
│ ├── components
│ │ ├── data_ingestion.py
│ │ ├── data_preprocessing.py
│ │ ├── model_training.py
│ ├── pipelines
│ │ ├── prediction_pipeline.py
│ │ ├── training_pipeline.py
│ ├── logger.py
│ ├── exception.py
│ └── utils.py
├── notebooks
│ ├── EDA.ipynb
│ ├── Model_Training.ipynb
├── artifacts
│ ├── models
│ ├── preprocessors
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Key Components

### Data Ingestion

The `data_ingestion.py` module is responsible for loading the dataset and performing initial preprocessing steps such as handling missing values and splitting the data into training and testing sets.

### Data Preprocessing

The `data_preprocessing.py` module includes functions to clean and transform the data. It utilizes pipelines to automate the preprocessing steps, including scaling and encoding.

### Model Training

The `model_training.py` module handles the training of regression models to predict diamond prices. It includes and model selection using r2 score.

### Pipelines

The project uses pipelines to streamline data processing and model training:

- `prediction_pipeline.py`: Automates the data preprocessing steps.
- `training_pipeline.py`: Automates the data ingestion, data pre-processing, model training and evaluation process.

### Streamlit App

The `app.py` script creates an interactive front end using Streamlit, allowing users to input diamond features and get price predictions.

## How to Run

### Prerequisites

Ensure you have Python 3.7 installed. Install the required packages using the following command:

`pip install -r requirements.txt`


Run the training_pipeline
`python src/pipelines/training_pipeline.py`

Run the front end
`streamlit run web_app/app.py`
