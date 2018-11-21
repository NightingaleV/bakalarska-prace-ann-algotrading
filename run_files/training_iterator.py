# IMPORTS
# ------------------------------------------------------------------------------
import os
import sys
import itertools
import gc
# DataScience
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Custom Packages
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
from dataset_manager.dataset_manager import DatasetManager
from neural_network_lab.model_preset.logger import Logger

# Import Your Model Setup
from neural_network_lab.ann_classification import ModelNeuralNetwork

# INITIALIZE DATASET
# ------------------------------------------------------------------------------
# Data Parameters
currency_pair = 'USD/JPY'
data_postfix = '12-16'
time_frame = '1H'
dm = DatasetManager(currency_pair, data_postfix).resample(time_frame)
dm.df.reset_index(drop=True, inplace=True)
dm.save_df_copy_into_memory()

# ITERATION PARAMETERS
# ------------------------------------------------------------------------------
# Length of predicted Moving Average
moving_average_periods = [15, 20, 30, 40, 60, 80]
# Periods: n_past, n_future
predictions_periods = [
    [10, 3],
    [15, 5],
    [25, 5],
    [50, 10],
]
# Postfix for multiple iterations of one model setup
multiple_iteration = ['iter-1', 'iter-2']

# ITERATOR PARAMETERS
# ------------------------------------------------------------------------------
for moving_average, periods, iteration in itertools.product(moving_average_periods,
                                                            predictions_periods,
                                                            multiple_iteration):

    # IMPORT NEURAL NETWORK
    # --------------------------------------------------------------------------
    model = ModelNeuralNetwork(data_manager=dm)
    model.predict_ma: int = 30
    model.n_past: int = 25
    model.n_future: int = 5
    model.model_task: str = 'classification'
    model.model_postfix: str = ''
    models_evaluations = []
