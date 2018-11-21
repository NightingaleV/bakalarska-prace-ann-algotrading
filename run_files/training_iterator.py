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
multiple_iterations = ['iter-1', 'iter-2']

# ITERATOR
# ------------------------------------------------------------------------------
# For exporting Results
models_evaluations = []
for moving_average, periods, iteration_postfix in itertools.product(moving_average_periods,
                                                                    predictions_periods,
                                                                    multiple_iterations):
    # IMPORT NEURAL NETWORK
    # --------------------------------------------------------------------------
    model = ModelNeuralNetwork(data_manager=dm)
    model.predict_ma: int = moving_average
    model.n_past: int = periods[0]
    model.n_future: int = periods[1]
    model.model_task: str = 'classification'
    model.model_postfix: str = iteration_postfix

    # INDICATORS
    # --------------------------------------------------------------------------
    dm.restore_df()
    dm.ewma(model.predict_ma)
    if moving_average != 15:
        dm.ewma(15)
    if moving_average != 20:
        dm.ewma(20)
    if moving_average != 30:
        dm.ewma(30)
    if moving_average != 40:
        dm.ewma(40)
    if moving_average != 60:
        dm.ewma(60)
    dm.rsi_indicator(25)
    dm.stochastic_oscilator(25, 3, 3)
    dm.get_indicators(target=model.model_task)

    # Derived Quantities
    dm.df['past_price_regression'] = dm.df[dm.mean_indicators[0]] / dm.df[
        dm.mean_indicators[0]].shift(
        model.n_future)
    dm.df['past_log_regression'] = np.log(dm.df['past_price_regression'])
    for mean_average in dm.mean_indicators:
        dm.df['mean_diff_{}'.format(mean_average[-2:])] = dm.df[mean_average] - dm.df[
            mean_average].shift(1)
        dm.df['mean_ret_{}'.format(mean_average[-2:])] = np.log(
            dm.df[mean_average] / dm.df[mean_average].shift(1))

    # CLASSIFICATION VALUES
    dm.df['future_price_regression'] = dm.df[dm.mean_indicators[0]].shift(-model.n_future) / dm.df[
        dm.mean_indicators[0]]
    dm.df[model.model_task] = np.where(dm.df['future_price_regression'] > 1, 1, 0)

    # For training without EMAs
    # dm.df.drop(dm.mean_indicators, axis=1, inplace=True)
    # Drop unnecessary values
    dm.df.drop(['low', 'high', 'open'], axis=1, inplace=True)
    dm.df.drop(['%d', 'past_price_regression', 'future_price_regression'], axis=1, inplace=True)
    dm.df = dm.df.iloc[30:-5]
    dm.df.reset_index(drop=True, inplace=True)
    dm.get_indicators(target=model.model_task)

    # SPLIT Train/Test/Test(close price)
    df_train, df_test, df_test_close = dm.test_train_split(model)

    # NORMALIZATION
    # --------------------------------------------------------------------------
    scaler = StandardScaler()
    scaled_df_train = scaler.fit_transform(df_train[dm.mean_indicators + dm.indicators])
    scaled_df_test = scaler.transform(df_test[dm.mean_indicators + dm.indicators])

    # INPUT VECTORS
    # --------------------------------------------------------------------------
    x_train, y_train = model.create_train_vectors(df_train, scaled_df_train)
    x_test, y_test, y_test_price = model.create_test_vectors(df_test, scaled_df_test, df_test_close)

    
