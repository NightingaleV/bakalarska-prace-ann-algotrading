# IMPORTS
# ------------------------------------------------------------------------------
import os
import sys
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
df = dm.df

# IMPORT NEURAL NETWORK
# ------------------------------------------------------------------------------
model = ModelNeuralNetwork(data_manager=dm)
model.predict_ma: int = 40
model.n_past: int = 50
model.n_future: int = 10
model.model_task: str = 'classification'
model.model_postfix: str = 'test_spyder'

# Logger
# ------------------------------------------------------------------------------
logger = Logger()
logger.set_model(model)
logger.set_data_manager(dm)

# INDICATORS
# ------------------------------------------------------------------------------
dm.restore_df()
dm.ewma(model.predict_ma)
dm.ewma(16)
dm.ewma(20)
dm.ewma(40)
dm.rsi_indicator(25)
dm.stochastic_oscilator(25, 3, 3)
dm.get_indicators(target=model.model_task)

# Derived Quantities
dm.df['past_price_regression'] = dm.df[dm.mean_indicators[0]] / dm.df[dm.mean_indicators[0]].shift(
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

# Drop unnecessary values
dm.df.drop(['low', 'high', 'open'], axis=1, inplace=True)
dm.df.drop(['%d', 'past_price_regression', 'future_price_regression'], axis=1, inplace=True)
# dm.df.drop(model.mean_indicators, axis=1, inplace=True)
dm.df = dm.df.iloc[30:-5]
dm.df.reset_index(drop=True, inplace=True)
dm.get_indicators(target=model.model_task)
df = dm.df

# Test/Train split
df_train, df_test, df_test_close = dm.test_train_split(model)

# NORMALIZATION
# ------------------------------------------------------------------------------
scaler = StandardScaler()
scaled_df_train = scaler.fit_transform(df_train[dm.mean_indicators + dm.indicators])
scaled_df_test = scaler.transform(df_test[dm.mean_indicators + dm.indicators])

# Create input vectors
x_train, y_train = model.create_train_vectors(df_train, scaled_df_train)
x_test, y_test, y_test_price = model.create_test_vectors(df_test, scaled_df_test, df_test_close)

# TRAIN NETWORK
# ------------------------------------------------------------------------------
trained_model, training_history = model.train_network(x_train, y_train)
# Plot Training Progress of Error
model.plot_training_loss()
# Plot Training Progress of Accuracy
model.plot_training_metric()

del (currency_pair, data_postfix, mean_average, nb_dir, time_frame)

# MAKE PREDICTION
# ------------------------------------------------------------------------------
# Load Best Model
classifier = model.load_network()
# Make Predictions
predictions_train = classifier.predict(x_train)
predictions_test = classifier.predict(x_test)
# Set values for evaluation
actual_train = y_train
actual_test = y_test

# CREATE SETS FOR EVALUATION
# Columns: Actual, Prediction, Close Price
# ------------------------------------------------------------------------------
# TRAIN Evaluation Set
df_train_eval = model.create_train_eval_set(actual_train, predictions_train)
# VALIDATION Evaluation Set
df_val_eval = model.create_val_eval_set(actual_train, predictions_train)
# TEST Evaluation Set
df_test_eval = model.create_test_eval_set(actual_test, predictions_test, y_test_price)

# ACCURACY EVALUATION
# ------------------------------------------------------------------------------
model.test_score = model.calc_acc(df_train_eval.copy(), origin=0.5, actual_col='actual',
                                  prediction_col='prediction')

# TRADING STRATEGIES OPTIMIZATION
# ------------------------------------------------------------------------------
# NN Trading Threshold Optimization on TRAIN Set
strategies = []
for threshold in np.linspace(0, 0.45, 61):
    df_eval = df_train_eval.copy()
    # Calc without drawdown, it is very time consuming
    strategy = model.prediction_strategy(df_eval, origin=0.5, threshold=threshold,
                                         calc_drawdown=False)
    strategies.append(strategy)
df_strategies = pd.DataFrame(data=strategies,
                             columns=['threshold', 'pip_profit', 'sharpe', 'winrate',
                                      'drawdown', 'fees', 'trades_n', ])
# SAVE Strategies into CSV
df_strategies.to_csv(f'{model.models_folder}/{model.model_name}/pred_strategy_optimization.csv',
                     encoding='utf-8',
                     index=False)
# SAVE threshold parameter of the best strategy
model.set_pred_best_threshold(df_strategies)
# PLOT threshold optimization
model.plot_threshold_optimization(df_strategies, plot_name='threshold_nn_pred_optimization')

# MACD Strategy optimization
strategies = []
for threshold in np.linspace(0, 0.45, 61):
    df_eval = df_train_eval.copy()
    # Calc without drawdown, it is very time consuming
    strategy = model.macd_strategy(df_eval, origin=0.5, threshold=threshold, calc_drawdown=False)
    strategies.append(strategy)
df_strategies = pd.DataFrame(data=strategies,
                             columns=['threshold', 'pip_profit', 'sharpe', 'winrate', 'drawdown',
                                      'fees', 'trades_n'])
# SAVE Strategies into CSV
df_strategies.to_csv('trained_models/{}/macd_strategy_optimization.csv'.format(model.model_name),
                     encoding='utf-8',
                     index=False)
# SAVE threshold parameter of the best strategy
model.set_macd_best_threshold(df_strategies)
# PLOT threshold optimization
model.plot_threshold_optimization(df_strategies, 'threshold_macd_optimization')

# TRADING STRATEGIES EVALUATION
# Calculating strategies with best threshold parameters
best_strategies_evaluations = []
# TEST SET EVALUATION
# ------------------------------------------------------------------------------
# PREDICTION STRATEGY
df_eval = df_test_eval.copy()
strategy = model.prediction_strategy(df_eval, origin=0.5,
                                     threshold=model.nn_pred_strategy_best_threshold)
strategy.insert(0, 'test_nn_pred')
# SAVE to list of strategies
best_strategies_evaluations.append(strategy)
# PLOT Returns
model.plot_cumulative_returns(df_eval, 'nn_pred_test_returns')
# SAVE to parameters for Logger
strategy_dict = model.prediction_strategy(df=df_test_eval.copy(), origin=0.5,
                                          threshold=model.nn_pred_strategy_best_threshold,
                                          form='dict')
model.set_nn_pred_strategy_parameters(strategy_dict)
# MACD STRATEGY
df_eval = df_test_eval.copy()
strategy = model.macd_strategy(df_eval, origin=0.5, threshold=model.macd_strategy_best_threshold)
strategy.insert(0, 'test_macd')
# SAVE to list of strategies
best_strategies_evaluations.append(strategy)
# PLOT Returns
model.plot_cumulative_returns(df_eval, 'macd_test_returns')
# SAVE to parameters for Logger
strategy_dict = model.macd_strategy(df=df_test_eval.copy(), origin=0.5,
                                    threshold=model.macd_strategy_best_threshold, form='dict')
model.set_macd_strategy_parameters(strategy_dict)

# TRAIN SET EVALUATION
# ------------------------------------------------------------------------------
# PREDICTION STRATEGY
df_eval = df_train_eval.copy()
strategy = model.prediction_strategy(df_eval, origin=0.5,
                                     threshold=model.nn_pred_strategy_best_threshold)
strategy.insert(0, 'train_nn_pred')
# SAVE to list of strategies
best_strategies_evaluations.append(strategy)
# PLOT Returns
model.plot_cumulative_returns(df_eval, 'nn_pred_train_returns')
# SAVE to parameters for Logger
model.nn_pred_train_pip_return = model.get_cumulative_pip_return(df_eval)
# MACD STRATEGY
df_eval = df_train_eval.copy()
strategy = model.macd_strategy(df_eval, origin=0.5, threshold=model.macd_strategy_best_threshold)
strategy.insert(0, 'train_macd')
# SAVE to list of strategies
best_strategies_evaluations.append(strategy)
# PLOT Returns
model.plot_cumulative_returns(df_eval, 'macd_train_returns')
# SAVE to parameters for Logger
model.macd_strategy_train_pip_return = model.get_cumulative_pip_return(df_eval)

# VALIDATION SET EVALUATION
# ------------------------------------------------------------------------------
# PREDICTION STRATEGY
df_eval = df_val_eval.copy()
strategy = model.prediction_strategy(df_eval, origin=0.5,
                                     threshold=model.nn_pred_strategy_best_threshold)
strategy.insert(0, 'val_nn_pred')
# SAVE to list of strategies
best_strategies_evaluations.append(strategy)
# PLOT Returns
model.plot_cumulative_returns(df_eval, 'nn_pred_val_returns')
# SAVE to parameters for Logger
model.nn_pred_val_pip_return = model.get_cumulative_pip_return(df_eval)
# MACD STRATEGY
df_eval = df_val_eval.copy()
strategy = model.macd_strategy(df_eval, origin=0.5, threshold=model.macd_strategy_best_threshold)
strategy.insert(0, 'val_macd')
# SAVE to list of strategies
best_strategies_evaluations.append(strategy)
# PLOT Returns
model.plot_cumulative_returns(df_eval, 'macd_val_returns')
# SAVE to parameters for Logger
model.macd_strategy_val_pip_return = model.get_cumulative_pip_return(df_eval)

# EXPORT INFORMATION
# ------------------------------------------------------------------------------
# Results of best strategy evaluation
df_strategies_eval = pd.DataFrame(data=best_strategies_evaluations,
                                  columns=['type', 'threshold', 'pip_profit', 'sharpe', 'winrate',
                                           'drawdown', 'fees', 'trades_n'])
# Export Results to CSV
df_strategies_eval.to_csv(
    f'{model.models_folder}/{model.model_name}/best_strategies_evaluation.csv',
    encoding='utf-8', index=False)

# LOG MODEL PARAMETERS
logger.log_model_info()
