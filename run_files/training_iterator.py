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

# LOGGER
# ------------------------------------------------------------------------------
logger = Logger()
logger.set_data_manager(dm)

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
    # SETUP MODEL
    # --------------------------------------------------------------------------
    model = ModelNeuralNetwork(data_manager=dm)
    model.predict_ma: int = moving_average
    model.n_past: int = periods[0]
    model.n_future: int = periods[1]
    model.model_task: str = 'classification'
    model.model_postfix: str = iteration_postfix
    
    model.set_model_name()
    logger.set_model(model)

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

    # TRAIN NEURAL NETWORK
    # --------------------------------------------------------------------------
    trained_model, training_history = model.train_network(x_train, y_train)
    # Plot Training Progress of Error
    model.plot_training_loss(show=False)
    # Plot Training Progress of Accuracy
    model.plot_training_metric(show=False)

    # MAKE PREDICTION
    # --------------------------------------------------------------------------
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
    # --------------------------------------------------------------------------
    # TRAIN Evaluation Set
    df_train_eval = model.create_train_eval_set(actual_train, predictions_train)
    # VALIDATION Evaluation Set
    df_val_eval = model.create_val_eval_set(actual_train, predictions_train)
    # TEST Evaluation Set
    df_test_eval = model.create_test_eval_set(actual_test, predictions_test, y_test_price)

    # ACCURACY EVALUATION
    # --------------------------------------------------------------------------
    model.test_score = model.calc_acc(df_train_eval.copy(), origin=0.5, actual_col='actual',
                                      prediction_col='prediction')

    # TRADING STRATEGIES OPTIMIZATION
    # --------------------------------------------------------------------------
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
    model.plot_threshold_optimization(df_strategies, plot_name='threshold_nn_pred_optimization',
                                      show=False)
    # MACD Strategy optimization
    strategies = []
    for threshold in np.linspace(0, 0.45, 61):
        df_eval = df_train_eval.copy()
        # Calc without drawdown, it is very time consuming
        strategy = model.macd_strategy(df_eval, origin=0.5, threshold=threshold,
                                       calc_drawdown=False)
        strategies.append(strategy)
    df_strategies = pd.DataFrame(data=strategies,
                                 columns=['threshold', 'pip_profit', 'sharpe', 'winrate',
                                          'drawdown',
                                          'fees', 'trades_n'])
    # SAVE Strategies into CSV
    df_strategies.to_csv(
        f'{model.models_folder}/{model.model_name}/macd_strategy_optimization.csv',
        encoding='utf-8',
        index=False)
    # SAVE threshold parameter of the best strategy
    model.set_macd_best_threshold(df_strategies)
    # PLOT threshold optimization
    model.plot_threshold_optimization(df_strategies, 'threshold_macd_optimization', show=False)

    # TRADING STRATEGIES EVALUATION
    # --------------------------------------------------------------------------
    # Calculating strategies with best threshold parameters
    best_strategies_evaluations = []
    # TEST SET EVALUATION
    # --------------------------------------------------------------------------
    # PREDICTION STRATEGY
    # -------------------
    df_eval = df_test_eval.copy()
    strategy = model.prediction_strategy(df_eval, origin=0.5,
                                         threshold=model.nn_pred_strategy_best_threshold)
    strategy.insert(0, 'test_nn_pred')
    # SAVE to list of strategies
    best_strategies_evaluations.append(strategy)
    # PLOT Returns
    model.plot_cumulative_returns(df_eval, 'nn_pred_test_returns', show=False)
    # SAVE to parameters for Logger
    strategy_dict = model.prediction_strategy(df=df_test_eval.copy(), origin=0.5,
                                              threshold=model.nn_pred_strategy_best_threshold,
                                              form='dict')
    model.set_nn_pred_strategy_parameters(strategy_dict)

    # MACD STRATEGY
    # -------------------
    df_eval = df_test_eval.copy()
    strategy = model.macd_strategy(df_eval, origin=0.5,
                                   threshold=model.macd_strategy_best_threshold)
    strategy.insert(0, 'test_macd')
    # SAVE to list of strategies
    best_strategies_evaluations.append(strategy)
    # PLOT Returns
    model.plot_cumulative_returns(df_eval, 'macd_test_returns', show=False)
    # SAVE to parameters for Logger
    strategy_dict = model.macd_strategy(df=df_test_eval.copy(), origin=0.5,
                                        threshold=model.macd_strategy_best_threshold, form='dict')
    model.set_macd_strategy_parameters(strategy_dict)

    # TRAIN SET EVALUATION
    # ------------------------------------------------------------------------------
    # PREDICTION STRATEGY
    # -------------------
    df_eval = df_train_eval.copy()
    strategy = model.prediction_strategy(df_eval, origin=0.5,
                                         threshold=model.nn_pred_strategy_best_threshold)
    strategy.insert(0, 'train_nn_pred')
    # SAVE to list of strategies
    best_strategies_evaluations.append(strategy)
    # PLOT Returns
    model.plot_cumulative_returns(df_eval, 'nn_pred_train_returns', show=False)
    # SAVE to parameters for Logger
    model.nn_pred_train_pip_return = model.get_cumulative_pip_return(df_eval)

    # MACD STRATEGY
    # -------------------
    df_eval = df_train_eval.copy()
    strategy = model.macd_strategy(df_eval, origin=0.5,
                                   threshold=model.macd_strategy_best_threshold)
    strategy.insert(0, 'train_macd')
    # SAVE to list of strategies
    best_strategies_evaluations.append(strategy)
    # PLOT Returns
    model.plot_cumulative_returns(df_eval, 'macd_train_returns', show=False)
    # SAVE to parameters for Logger
    model.macd_strategy_train_pip_return = model.get_cumulative_pip_return(df_eval)

    # VALIDATION SET EVALUATION
    # ------------------------------------------------------------------------------
    # PREDICTION STRATEGY
    # -------------------
    df_eval = df_val_eval.copy()
    strategy = model.prediction_strategy(df_eval, origin=0.5,
                                         threshold=model.nn_pred_strategy_best_threshold)
    strategy.insert(0, 'val_nn_pred')
    # SAVE to list of strategies
    best_strategies_evaluations.append(strategy)
    # PLOT Returns
    model.plot_cumulative_returns(df_eval, 'nn_pred_val_returns', show=False)
    # SAVE to parameters for Logger
    model.nn_pred_val_pip_return = model.get_cumulative_pip_return(df_eval)

    # MACD STRATEGY
    # -------------------
    df_eval = df_val_eval.copy()
    strategy = model.macd_strategy(df_eval, origin=0.5,
                                   threshold=model.macd_strategy_best_threshold)
    strategy.insert(0, 'val_macd')
    # SAVE to list of strategies
    best_strategies_evaluations.append(strategy)
    # PLOT Returns
    model.plot_cumulative_returns(df_eval, 'macd_val_returns', show=False)
    # SAVE to parameters for Logger
    model.macd_strategy_val_pip_return = model.get_cumulative_pip_return(df_eval)

    # EXPORT MODEL INFORMATION
    # ------------------------------------------------------------------------------
    # Results of best strategy evaluation
    df_strategies_eval = pd.DataFrame(data=best_strategies_evaluations,
                                      columns=['type', 'threshold', 'pip_profit', 'sharpe',
                                               'winrate',
                                               'drawdown', 'fees', 'trades_n'])
    # Export Results to CSV
    df_strategies_eval.to_csv(
        f'{model.models_folder}/{model.model_name}/best_strategies_evaluation.csv',
        encoding='utf-8', index=False)

    # Log Model Parameters
    logger.log_model_info()

    # SAVE ITERATION INFORMATION
    # ------------------------------------------------------------------------------
    iteration_variables = [moving_average, model.n_past, model.n_future]
    model_score = [model.train_score, model.val_score, model.test_score]
    model_prediction_strategy = [model.nn_pred_strategy_best_threshold,
                                 model.nn_pred_strategy_pip_return,
                                 model.nn_pred_strategy_fees,
                                 model.nn_pred_strategy_sharpe,
                                 model.nn_pred_strategy_max_drawdown,
                                 model.nn_pred_strategy_win_pct,
                                 model.nn_pred_train_pip_return,
                                 model.nn_pred_val_pip_return]
    model_macd_strategy = [model.macd_strategy_best_threshold,
                           model.macd_strategy_pip_return,
                           model.macd_strategy_fees,
                           model.macd_strategy_sharpe,
                           model.macd_strategy_max_drawdown,
                           model.macd_strategy_win_pct,
                           model.macd_strategy_train_pip_return,
                           model.macd_strategy_val_pip_return]

    model_info = iteration_variables + model_score + model_prediction_strategy + model_macd_strategy
    models_evaluations.append(model_info)
    # Del Main Variables
    del (model, classifier, df_eval, best_strategies_evaluations)

    # END OF CYCLE
    # ------------------------------------------------------------------------------

# EXPORT RESULTS OF ITERATIONS
# ------------------------------------------------------------------------------
iteration_variables_labels = ['MA', 'n_past', 'n_future']
model_score_labels = ['Train Acc', 'Val Acc', 'Test Acc']
model_prediction_strategy_labels = ['Pred Threshold',
                                    'Pred Pip Return',
                                    'Pred Fees',
                                    'Pred Sharpe',
                                    'Pred Drawdown',
                                    'Pred Winrate',
                                    'Pred Train Ret',
                                    'Pred Val Ret', ]
model_macd_strategy_labels = ['MACD Threshold',
                              'MACD Pip Return',
                              'MACD Fees',
                              'MACD Sharpe',
                              'MACD Drawdown',
                              'MACD Winrate',
                              'MACD Train Ret',
                              'MACD Val Ret',
                              ]

labels = iteration_variables_labels + model_score_labels + model_prediction_strategy_labels \
         + model_macd_strategy_labels

df_iterations = pd.DataFrame(data=models_evaluations, columns=labels)
df_iterations.to_csv(f'{ModelNeuralNetwork.models_folder}/models_evaluations.csv',
                     encoding='utf-8', index=False)
