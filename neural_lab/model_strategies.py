import numpy as np
from typing import Dict


class ModelStrategies:
    spread = 1.5

    def __init__(self):
        self.pip = None

        # PREDICTION STRATEGY
        # Optimized Treshold
        self.nn_pred_strategy_best_threshold = 0
        # Strategy parameters on Test Set
        self.nn_pred_strategy_pip_return = 0
        self.nn_pred_strategy_fees = 0
        self.nn_pred_strategy_sharpe = 0
        self.nn_pred_strategy_max_drawdown = 0
        self.nn_pred_strategy_win_pct = 0
        # on Train Set
        self.nn_pred_train_pip_return = 0
        # on Validation Set
        self.nn_pred_val_pip_return = 0

        # MACD STRATEGY
        # Optimized Treshold
        self.macd_strategy_best_threshold = 0
        # Strategy parameters on Test Set
        self.macd_strategy_pip_return = 0
        self.macd_strategy_fees = 0
        self.macd_strategy_sharpe = 0
        self.macd_strategy_max_drawdown = 0
        self.macd_strategy_win_pct = 0
        # on Train Set
        self.macd_strategy_train_pip_return = 0
        # on Validation Set
        self.macd_strategy_val_pip_return = 0

    # Set parameters from dictionary
    def set_nn_pred_strategy_parameters(self, strategy: Dict):
        self.nn_pred_strategy_pip_return = strategy['pip_profit']
        self.nn_pred_strategy_fees = strategy['pip_fees']
        self.nn_pred_strategy_sharpe = strategy['sharpe']
        self.nn_pred_strategy_max_drawdown = strategy['max_drawdown']
        self.nn_pred_strategy_win_pct = strategy['winrate']

    # Set parameters from dictionary
    def set_macd_strategy_parameters(self, strategy: Dict):
        self.macd_strategy_pip_return = strategy['pip_profit']
        self.macd_strategy_fees = strategy['pip_fees']
        self.macd_strategy_sharpe = strategy['sharpe']
        self.macd_strategy_max_drawdown = strategy['max_drawdown']
        self.macd_strategy_win_pct = strategy['winrate']

    def set_pip_size(self):
        if self.dataset == 'USD/JPY':
            self.pip = 0.01
        else:
            self.pip = 0.0001

    # Calculate return in pip for dataset - Needed close/long/short
    def calc_pips_return(self, dataset):
        pip = self.pip

        # Fees for every trade executed
        dataset['trans_cost'] = np.where(
            (((dataset['long'] == True) & (dataset['long'].shift(1) == False)) |
             ((dataset['short'] == True) & (dataset['short'].shift(1) == False))), self.spread,
            0)

        # Calculate change in pips
        dataset['pips'] = (dataset['close'] - dataset['close'].shift(1)) * (1 / pip)

        # Calculating returns
        dataset['pip_ret'] = (((dataset['pips'] - dataset['trans_cost']) * dataset['long']) + (
                (-(dataset['pips'] + dataset['trans_cost'])) * dataset['short']))

        # Total Cumulative Return
        dataset['cum_pip_ret'] = (dataset['pip_ret']).cumsum()
        cum_pip_return = self.get_cumulative_pip_return(dataset)

        # Total Fees
        dataset['fees'] = dataset['trans_cost'].cumsum()
        cum_pip_fees = dataset.iloc[-1][-1]

        return cum_pip_return, cum_pip_fees

    @staticmethod
    def get_cumulative_pip_return(dataset):
        return round(dataset['cum_pip_ret'].iloc[-1], 2)
