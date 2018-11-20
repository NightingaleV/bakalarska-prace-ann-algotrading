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

