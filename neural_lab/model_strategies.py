import numpy as np
from typing import Dict


class ModelStrategies:
    SPREAD = 1.5

    def __init__(self):
        self.pip: float = 0.
        self.set_pip_size()
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

    # Set Threshold of Best NN Pred Strategy
    def set_pred_best_threshold(self, dataset):
        best_row = dataset['pip_profit'].idxmax()
        self.nn_pred_strategy_best_threshold = dataset['threshold'].iloc[best_row]

    # Set Threshold of Best NN Pred Strategy
    def set_macd_best_threshold(self, dataset):
        best_row = dataset['pip_profit'].idxmax()
        self.macd_strategy_best_threshold = dataset['threshold'].iloc[best_row]

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

    # TRADING STRATEGIES
    # ------------------------------------------------------------------------------
    # Prediction Threshold Strategy
    def prediction_strategy(self, df, origin, threshold, calc_drawdown=True, form='list'):
        # Threshold
        pred_treshold = round(threshold, 6)
        # Set trading orders
        self.set_trade_positions(df, origin=origin, threshold=pred_treshold)
        # Calc Returns / Fees
        cum_pip_ret, cum_pip_fees = self.calc_pips_return(df)
        # Calc Sharpe
        sharpe = self.calc_sharpe_ratio(df, 'pip_ret')
        # Calc Win rate
        winrate = self.calc_win_rate(df, 'cum_pip_ret')
        # Calc Drawdown
        max_drawdown = 0
        if calc_drawdown:
            max_drawdown, max_drawdown_pct, duration = self.calc_drawdown(df, 'cum_pip_ret')
        # Number of Trades
        n_trades = int(cum_pip_fees / self.SPREAD)
        # Return List or Dictionary
        strategy = None
        if form == 'list':
            strategy = [pred_treshold, cum_pip_ret, sharpe, winrate, max_drawdown, cum_pip_fees,
                        n_trades]

        elif form == 'dict':
            strategy = {'threshold': pred_treshold,
                        'pip_profit': cum_pip_ret,
                        'sharpe': sharpe,
                        'winrate': winrate,
                        'max_drawdown': max_drawdown,
                        'pip_fees': cum_pip_fees,
                        'n_trades': n_trades}
        return strategy

    @staticmethod
    def set_trade_positions(dataset, origin=0, threshold=0):
        dataset['long'] = np.where(dataset['prediction'] > origin + threshold, 1, 0)
        dataset['short'] = np.where(dataset['prediction'] < origin - threshold, 1, 0)
        return dataset

    # Calculate return in pip for dataset - Needed close/long/short
    def calc_pips_return(self, dataset):
        pip = self.pip

        # Fees for every trade executed
        dataset['trans_cost'] = np.where(
            (((dataset['long'] == True) & (dataset['long'].shift(1) == False)) |
             ((dataset['short'] == True) & (dataset['short'].shift(1) == False))), self.SPREAD,
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
