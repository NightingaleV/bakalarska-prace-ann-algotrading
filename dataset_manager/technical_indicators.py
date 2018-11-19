# Imports
import numpy as np
import pandas as pd


class TechnicalIndicators:
    cci_constant = 0.015

    def __init__(self):
        self.df = None

    # Exponentially-weighted moving average
    def ewma(self, periods):
        indicator = 'EWMA{}'.format(periods)
        self.df[indicator] = self.df['close'].ewm(span=periods).mean()
        return self

    # Stochastic Oscillator
    def stochastic_oscilator(self, k_period, d_period, smooth=1):
        lows = 'l{}'.format(k_period)
        highs = 'h{}'.format(k_period)
        self.df = self.calc_roll_min(self.df, k_period)
        self.df = self.calc_roll_max(self.df, k_period)

        self.df = self.stok(self.df, k_period)
        if smooth >= 1:
            self.df = self.smooth_stok(self.df, smooth)
        self.df = self.stod(self.df, d_period)
        self.df.drop([lows, highs], axis=1, inplace=True)
        return self

    @staticmethod
    def calc_roll_min(dataset, k_period):
        lows = 'l{}'.format(k_period)
        dataset[lows] = dataset['low'].rolling(window=k_period).min()
        return dataset

    @staticmethod
    def calc_roll_max(dataset, k_period):
        highs = 'h{}'.format(k_period)
        dataset[highs] = dataset['high'].rolling(window=k_period).max()
        return dataset

    @staticmethod
    def stok(dataset, k_period):
        lows = 'l{}'.format(k_period)
        highs = 'h{}'.format(k_period)
        dataset['%k'] = ((dataset['close'] - dataset[lows]) / (
                dataset[highs] - dataset[lows])) * 100
        return dataset

    @staticmethod
    def smooth_stok(dataset, smooth):
        dataset['%k'] = dataset['%k'].rolling(window=smooth).mean()
        return dataset

    @staticmethod
    def stod(dataset, d_period):
        dataset['%d'] = dataset['%k'].rolling(window=d_period).mean()
        return dataset
