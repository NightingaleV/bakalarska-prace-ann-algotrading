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


