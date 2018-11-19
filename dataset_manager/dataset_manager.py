import os
import pandas as pd
from .technical_indicators import TechnicalIndicators


class DatasetManager(TechnicalIndicators):
    PARENT_DIRECTORY = os.getcwd()
    PACKAGE_FOLDER = os.path.abspath(os.path.dirname(__file__))
    DATASET_FOLDER = os.path.join(PACKAGE_FOLDER, 'datasets')

    def __init__(self, symbols='USD/JPY', postfix='12-16'):
        TechnicalIndicators.__init__(self)
        self.symbols = symbols.upper()
        self.symbol_arr = self.symbols.lower().split('/')
        self.postfix = postfix
        self.filename = '{}{}_{}.csv'.format(self.symbol_arr[0], self.symbol_arr[1], self.postfix)
        self.file = os.path.join(self.DATASET_FOLDER, self.filename)
        self.df = None
        self.df_copy = None


