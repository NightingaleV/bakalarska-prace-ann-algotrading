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

        # Basic Workflow Tasks
        self.init_dataset()
        self.change_index()
        self.remove_nan_values()

    # Import Dataset from CSV
    def init_dataset(self):
        print('Importing Dataset: ' + self.filename)
        self.df = pd.read_csv(self.file, sep=';')

    # From timestamp to datetime
    def change_index(self):
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df.set_index('datetime', inplace=True)

    # Clean Dataset from missing values
    def remove_nan_values(self):
        print('Cleaning dataset')
        while self.df.isnull().any().any():
            self.df.fillna(method='ffill', inplace=True)

    # Aggregate Dataset - 1D, 1H, 5Min etc...
    def resample(self, period):
        print('Aggregating dataset on ' + period + ' candles')
        self.df = self.df.resample(period).agg({'open': 'first',
                                                'high': 'max',
                                                'low': 'min',
                                                'close': 'last'})
        self.df = self.df.dropna()
        return self

    # Make Copy of DataFrame in current state
    def save_df_copy_into_memory(self):
        self.df_copy = self.df.copy()

    # Load Copy of saved DataFrame
    def restore_df(self):
        self.df = self.df_copy.copy()

    # Change borders of dataset
    def restrict(self, from_date, to_date=None):
        from_date = f"{from_date} 00:00:00"
        if to_date is not None:
            to_date = f"{to_date} 00:00:00"
            self.df = self.df.loc[from_date:to_date]
        else:
            self.df = self.df.loc[from_date:]
        return self

    # Push position of columns to right by number
    def reorder(self, positions=1):
        cols = self.df.columns.tolist()
        cols = cols[-positions:] + cols[:-positions]
        self.df = self.df[cols]
        return self.df
