import os
import pandas as pd
from .technical_indicators import TechnicalIndicators


class DatasetManager(TechnicalIndicators):
    PARENT_DIRECTORY = os.getcwd()
    PACKAGE_FOLDER = os.path.abspath(os.path.dirname(__file__))
    DATASET_FOLDER = os.path.join(PACKAGE_FOLDER, 'datasets')

    def __init__(self, symbols='USD/JPY', timeframe=None, postfix='12-16'):
        TechnicalIndicators.__init__(self)
        self.symbol = symbols.upper()
        self.symbol_arr = self.symbol.lower().split('/')
        self.symbol_slug = self.symbol_arr[0] + self.symbol_arr[1]
        self.postfix = postfix
        self.timeframe = timeframe
        self.filename = f'{self.symbol_arr[0]}{self.symbol_arr[1]}_{self.postfix}.csv'
        self.file = os.path.join(self.DATASET_FOLDER, self.filename)
        self.df = None
        self.df_copy = None

        # Basic Workflow Tasks
        self.init_dataset()
        try:
            self.change_index()
            self.remove_nan_values()
        except (KeyError, TypeError):
            print('Unable to initialize dataset')
            raise

    # Import Dataset from CSV
    def init_dataset(self):
        print('Import dataset: ' + self.filename)
        try:
            self.df = pd.read_csv(self.file)
        except FileNotFoundError:
            print('Unable to import ' + self.filename)
            raise
        # TODO
        # If file has different separator
        # self.df = pd.read_csv(self.file, sep=';')

    # From timestamp to datetime
    def change_index(self):
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df.set_index('datetime', inplace=True)

    # Clean Dataset from missing values
    def remove_nan_values(self):
        print('Clean data')
        while self.df.isnull().any().any():
            self.df.fillna(method='ffill', inplace=True)

    # Aggregate Dataset - 1D, 1H, 5Min etc...
    def resample(self, period):
        print('Aggregate data on ' + period + ' candles')
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

    def test_train_split(self, model):
        self.calc_set_size(model=model)
        df = self.df.copy()
        df_train, df_test = df[0:model.train_rows].copy(), df[model.train_rows - model.n_past:len(
            df)].copy()
        df_test_close_price = df_test['close']
        df_train.drop(['close'], axis=1, inplace=True)
        df_test.drop(['close'], axis=1, inplace=True)
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        df_test_close_price.reset_index(drop=True, inplace=True)
        return df_train, df_test, df_test_close_price

    def calc_set_size(self, model):
        model.train_rows = int((1 - model.test_size) * len(self.df))
        model.test_rows = int(model.test_size * len(self.df))
        model.validation_rows = int((1 - model.test_size) * model.val_size * len(self.df))
