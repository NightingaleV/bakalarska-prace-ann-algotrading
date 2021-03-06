# Data Science
import numpy as np
import pandas as pd
# Custom
from .model_builder import ModelBuilder


class ModelEvaluation(ModelBuilder):

    def __init__(self, data_manager):
        super(ModelEvaluation, self).__init__()
        print('Initialize Model Evaluation')
        # Calculated by DataManager in test_train_split
        self.data_manager = data_manager

        # Accuracy
        self.test_score = 0

        # RootMeanSquaredError
        self.rmse_train = 0
        self.rmse_test = 0

    # CREATE SETs FOR EVALUATION
    # ------------------------------------------------------------------------------
    @staticmethod
    def create_test_eval_set(np_actual_test, np_prediction_test=None, np_close=None):
        # TEST Dataframe for Evaluation
        # Columns: Actual, Prediction, Close Price
        if np_prediction_test is None:
            df_test_eval = pd.DataFrame(
                data=np.hstack((np_actual_test, np_close)),
                columns=['actual', 'close'])
        else:
            df_test_eval = pd.DataFrame(
                data=np.hstack((np_actual_test, np_prediction_test, np_close)),
                columns=['actual', 'prediction', 'close'])

        df_test_eval.reset_index(drop=True, inplace=True)
        return df_test_eval

    def create_train_eval_set(self, np_actual_train, np_predictions_train):
        # TRAIN Dataframe for Evaluation
        # Columns: Actual, Prediction, Close Price
        if np_predictions_train is None:
            df_train_eval = pd.DataFrame(data=np.hstack(np_actual_train),
                                         columns=['actual'])
        else:
            df_train_eval = pd.DataFrame(data=np.hstack((np_actual_train, np_predictions_train)),
                                         columns=['actual', 'prediction'])

        # Add close prices
        df_train_eval['close'] = self.data_manager.df['close'][self.n_past:].reset_index(drop=True)
        # Slice DataFrame - Only Train set without validation part
        index_start = round(len(df_train_eval) * 0)
        index_end = round((len(df_train_eval)) * (1 - self.val_size))
        df_train_eval = df_train_eval[index_start:index_end]
        df_train_eval.reset_index(drop=True, inplace=True)
        return df_train_eval

    def create_val_eval_set(self, np_actual_train, np_predictions_train=None):
        # VALIDATION Dataframe for Evaluation
        # Columns: Actual, Prediction, Close Price
        if np_predictions_train is None:
            df_val_eval = pd.DataFrame(data=np.hstack(np_actual_train),
                                       columns=['actual'])
        else:
            df_val_eval = pd.DataFrame(data=np.hstack((np_actual_train, np_predictions_train)),
                                       columns=['actual', 'prediction'])

        df_val_eval['close'] = self.data_manager.df['close'][self.n_past:].reset_index(drop=True)
        # Slice DataFrame - Only Validation set without train part
        index_start = round(len(df_val_eval) * (1 - self.val_size))
        index_end = round((len(df_val_eval)) * 1)
        df_val_eval = df_val_eval[index_start:index_end]
        df_val_eval.reset_index(drop=True, inplace=True)
        return df_val_eval

    # PREDICTION ACCURACY
    # OTHER
    # ------------------------------------------------------------------------------
    @staticmethod
    def calc_acc(dataset, origin=0, actual_col='actual', prediction_col='prediction'):
        # True if actual are in same direction as predictions
        dataset['same_slope'] = np.where(
            ((dataset[actual_col] >= origin) & (dataset[prediction_col] >= origin)) |
            ((dataset[actual_col] < origin) & (dataset[prediction_col] < origin)), 1, 0)
        directions = dataset['same_slope'].value_counts()
        # Percentage of positive values
        acc = (directions.iloc[0] / (directions.sum())) * 100
        acc = acc.round(2)
        print(f'Models Accuracy on the Test set is: {acc} %')
        return acc

    # ROOT MEAN SQUARED ERROR
    @staticmethod
    def calc_rmse(actual, prediction):
        df_rmse = pd.DataFrame(data=np.hstack((actual, prediction)),
                               columns=['actual', 'prediction'])
        rmse = ((df_rmse['actual'] - df_rmse['prediction']) ** 2).mean() ** .5
        return rmse
