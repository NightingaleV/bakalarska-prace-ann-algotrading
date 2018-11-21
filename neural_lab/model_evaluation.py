import numpy as np
import pandas as pd


class ModelEvaluation:

    def __init__(self, data_manager):
        # TODO Inherit Attributes For reference
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
    def create_test_eval_set(np_close, np_actual_test, np_prediction_test):
        df_test_eval = pd.DataFrame(data=np.hstack((np_close, np_actual_test, np_prediction_test)),
                                    columns=['close', 'actual', 'prediction'])
        df_test_eval.reset_index(drop=True, inplace=True)
        return df_test_eval

    def create_train_eval_set(self, np_actual_train, np_predictions_train):
        # TRAIN Dataframe for Evaluation - Close price, Actual / Prediction
        df_train_eval = pd.DataFrame(data=np.hstack((np_actual_train, np_predictions_train)),
                                     columns=['actual', 'prediction'])
        # Add close prices
        df_train_eval['close'] = self.data_manager.df['close'][self.n_past:].reset_index(drop=True)
        # Slice DataFrame - Only Train set without validation part
        index_start = int(self.data_manager.train_rows * 0)
        index_end = int(self.data_manager.train_rows * (1 - self.val_size))
        df_train_eval = df_train_eval[index_start:index_end]
        df_train_eval.reset_index(drop=True, inplace=True)
        return df_train_eval

    def create_val_eval_set(self, np_actual_train, np_predictions_train):
        # VALIDATION Dataframe for Evaluation - Close price, Actual / Prediction
        df_val_eval = pd.DataFrame(data=np.hstack((np_actual_train, np_predictions_train)),
                                   columns=['actual', 'prediction'])
        df_val_eval['close'] = self.data_manager.df['close'][self.n_past:].reset_index(drop=True)
        # Slice DataFrame - Only Validation set without train part
        index_start = int(self.data_manager.train_rows * (1 - self.val_size))
        index_end = int(self.data_manager.train_rows * 1)
        df_val_eval = df_val_eval[index_start:index_end]
        df_val_eval.reset_index(drop=True, inplace=True)

    # PREDICTION ACCURACY
    # OTHER
    # ------------------------------------------------------------------------------
    @staticmethod
    def calc_acc(dataset, origin=0, actual_slope='actual', predicted_slope='prediction'):
        # True if actual are in same direction as predictions
        dataset['same_slope'] = np.where(
            ((dataset[actual_slope] >= origin) & (dataset[predicted_slope] >= origin)) |
            ((dataset[actual_slope] < origin) & (dataset[predicted_slope] < origin)), 1, 0)
        directions = dataset['same_slope'].value_counts()
        # Percentage of positive values
        acc = (directions.iloc[0] / (directions.sum())) * 100
        return acc.round(2)

    # ROOT MEAN SQUARED ERROR
    @staticmethod
    def calc_rmse(actual, prediction):
        df_rmse = pd.DataFrame(data=np.hstack((actual, prediction)),
                               columns=['actual', 'prediction'])
        rmse = ((df_rmse['actual'] - df_rmse['prediction']) ** 2).mean() ** .5
        return rmse