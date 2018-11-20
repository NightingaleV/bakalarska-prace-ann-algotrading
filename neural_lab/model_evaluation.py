import numpy as np
import pandas as pd


class ModelEvaluation:

    def __init__(self):
        # Accuracy
        self.test_score = 0

        # RootMeanSquaredError
        self.rmse_train = 0
        self.rmse_test = 0

    # PREDICTION ACCURACY
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