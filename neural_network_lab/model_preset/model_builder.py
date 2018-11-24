# Imports
import os
import numpy as np
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Building Neural Network
from keras.layers import Dense, Activation, Dropout
from keras.layers import BatchNormalization, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam


# Custom
# from .model_strategies import ModelStrategies
# from .model_evaluation import ModelEvaluation


class ModelBuilder:
    _models_folder = 'trained_models'

    def __init__(self):
        print('Initialize Model Builder')
        # MODELS

        self._model_name = None
        self.compiled_model = None
        self.trained_model = None
        self.training_history = None
        self.vector_shape = 0

        # GENERAL
        self._model_name = None
        self.val_size: float = 0.5

        # PREDICTION PERIODS
        self._predict_ma: int = 30
        self.n_past: int = 10
        self.n_future: int = 3

        # PROPERTIES
        self.epochs: int = 100
        self.loss_func: str = 'binary_crossentropy'
        self.activation_func: str = 'tanh'
        self.output_func: str = 'sigmoid'
        self.batch_size: int = 32
        self.starting_learn_rate: float = 0.01

        # BEHAVIORS
        self.shuffle_inputs: bool = False
        self.do_batch_norm: bool = False
        # Dropout
        self.dropout_rate: float = 0.5
        self.do_dropout: bool = False
        # Dynamic Learning Rate
        self.learning_rate_patience: int = 0
        self.learning_rate_reduction: float = 0.90
        # Dynamic Training Stop
        self.stop_training_patience = 50

        # SCORING
        self.monitor_metric: str = 'acc'
        self.val_monitor_metric: str = 'val_' + self.monitor_metric
        self.train_score: float = 0.
        self.val_score: float = 0.
        self.val_score_max_in_epoch: int = 0

    @property
    def models_folder(self):
        return self._models_folder

    @models_folder.setter
    def models_folder(self, value):
        self._models_folder = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def predict_ma(self):
        return self._predict_ma

    @predict_ma.setter
    def predict_ma(self, value):
        self._predict_ma = value

    # MODEL WORKFLOW
    # ------------------------------------------------------------------------------
    def create_train_vectors(self, df_train, scaled_df_train):
        n_columns = len(df_train.columns)
        last_column = n_columns - 1
        last_column_scaled_df = last_column - 2
        # Train Vectors
        x_train, y_train = [], []
        for i in range(self.n_past, len(df_train) - self.n_future + 1):
            x_train.append(scaled_df_train[i - self.n_past:i, 0:last_column])
            y_train.append(df_train.iloc[i:i + 1, last_column].values)
        # Vectors must be in numpy arr
        x_train, y_train = np.array(x_train), np.array(y_train)
        # Reshape Vector
        y_train = y_train.reshape(y_train.shape[0], 1)
        # Save vector shape for NN input layer
        self.vector_shape = x_train.shape

        return x_train, y_train

    def create_test_vectors(self, df_test, scaled_df_test, df_test_close):
        n_columns = len(df_test.columns)
        last_column = n_columns - 1
        # Test Vectors
        x_test, y_test, y_test_price = [], [], []
        for i in range(self.n_past, len(df_test) - self.n_future + 1):
            x_test.append(scaled_df_test[i - self.n_past:i, 0:last_column])
            y_test.append(df_test.iloc[i:i + 1, last_column].values)
            y_test_price.append(df_test_close.iloc[i:i + 1].values)
        # Vectors must be in numpy arr
        x_test, y_test, y_test_price = np.array(x_test), np.array(y_test), np.array(
            y_test_price)
        # Reshape Vector
        y_test, y_test_price = y_test.reshape(y_test.shape[0], 1), y_test_price.reshape(
            y_test.shape[0], 1)

        return x_test, y_test, y_test_price

    def build_network(self):
        # Build model

        # Compile model
        optimizer = Adam(lr=self.starting_learn_rate)
        self.compiled_model.compile(
            optimizer=optimizer,
            loss=self.loss_func,
            metrics=[self.monitor_metric])
        print('Neural Network successfully compiled')
        return self.compiled_model

    # Compile model + load saved weights
    def load_network(self):
        self.build_network()
        self.trained_model = self.compiled_model
        self.trained_model.load_weights(
            filepath=f'{self.models_folder}/{self.model_name}/{self.model_name}.hdf5',
            by_name=False)
        print('Weights successfully imported')
        return self.trained_model

    # Compile model + retrain model
    def train_network(self, x_train, y_train, verbose: int = 1):
        """
        :param x_train:
        :param y_train:
        :param verbose: int -- printing training progress
        :return:
        """
        # Create Folder if not exist
        self.create_folder(self.model_name)

        # Compile Model
        self.build_network()
        self.trained_model = self.compiled_model
        # CALLBACKS
        # Dynamic Training Stop
        stop_training = EarlyStopping(monitor=self.val_monitor_metric,
                                      min_delta=1e-10,
                                      patience=self.stop_training_patience,
                                      verbose=verbose)
        # Dynamic Learning Rate
        reduce_learning_rate = ReduceLROnPlateau(monitor=self.val_monitor_metric,
                                                 factor=self.learning_rate_reduction,
                                                 patience=self.learning_rate_patience,
                                                 min_lr=0.000001, verbose=verbose)
        # Save Best Model
        self_checkpoint = ModelCheckpoint(
            filepath=f'{self.models_folder}/{self.model_name}/{self.model_name}.hdf5',
            monitor=self.val_monitor_metric,
            verbose=1, save_best_only=True)

        # Tensorboard
        # tensor_board = TensorBoard(log_dir='{}'.format(self), write_graph=True, write_images=True)

        # Train Model
        self.training_history = self.trained_model.fit(x_train,
                                                       y_train,
                                                       shuffle=self.shuffle_inputs,
                                                       epochs=self.epochs,
                                                       callbacks=[stop_training,
                                                                  reduce_learning_rate,
                                                                  self_checkpoint],
                                                       validation_split=self.val_size,
                                                       verbose=verbose,
                                                       batch_size=self.batch_size)

        # Set Score Metrics
        self.set_score_values()

        # Round Score Metrics
        self.round_score_values()

        return self.trained_model, self.training_history

    def set_score_values(self):
        # Set Value of best val metric
        self.val_score = np.max(self.training_history.history[self.val_monitor_metric], axis=0)
        # Set Index of best val metric
        self.val_score_max_in_epoch = np.argmax(
            self.training_history.history[self.val_monitor_metric],
            axis=0)
        # Set Value of best val metric
        self.train_score = self.training_history.history[self.monitor_metric][
            self.val_score_max_in_epoch]

    def round_score_values(self):
        if self.monitor_metric is 'acc':
            # For Classification
            self.train_score = (self.train_score * 100).round(2)
            self.val_score = (self.val_score * 100).round(2)
        else:
            # For Regression
            self.train_score = self.train_score.round(6)
            self.val_score = self.val_score.round(6)

    # MODEL BUILDING TASKS
    # ------------------------------------------------------------------------------
    # Add Dropout layer inside model
    def add_dropout(self):
        if self.do_dropout:
            self.compiled_model.add(Dropout(self.dropout_rate))
            return self.compiled_model

    # Batch Normalization - Apply Z-score on inputs inside NN
    def add_batch_norm(self):
        if self.do_batch_norm:
            self.compiled_model.add(BatchNormalization())
            return self.compiled_model

    # Add layer for flattening the dimension of input
    def add_flat(self):
        self.compiled_model.add(Flatten())
        return self.compiled_model

    # VISUALIZE TRAINING
    # ------------------------------------------------------------------------------
    def plot_training_loss(self, show=True):
        sns.set()
        plt.plot(self.training_history.history['loss'])
        plt.plot(self.training_history.history['val_loss'])
        y_bottom_border = self.training_history.history['loss'][-1] - 0.05
        y_top_border = self.training_history.history['loss'][1] + 0.125
        plt.ylim(y_bottom_border, y_top_border)
        plt.title('Model Training Error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(f'{self.models_folder}/{self.model_name}/training_error.png',
                    bbox_inches='tight', dpi=150)
        if show:
            return plt.show()
        else:
            plt.show(block=False)
            return plt.close()

    def plot_training_metric(self, show=True):
        sns.set()
        plt.plot(self.training_history.history[self.monitor_metric])
        plt.plot(self.training_history.history[self.val_monitor_metric])
        plt.title('Model Training Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='lower right')
        plt.savefig(f'{self.models_folder}/{self.model_name}/training_accuracy.png',
                    bbox_inches='tight', dpi=150)
        if show:
            return plt.show()
        else:
            plt.show(block=False)
            return plt.close()

    # OTHER
    # ------------------------------------------------------------------------------
    def create_folder(self, name):
        # Dir to save results
        print(f'Inside {self.models_folder} create {name}')
        if not os.path.exists('{}/{}'.format(self.models_folder, name)):
            os.makedirs('{}/{}'.format(str(self.models_folder), name))


# class ModelPreset(ModelBuilder, ModelEvaluation, ModelStrategies):
#
#     def __init__(self, data_manager=None):
#         super(ModelPreset, self).__init__()
#         print('Initialize ModelPreset')
#
