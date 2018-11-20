# Imports
import os
import numpy as np
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Building Neural Network
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
# Custom
from .model_strategies import ModelStrategies
from .model_evaluation import ModelEvaluation


class ModelBuilder:

    models_folder = 'trained_models'

    def __init__(self):

        # MODELS
        self.compiled_model = None
        self.trained_model = None
        self.training_history = None

        # GENERAL
        self.model_name: str = 'model_ann'
        self.val_size: float = 0.5

        # PROPERTIES
        self.epochs: int = 100
        self.loss_func: str = 'binary_crossentropy'
        self.batch_size: int = 32
        self.optimizer: str = 'Adam'
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

    # MODEL WORKFLOW
    # ------------------------------------------------------------------------------
    def build_network(self):
        # Build model

        # Compile model
        self.compiled_model.compile(
            optimizer=eval(self.optimizer + f'(lr={self.starting_learn_rate})'),
            loss=self.loss_func,
            metrics=[self.monitor_metric])

        return self.compiled_model

    # Compile model + load saved weights
    def load_network(self):
        self.trained_model = self.build_network()
        self.trained_model.load_weights(
            filepath=f'{self.models_folder}/{self.model_name}/{self.model_name}.hdf5',
            by_name=False)
        return self.trained_model

    # Compile model + retrain model
    def train_network(self, X_train, y_train, verbose: int = 1):
        """
        :param X_train:
        :param y_train:
        :param verbose: int -- printing training progress
        :return:
        """
        # Compile Model
        self.trained_model = self.build_network()
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
        self.training_history = self.trained_model.fit(X_train,
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
    def plot_training_loss(self):
        sns.set()
        plt.plot(self.training_history.history['loss'])
        plt.plot(self.training_history.history['val_loss'])
        y_bottom_border = self.training_history.history['loss'][-1] - 0.02
        y_top_border = self.training_history.history['loss'][1] + 0.125
        plt.ylim(y_bottom_border, y_top_border)
        plt.title('Model Training Error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(f'{self.models_folder}/{self.model_name}/{self.model_name}_error.png',
                    bbox_inches='tight', dpi=150)
        return plt.show()

    def plot_training_metric(self):
        sns.set()
        plt.plot(self.training_history.history[self.monitor_metric])
        plt.plot(self.training_history.history[self.val_monitor_metric])
        plt.title('Model Training Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='lower right')
        plt.savefig(f'{self.models_folder}/{self.model_name}/{self.model_name}_accuracy.png',
                    bbox_inches='tight', dpi=150)
        return plt.show()

    # OTHER
    # ------------------------------------------------------------------------------
    @classmethod
    def create_folder(cls, name):
        # Dir to save results
        if not os.path.exists(f'{cls.models_folder}/{name}'):
            os.makedirs(f'{cls.models_folder}/{name}')


class ModelPreset(ModelBuilder, ModelEvaluation, ModelStrategies):

    def __init__(self):
        ModelBuilder.__init__(self)
        ModelEvaluation.__init__(self)
        ModelStrategies.__init__(self)
