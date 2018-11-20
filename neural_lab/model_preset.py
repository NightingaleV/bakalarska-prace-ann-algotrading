# Imports
import numpy as np
# Building Neural Network
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


class ModelBuilder:

    def __init__(self):
        # self.model_name: str = None
        # self.compiled_model = None
        # self.trained_model = None
        # self.history = None
        #
        # # VECTOR SEQUENCES
        # self.n_past: int = None
        # self.n_future: int = None
        #
        # self.do_batch_norm = None
        #
        # # Scoring
        # self.monitor_metric: str = None
        # self.val_monitor_metric = 'val_' + self.monitor_metric
        # self.train_score: float = None
        # self.val_score: float = None
        # self.val_score_in_epoch_n: int = None
        # self.test_score: float = None

        # GENERAL
        self.starting_learn_rate = None
        self.optimizer: str = 'Adam'
        self.model_name: str = 'model_ann'
        self.val_split: float = 0.5

        # PROPERTIES
        self.epochs: int = 100
        self.loss_func: str = 'binary_crossentropy'
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

    # MODEL WORKFLOW
    # ------------------------------------------------------------------------------
    def build_network(self, model):
        model.compile(optimizer=eval(self.optimizer + f'(lr={self.starting_learn_rate})'),
                      loss=self.loss_func,
                      metrics=[self.monitor_metric])

        return model

    # Compile model + load saved weights
    def load_network(self):
        self.trained_model = self.build_network()
        self.trained_model.load_weights(
            filepath='trained_models/{}/{}.hdf5'.format(self.model_name, self.model_name),
            by_name=False)
        return self.trained_model

    # Compile model + retrain model
    def train_network(self, X_train, y_train):
        self.trained_model = self.build_network()
        # Callbacks
        stop_training = EarlyStopping(monitor=self.val_monitor_metric, min_delta=1e-10,
                                      patience=self.stop_training_patience,
                                      verbose=1)
        reduce_learning_rate = ReduceLROnPlateau(monitor=self.val_monitor_metric,
                                                 factor=self.learning_rate_reduction,
                                                 patience=self.learning_rate_patience,
                                                 min_lr=0.000001, verbose=1)
        self_checkpoint = ModelCheckpoint(
            filepath='trained_models/{}/{}.hdf5'.format(self.model_name, self.model_name),
            monitor=self.val_monitor_metric,
            verbose=1, save_best_only=True)
        # tensor_board = TensorBoard(log_dir='{}'.format(self), write_graph=True, write_images=True)
        self.history = self.trained_model.fit(X_train, y_train, shuffle=self.shuffle_inputs,
                                              epochs=self.epochs,
                                              callbacks=[stop_training, reduce_learning_rate,
                                                         self_checkpoint],
                                              validation_split=self.val_split, verbose=1,
                                              batch_size=self.batch_size)

        # Set Score Metrics
        self.set_score_values()

        # Round Score Metrics
        self.round_score_values()

        return self.trained_model, self.history

    def set_score_values(self):
        # Set Value of best val metric
        self.val_score = np.max(self.history.history[self.val_monitor_metric], axis=0)
        # Set Index of best val metric
        self.val_score_max_in_epoch = np.argmax(self.history.history[self.val_monitor_metric],
                                                axis=0)
        # Set Value of best val metric
        self.train_score = self.history.history[self.monitor_metric][self.val_score_max_in_epoch]

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
    def add_dropout(self, model):
        if self.do_dropout:
            model.add(Dropout(self.dropout_rate))
            return model

    # Batch Normalization - Apply Z-score on inputs inside NN
    def add_batch_norm(self, model):
        if self.do_batch_norm:
            model.add(BatchNormalization())
            return model

    # Add layer for flattening the dimension of input
    @staticmethod
    def add_flat(model):
        model.add(Flatten())
        return model
