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

    
