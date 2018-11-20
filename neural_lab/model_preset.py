# Building Neural Network
# Import
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

class ModelBuilder:

    def __init__(self):
        self.model_name: str = None
        self.compiled_model = None
        self.trained_model = None
        self.history = None

        # VECTOR SEQUENCES
        self.n_past: int = None
        self.n_future: int = None

        self.do_batch_norm = None

        # Scoring
        self.monitor_metric: str = None
        self.val_monitor_metric = 'val_' + self.monitor_metric
        self.train_score: float = None
        self.val_score: float = None
        self.val_score_in_epoch_n: int = None
        self.test_score: float = None

    # MODEL WORKFLOW
    # ------------------------------------------------------------------------------
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
                                      patience=self.stop_patience,
                                      verbose=1)
        reduce_learning_rate = ReduceLROnPlateau(monitor=self.val_monitor_metric,
                                                 factor=self.learn_rate_reduce,
                                                 patience=self.learn_rate_patience,
                                                 min_lr=0.000001, verbose=1)
        self_checkpoint = ModelCheckpoint(
            filepath='trained_models/{}/{}.hdf5'.format(self.model_name, self.model_name),
            monitor=self.val_monitor_metric,
            verbose=1, save_best_only=True)
        # tensor_board = TensorBoard(log_dir='{}'.format(self), write_graph=True, write_images=True)
        self.history = self.trained_model.fit(X_train, y_train, shuffle=self.shuffle,
                                              epochs=self.epochs,
                                              callbacks=[stop_training, reduce_learning_rate,
                                                         self_checkpoint],
                                              validation_split=self.val_split, verbose=1,
                                              batch_size=self.batch_size)

        self.val_metric = np.max(self.history.history[self.val_monitor_metric], axis=0)
        self.val_metric_in_epoch_n = np.argmax(self.history.history[self.val_monitor_metric],
                                               axis=0)
        self.train_metric = self.history.history[self.monitor_metric][
            self.val_metric_in_epoch_n]

        # Rounding Score
        if self.monitor_metric is 'acc':
            self.train_metric = (self.train_metric * 100).round(2)
            self.val_metric = (self.val_metric * 100).round(2)
        else:
            self.train_metric = self.train_metric.round(6)
            self.val_metric = self.val_metric.round(6)

        return self.trained_model, self.history

    

