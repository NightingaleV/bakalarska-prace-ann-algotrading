# KERAS
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras import backend as K
# CUSTOM
from neural_lab.model_builder import ModelBuilder
from neural_lab.model_evaluation import ModelEvaluation
from neural_lab.model_strategies import ModelStrategies


class ModelNeuralNetwork(ModelStrategies, ModelEvaluation, ModelBuilder):

    def __init__(self, data_manager):
        # Inherit Model Preset
        super(ModelNeuralNetwork, self).__init__(data_manager)
        print('Initialize ModelNeuralNetwork')

        # GENERAL
        self.model_task: str = 'classification'
        self.model_postfix: str = ''
        self.model_name: str = 'model_name'
        self.val_size: float = 0.15
        self.test_size: float = 0.2

        # PREDICTION PERIODS
        self.predict_ma: int = 30
        self.n_past: int = 10
        self.n_future: int = 3

        # PROPERTIES
        self.neurons_hidden: int = 25
        self.neurons_output: int = 1
        self.epochs: int = 100
        self.loss_func: str = 'binary_crossentropy'
        self.activation_func: str = 'tanh'
        self.output_func: str = 'sigmoid'
        self.batch_size: int = 32
        self.optimizer: str = 'Adam'
        self.starting_learn_rate: float = 0.01

        # BEHAVIORS
        self.shuffle_inputs: bool = True
        self.do_batch_norm: bool = True
        # Dropout
        self.dropout_rate: float = 0.5
        self.do_dropout: bool = True
        # Dynamic Learning Rate
        self.learning_rate_patience: int = 5
        self.learning_rate_reduction: float = 0.90
        # Dynamic Training Stop
        self.stop_training_patience = 50

        # SCORING
        self.monitor_metric: str = 'acc'
        self.val_monitor_metric: str = 'val_' + self.monitor_metric

        # Set custom model name
        self.set_model_name()

    def build_network(self):
        # Clear Tensors, for iteration purpose
        K.clear_session()

        # NEURAL NETWORK
        self.compiled_model = Sequential()

        # 1. HIDDEN LAYER
        self.compiled_model.add(Dense(units=self.neurons_hidden,
                                      kernel_initializer='uniform',
                                      input_shape=(self.vector_shape[1], self.vector_shape[2])))
        # Batch Normalization
        self.add_batch_norm()
        # Activation Function
        self.compiled_model.add(Activation(self.activation_func))
        # Dropout
        self.add_dropout()

        # 2. HIDDEN LAYER
        self.compiled_model.add(Dense(units=self.neurons_hidden,
                                      kernel_initializer='uniform',
                                      activity_regularizer=regularizers.l2(0.01)))
        # Batch Normalization
        self.add_batch_norm()
        # Activation Function
        self.compiled_model.add(Activation(self.activation_func))
        # Dropout
        self.add_dropout()

        # OUTPUT LAYER
        self.add_flat()
        self.compiled_model.add(Dense(units=self.neurons_output, activation=self.output_func,
                                      kernel_initializer='uniform'))
        # COMPILE MODEL
        super(ModelNeuralNetwork, self).build_network()

    def set_model_name(self):
        self.model_name = f'{self.model_task}_{self.data_manager.symbol_slug}_' \
                          f'{self.data_manager.postfix}' \
                          f'_MA{self.predict_ma}' \
                          f'_past{self.n_past}_fut{self.n_future}_{self.model_postfix}'
