from neural_lab.model_preset import ModelPreset


class NeuralNetworkModel(ModelPreset):

    def __init__(self):
        # Inherit Model Preset
        super(NeuralNetworkModel).__init__()

        self.n_past: int = 10
        self.n_future: int = 3

        # GENERAL
        self.model_name: str = 'model_ann'
        self.val_size: float = 0.15
        self.test_size: float = 0.2

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
