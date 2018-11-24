import logging
import datetime


class Logger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Data Manager
        self.dm = None
        # Model
        self.model = None
        self.file_handler = None
        self.logger.setLevel(logging.DEBUG)

    def set_model(self, model):
        self.model = model
        self.delete_handler()
        if self.model is not None:
            self.model.create_folder(self.model.model_name)
        self.file_handler = logging.FileHandler(
            filename=f'{self.model.models_folder}/{self.model.model_name}/'
                     f'{self.model.model_name}_info.log',
            mode='a')
        formatter = logging.Formatter('%(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        print('Model was synchronized with Logger')

    def set_data_manager(self, data_manager):
        self.dm = data_manager
        print('Data Manager was synchronized with Logger')

    def delete_handler(self):
        log = self.logger
        for hdlr in log.handlers[:]:
            log.removeHandler(hdlr)

    # Log optional string as a message
    def log_info(self, message):
        self.logger.info(message)

    # Log model report
    def log_model_info(self):
        model_info = self.get_report()
        self.logger.info(model_info)

    def get_report(self):
        main_content = f"""
****************************************
# GENERAL
MODEL:      {self.model.model_name} 
DATASET:    {self.dm.symbol} Timestamp: {self.dm.timeframe}
TIME:       {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Test size:  {self.model.test_size} / Val Size {self.model.val_size}
Task:       {self.model.model_task}
Moving Averages: {self.dm.mean_indicators}
Indicators: {self.dm.indicators}


# PREDICTIONS
n_past: {self.model.n_past} / n_future: {self.model.n_future}, 

# PROPERTIES
Hidden Neurons: {self.model.neurons_hidden} / Output Neurons: {self.model.neurons_output}
Epochs: {self.model.epochs}
Cost function: {self.model.loss_func}
Activation function: {self.model.activation_func}, Output function: {self.model.output_func}
Batch_size: {self.model.batch_size}, Optimizer: Adam - LR = {self.model.starting_learn_rate}

# BEHAVIORS
Shuffle Inputs:      {self.model.shuffle_inputs} 
Batch Normalization: {self.model.do_batch_norm}
Dropout:             {self.model.do_dropout} - {self.model.dropout_rate}

# SCORING - {self.model.monitor_metric}
Train metric:   {self.model.train_score}  
Val metric:     {self.model.val_score}, In Epoch {self.model.val_score_max_in_epoch}
Test metric:    {self.model.test_score}

# NN PREDICTION TRADING
Test Pip Profit:{self.model.nn_pred_strategy_pip_return} / Fees: {
self.model.nn_pred_strategy_fees}
Test Sharpe:    {self.model.nn_pred_strategy_sharpe} / Drawdown: {
self.model.nn_pred_strategy_max_drawdown} / WinRate: {self.model.nn_pred_strategy_win_pct}
Pip Profit: Train:{self.model.nn_pred_train_pip_return} / Val:{
self.model.nn_pred_val_pip_return}
# MACD STRATEGY TRADING
Test Pip Profit:{self.model.macd_strategy_pip_return} / Fees: {self.model.macd_strategy_fees}
Test Sharpe:    {self.model.macd_strategy_sharpe} / Drawdown: {
self.model.macd_strategy_max_drawdown} / Winrate: {self.model.macd_strategy_win_pct}
Pip Profit: Train:{self.model.macd_strategy_train_pip_return} / Val:{
self.model.macd_strategy_val_pip_return}
****************************************
        """

        return main_content
