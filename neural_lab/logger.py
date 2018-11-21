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
            filename=f'{self.model.models_folder}/{self.model.name}/{self.model.name}_info.log',
            mode='a')
        formatter = logging.Formatter('%(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

    def set_data_manager(self, data_manager):
        self.dm = data_manager

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
        MODEL: {self.model.model_name} 
        DATASET: {self.dm.symbol} Timestamp: {self.dm.timeframe}
        TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Test size: {self.model.test_size} / Val Size {self.model.val_size}
        Moving Averages: {self.model.mean_indicators}
        Indicators: {self.model.indicators}
        Target: {self.model.target}

        # VECTORS
        N_PAST: {self.model.n_past} / N_FUTURE: {self.model.n_future}, 

        # PROPERTIES
        Hidden / Output: {self.model.neurons} / {self.model.output_neurons}
        Epochs: {self.model.epochs}
        Cost function: {self.model.loss_func}
        Activation function: {self.model.active_func}, Output function: {self.model.output_func}
        Batch_size: {self.model.batch_size}, Optimizer: {self.model.optimizer} \- LR {
        self.model.starting_learn_rate}

        # BEHAVIORS
        Shuffle:    {self.model.shuffle} / Batch Normalization: {self.model.do_batch_norm} / \
        Dropout:{self.model.do_dropout} - {self.model.dropout_num}

        # SCORING
        Train metric:   {self.model.monitor_metric}: {self.model.train_score}, 
        Val metric:     {self.model.val_score}, In Epoch {self.model.val_score_max_in_epoch}
        TEST metric:    {self.model.test_score}

        # NN PREDICTION TRADING
        Pip Profit:     {self.model.nn_pred_strategy_pip_return} / Fees: {self.model.nn_pred_strategy_fees}
        Sharpe:         {self.model.nn_pred_strategy_sharpe} / Drawdown: \{
        self.model.nn_pred_strategy_max_drawdown} / WinRate: {self.model.nn_pred_strategy_win_pct}
        Pip Profit: Train Set:{self.model.nn_pred_train_pip_return} / Val:\{
        self.model.nn_pred_val_pip_return}
        # MACD STRATEGY TRADING
        Pip Profit:     {self.model.macd_strategy_pip_return} / Fees: {self.model.macd_strategy_fees}
        Sharpe:         {self.model.macd_strategy_sharpe} / Drawdown: \{
        self.model.macd_strategy_max_drawdown} / Winrate: {self.model.macd_strategy_win_pct}
        Pip Profit: Train Set:{self.model.macd_strategy_train_pip_return} / Val:{
        self.model.macd_strategy_val_pip_return}
        ****************************************
        """

        return main_content



