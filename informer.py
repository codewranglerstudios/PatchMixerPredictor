import asyncio
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
from exp.exp_main import Exp_Main
import matplotlib.pyplot as plt
from data.stock_data_fetcher import StockDataFetcher
from data.sentiment_analyzer_ver2 import SentimentAnalyzer
from utils.tools import dotdict, visual


class Informer:
    def __init__(self, symbol, data=None, root_path=None, data_path=None, batch_size=128, seq_len=288, label_len=0,
                 pred_len=96,
                 output_size=1,
                 use_saved_data=False,
                 load_saved_model=True,
                 add_unique_model_name=False,
                 train_now=True,
                 ignore_data=False,
                 d_model=256,
                 target='Close',
                 revin=1,  # revin normalisation
                 affin=1,  # affin normalisation
                 use_amp=True,  # 'use automatic mixed precision training'
                 train_epochs=20,
                 patients=50,
                 e_layers=2,
                 d_layers=1,
                 n_heads=8,
                 is_training=1,
                 do_predict=True
                 ):
        self.features = ['date',  # date is not a feature: length = len(features)-1
                         'id', 'interval', 'sentiment', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                         'RSI', 'EMA', 'SMA',
                         'BB_Upper', 'BB_Middle', 'BB_Lower',
                         'support', 'resistance', 'Close']
        self.d_model = d_model
        self.patients = patients
        self.is_training = is_training
        self.do_predict = do_predict
        self.target = target
        self.use_amp = use_amp
        self.output_size = output_size
        self.enc_in = len(self.features) - 1
        self.dec_in = len(self.features) - 1
        self.revin = revin
        self.affin = affin
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.train_epochs = train_epochs
        self.train_now = train_now
        self.symbol = symbol
        self.load_saved_model = load_saved_model
        self.add_unique_model_name = add_unique_model_name
        if root_path is None:
            root_path = f'./data/stock_data/'
        if data_path is None:
            data_path = f'{symbol}_stock_data.csv'
        self.root_path = root_path
        self.data_path = data_path
        self.symbol = symbol
        self.batch_size = batch_size  # input_shape - (batch_size, seq_len, len(features)-1)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.input_shape = [seq_len, label_len, pred_len]
        # if not ignore_data and data is None:
        # self.data = asyncio.run(self.get_data(symbol=self.symbol, use_saved=use_saved_data))
        # else:
        #   self.data = data
        self.data = data
        self.exp = None  # Initialize the experiment
        self.args = self.get_default_args()
        self.settings = None

    def get_default_args(self):
        args = dotdict()

        # random seed
        args.random_seed = 2021  # 'random seed')

        # basic config
        args.is_training = self.is_training  # 'status')
        args.model_id = 'test_1'  # 'model id')
        args.model = 'PatchMixer'  # help='model name, options: [PatchMixer, PatchTST, Autoformer, Informer, Transformer]')
        args.des = 'stock_predictor'  # 'exp description')
        if self.add_unique_model_name:
            args.des = f"{self.symbol}_stock_predictor"

        # forecasting task
        args.seq_len = self.seq_len  # 'input sequence length')
        args.label_len = self.label_len  # 'start token length')
        args.pred_len = self.pred_len  # 'prediction sequence length')

        # optimization
        args.num_workers = 2  # 10  # 'data loader num workers')
        args.itr = 1  # 'experiments times')
        args.train_epochs = self.train_epochs  # 'train epochs')
        args.batch_size = self.batch_size  # 128  # 'batch size of train input data')
        args.patience = self.patients  # 'early stopping patience')
        args.learning_rate = 0.0001  # 'optimizer learning rate')
        args.loss = 'mse'  # 'loss function')
        args.lradj = 'type3'  # 'adjust learning rate')
        args.pct_start = 0.3  # 'percentage start')
        args.use_amp = self.use_amp  # 'use automatic mixed precision training'  # default=False)

        # data loader
        args.data = 'custom'  # custom dataset type
        args.root_path = self.root_path  # 'root path of the data file')
        args.data_path = self.data_path  # 'data file')
        args.features = 'MS'  # help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        args.target = self.target  # 'target feature in S or MS task')
        args.freq = 'h'  # help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        args.checkpoints = './checkpoints/'  # 'location of model checkpoints')
        print(f"informer cols {self.features}")
        args.cols = self.features
        # DLinear
        # args.individual', action='store_true=False  # 'DLinear: a linear layer for each variate(channel) individually')

        # PatchTST
        args.fc_dropout = 0.05  # 'fully connected dropout')
        args.head_dropout = 0.0  # 'head dropout')
        args.patch_len = 16  # 'patch length')
        args.stride = 8  # 'stride')
        args.padding_patch = 'end'  # 'None: None; end: padding on the end')
        args.revin = self.revin  # 'RevIN; True 1 False 0 - normalisation technique
        args.affine = self.affin  # 'RevIN-affine; True 1 False 0 - normalisation
        args.subtract_last = 1  # '0: subtract mean; 1: subtract last')
        args.decomposition = 0  # 'decomposition; True 1 False 0')
        args.kernel_size = 25  # 'decomposition-kernel')
        args.individual = 0  # 'individual head; True 1 False 0')

        # Formers 
        args.embed_type = 2  # 1  # help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
        args.enc_in = self.enc_in  # len(self.features)  # help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
        args.dec_in = self.dec_in  # len(self.features)  # 'decoder input size')
        args.c_out = self.output_size  # 'output size')
        args.d_model = self.d_model  # 256  # 'dimension of model')
        args.n_heads = self.n_heads  # 'num of heads')
        args.e_layers = self.e_layers  # 'num of encoder layers')
        args.d_layers = self.d_layers  # 'num of decoder layers')
        args.d_ff = 128  # 'dimension of fcn')
        args.moving_avg = 128  # 'window size of moving average')
        args.factor = 1  # 'attn factor')
        args.distil = True  # help='whether to use distilling in encoder, using this argument means not using distilling'  # default=True)
        args.dropout = 0.05  # 'dropout')
        args.embed = 'learned'  # help='time features encoding, options:[timeF, fixed, learned]')
        args.activation = 'gelu'  # 'activation')
        args.output_attention = False  # 'whether to output attention in ecoder')
        args.do_predict = self.do_predict  # 'whether to predict unseen future data')

        # GPU
        args.use_gpu = False  # 'use gpu')
        args.gpu = 0  # 'gpu')
        args.use_multi_gpu = False  # 'use multiple gpus=False)
        args.devices = '0,1'  # 'device ids of multile gpus')
        args.test_flope = False  # 'See utils/tools for usage')

        # PatchMixer
        args.mixer_kernel_size = 8  # 'patchmixer-kernel')
        # args.a=1  # 'degree of patches aggregation, limited to 1-N, other value for N')
        args.loss_flag = 2  # help='loss function flag, 0 for MSE, 1 for MAE, 2 for both of MSE & MAE, 3 for SmoothL1loss')

        # random seed
        fix_seed = args.random_seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.dvices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        for _ in range(args.itr):
            # setting record of experiments
            self.settings = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.attn,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.mix,
                    args.des,
                _)
        self.args = args
        if self.train_now:
            self.train_and_predict(settings=self.settings, args=args)
        else:
            self.get_predictions()
        return args

    def train_and_predict(self, settings=None, args=None):
        print("starting training")
        if args is not None:
            self.args = args
        else:
            args = self.args
        if settings is not None:
            self.settings = settings
        else:
            settings = self.settings

        fix_seed = args.random_seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        self.exp = Exp_Main(args)

        # Train the model
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(settings))
        self.exp.train(setting=settings, load=self.load_saved_model)

        # Test the model
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(settings))
        self.exp.test(setting=settings, test=1, load=self.load_saved_model)

        # Predict
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(settings))
        prediction = self.exp.predict(setting=settings, load=self.load_saved_model)

        # Load predictions
        # prediction = np.save('../results/' + settings + '/real_prediction.npy')
        torch.cuda.empty_cache()

        return prediction

    def get_predictions(self):
        if self.args is None:
            self.args = self.get_default_args()
        if self.exp is None:
            self.exp = Exp_Main(self.args)
        if self.settings is None:
            args=self.args
            for _ in range(args.itr):
                self.settings = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.attn,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.mix,
                    args.des, _
                )
        settings=self.settings
        # Predict
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(settings))
        prediction = self.exp.predict(settings, True)

        # Load predictions
        # prediction = np.load(f'./results/{settings}/{self.symbol}_real_prediction.npy')
        torch.cuda.empty_cache()

        visual(preds=prediction)
        return prediction

    def plot_predictions(self, prediction):
        plt.figure()
        plt.plot(prediction[0, :, -1], label='Prediction')
        plt.legend()
        plt.show()

    async def get_data(self, symbol, use_api=False, use_saved=False, days=1500):

        print(f"loading use_saved {use_saved} ")
        if use_saved:
            print(f"loading")
            data = pd.read_csv(f"{self.root_path}{self.data_path}", index_col=0)
            data.sort_index(inplace=True)
            data.reset_index(inplace=True)
            print(f"data {data.columns}")
            self.data = data[self.features]
            return self.data
        if use_api:
            # Fetch stock data from API
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
            fetcher = StockDataFetcher(symbol)
            stock_data = fetcher.fetch_historical_stock_data(start_date, end_date)
        else:
            # Fetch stock data from historical CSV
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
            fetcher = StockDataFetcher(symbol)
            stock_data = fetcher.fetch_historical_stock_data(start_date, end_date)
        sentiment = SentimentAnalyzer(symbol)
        stock_data = sentiment.add_sentiment(stock_data)
        # Convert 'Datetime' to datetime object and sort as index
        if 'Datetime' in stock_data.index.names:
            stock_data.index.name = 'date'
            stock_data.reset_index(inplace=True)
        if 'Datetime' in stock_data.columns:
            stock_data['date'] = stock_data['Datetime']
            stock_data.drop('Datetime', axis=1, inplace=True)
        stock_data['Close'] = stock_data['Adj Close'].where(stock_data['Adj Close'] != 0, stock_data['Close'])
        stock_data.sort_index(inplace=True)
        if 'date' not in stock_data.columns:
            stock_data.reset_index(inplace=True)
        stock_data['id'] = self.string_to_float(symbol)
        stock_data['RSI'] = 0
        stock_data['EMA'] = 0
        # Reverse the DataFrame
        reversed_stock_data = stock_data.iloc[::-1]
        # Forward fill 'RSI' 'EMA' column
        stock_data['RSI'] = stock_data['RSI'].ffill()
        stock_data['EMA'] = stock_data['EMA'].ffill()
        stock_data = stock_data.ffill()

        # Replace any remaining NaN values in 'RSI' column with 0
        stock_data['RSI'].fillna(0, inplace=True)
        stock_data['EMA'].fillna(0, inplace=True)
        print(stock_data[['RSI', 'EMA']])
        # scraper = RedditScraper(self.symbol)
        # subreddits = await scraper.get_reddit_sentiment_by_days(days=days)
        # stock_data = scraper.merge_sentiment(stock_data)
        stock_data = stock_data[self.features]
        self.data = stock_data
        print(f"features: {self.features}")
        print(f"data: {self.data}")
        data_path = f'{symbol}_stock_data.csv'
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.to_csv(f"{self.root_path}{data_path}", index=True, index_label='date')
        self.args = self.get_default_args()
        return self.data

    def string_to_float(self, s: str) -> float:
        # Define a mapping for each character to a unique integer
        CHAR_MAP = {chr(i): i - 32 for i in range(32, 127)}
        # Reverse mapping to decode integer back to string
        REVERSE_CHAR_MAP = {v: k for k, v in CHAR_MAP.items()}
        # Maximum length of string we want to encode
        MAX_LENGTH = 6
        # Ensure the string is at most MAX_LENGTH characters long
        if len(s) > MAX_LENGTH:
            raise ValueError(f"String length must be at most {MAX_LENGTH} characters")

        # Encode the string into a unique integer
        encoded_val = 0
        for char in s:
            encoded_val = encoded_val * len(CHAR_MAP) + CHAR_MAP[char]

        # Normalize the integer to a float between 0 and 1 with up to 6 decimal places
        max_val = (len(CHAR_MAP) ** MAX_LENGTH) - 1
        normalized_float = round(encoded_val / max_val, 6)

        return normalized_float

    def float_to_string(self, f: float) -> str:
        # Define a mapping for each character to a unique integer
        CHAR_MAP = {chr(i): i - 32 for i in range(32, 127)}
        # Reverse mapping to decode integer back to string
        REVERSE_CHAR_MAP = {v: k for k, v in CHAR_MAP.items()}
        # Maximum length of string we want to encode
        MAX_LENGTH = 6
        if f < 0 or f > 1:
            raise ValueError("Float must be between 0 and 1")
        # Denormalize the float back to an integer
        max_val = (len(CHAR_MAP) ** MAX_LENGTH) - 1
        denormalized_int = round(f * max_val)
        # Decode the integer back to the original string
        chars = []
        while denormalized_int > 0:
            chars.append(REVERSE_CHAR_MAP[denormalized_int % len(CHAR_MAP)])
            denormalized_int //= len(CHAR_MAP)

        # The characters are decoded in reverse order
        return ''.join(reversed(chars)).rjust(MAX_LENGTH, REVERSE_CHAR_MAP[0])
# Example parameters (adjustable based on specific use case)

# enc_in: Number of features in the encoder input (e.g., 7 for OHLCV + additional features)
# dec_in: Number of features in the decoder input (usually the same as enc_in)
# c_out: Number of output features (e.g., 1 for predicting the closing price)
# seq_len: Length of the input sequence (e.g., 300 time steps)
# label_len: Length of the sequence used for teacher forcing during training (e.g., 100)
# out_len: Length of the prediction sequence (e.g., 100)
# factor: Compression factor in ProbAttention (default is 5)
# d_model: Dimension of the model (default is 512)
# n_heads: Number of attention heads (default is 8)
# e_layers: Number of encoder layers (default is 3)
# d_layers: Number of decoder layers (default is 2)
# d_ff: Dimension of feedforward layers (default is 512)
# lstm_hidden_size: Number of hidden units in the LSTM layer (default is 256)
# lstm_num_layers: Number of LSTM layers (default is 1)
# dropout: Dropout rate (default is 0.05)
# attn: Type of attention ('prob' for ProbAttention, 'full' for FullAttention)
# embed: Type of embedding ('fixed' or 'learnable')
# freq: Frequency encoding for time series data ('h' for hourly, 'd' for daily)
# activation: Activation function (default is 'gelu')
# output_attention: Whether to output attention weights (default is False)
# distil: Whether to use a distilled model with ConvLayer (default is True)
# mix: Whether to mix attention mechanisms (default is True)
# device: Device to run the model on (default is GPU if available)
