INPUT_DIR = 'data/raw/'  # The input directory of log file
OUTPUT_DIR = 'result/'  # The output directory of parsing results
LOG_FILE = 'log4j-2020-02-28-22.log'  # The input log file name
LOG_FORMAT = '<Date> <Time> <Level> <Component>: <Content>'  # HDFS log format
EVENT_KEY = 'EventTemplate'  # The column name for event unique name
SEQUENCE_KEY = 'Event'  # The column name for encoded event key, used for generating sequences
WINDOW_SIZE = 10  # The windows size for LSTM
BATCH_SIZE = 16  # The data loader batch size
NUM_EPOCHS = 100  # Number of epochs for training
INPUT_SIZE = 1  # The length of LSTM input vector
HIDDEN_SIZE = 64  # The size of the hidden layer of LSTM
NUM_LAYERS = 2  # How many layers do we want to build the LSTM
DEVICE = 'cpu'  # The hardware device to use for training and predicting
MODEL_DIR = OUTPUT_DIR + '/model'  # The path for saving model file
NUM_CANDIDATES = 9  # The number of possible predicted classes to consider
# The testing sample for normal log sequence
NORMAL_SAMPLE = [67, 83, 30, 74, 73, 0, 22, 22, 24, 173, 186, 117, 79, 44, 45, 66, 205]
# The testing sample for abnormal log sequence
ABNORMAL_SAMPLE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
