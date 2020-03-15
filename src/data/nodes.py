"""step: parsing the log"""
import os.path as path

import config.local as config
import mlflow
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from . import drain


def parse_log():
    """parsing log"""
    # Regular expression list for optional pre-processing (default: [])
    regex = []
    similarity_threshold = 0.6  # Similarity threshold
    depth = 4  # Depth of all leaf nodes
    result_dir = config.PARSING_RESULT_DIR  # Parsing result path

    # parsing log into structured CSV
    parser = drain.LogParser(config.LOG_FORMAT, indir=config.INPUT_DIR,
                             outdir=result_dir, depth=depth, st=similarity_threshold, rex=regex)
    parser.parse(config.LOG_FILE)

    mlflow.log_artifacts(result_dir, "parsed_log")

    return path.join(result_dir, config.LOG_FILE + '_structured.csv')


def generate_dataset(series):
    inputs = []
    outputs = []
    sequence = series.tolist()
    print('sequence size: {}'.format(len(sequence)))
    for i in range(len(sequence) - config.WINDOW_SIZE):
        inputs.append(sequence[i:i + config.WINDOW_SIZE])
        outputs.append(sequence[i + config.WINDOW_SIZE])

    dataset = TensorDataset(torch.tensor(
        inputs, dtype=torch.float), torch.tensor(outputs))

    print('Number of sequences({}): {}'.format('Event', len(inputs)))
    return dataset


def make_dataloader(csv_file):
    # read CSV
    df = pd.read_csv(csv_file)

    num_classes = len(df[config.EVENT_KEY].unique())
    print('Total event count:', num_classes)

    le = LabelEncoder()

    # generate dataset
    df[config.SEQUENCE_KEY] = le.fit_transform(df[config.EVENT_KEY])

    dataset = generate_dataset(df[config.SEQUENCE_KEY])

    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, pin_memory=True)

    return dataloader, num_classes
