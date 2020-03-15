"""step: parsing the log"""
import os.path as path

import mlflow

import config.local as config
from src.data import drain


def main():
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
