# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.data import parse, preprocess


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath=os.environ.get('INPUT_FILEPATH'), output_filepath=os.environ.get('OUTPUT_FILEPATH')):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    log = logging.getLogger(__name__)
    log.info('making final data set from raw data')

    csv_file = parse.main()

    dataloader, num_classes = preprocess.main(csv_file)

    return dataloader, num_classes


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
