import argparse 
import collections

import torch 

from parse_config import ConfigParser
from utils.train_utils import load_data

if __name__ == "__main__": 
    args = argparse.ArgumentParser(description="Train for CLWE")
    
    args.add_argument(
        "-c", 
        "--config", 
        type=str,
        default=None,
        help="config file path (default:None)"
        )

    args.add_argument(
        "-d", 
        "--data",
        type=str,
        default=None,
        required=True,
        help="data file path (can be multiple)"
    )

    args.add_argument(
        "--lang",
        type=str,
        nargs="+",
        default=None,
        help="name of the language"
    )

    args.add_argument(
        "--data_prefix",
        type=str,
        nargs="+",
        default=None,
        help="file name not including extension. eg: en_50k"
    )

    args.add_argument(
        "-s", 
        "--save",
        type=str,
        default=None,
        help="file path to save the output"
    )

    # custom cli options to modify configuration from default values give in config file
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('batch_size'))
    ]
    config = ConfigParser(args, options)

    # get data 
    vocabs = []
    datasets = []
    langs = [] 

    for i in range(config.num_lang):
        vocab, dataset = load_data(config['data'], config["lang"][i], config["data_prefix"][i])

        vocabs.append(vocab)
        datasets.append(dataset)
        langs.append(config["lang"][i])

    # Train 

    ################################
    # Test run 
    # python3 train.py -c parameters.json -d data/processed_data --lang en --data_prefix en_50k -s result_emb/