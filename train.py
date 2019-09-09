import argparse 
import collections

import torch 

from parse_config import ConfigParser

def main(config):
    pass


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
        nargs="+",
        default=None,
        required=True,
        help="data file path (can be multiple)"
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
    
    main(config)