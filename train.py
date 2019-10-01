import argparse 
import collections

import torch 

from parse_config import ConfigParser
from utils.train_utils import load_data, oversampling, mini_batchfy

def main_train(config): 
    
    logger = config.get_logger('train')

    vocabs = []
    datasets = []
    langs = [] 
    logger.info("SRC LANG : %s", config["lang"][0])
    logger.info("TGT LANG : %s", config["lang"][1])

    logger.info("Writing this to a log")

    for i in range(config.num_lang):
        # logger.debug(f"Loading {config["data_prefix"][i]} dataset...")
        dataset, vocab = load_data(config['data'], config["lang"][i], config["data_prefix"][i])

        vocabs.append(vocab)
        datasets.append(dataset)
        langs.append(config["lang"][i])

    assert len(datasets) == 2, "Sorry, we only supports cross-lingual embedding at time."

    # check for needs of oversampling 
    if len(datasets[0].tokenized_corpus) != len(datasets[1].tokenized_corpus):
        oversampling(datasets)

       
    # generate minibatches 
    data = mini_batchfy

    # print("Number of mini-batches", len(dataset.batch_idx_list))
    
    
    # Save the embedding 
    # config.save_file()

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
        help="name of the language (must be in ISO 2 Letter Language Codes)"
    )

    args.add_argument(
        "--data_prefix",
        type=str,
        nargs="+",
        default=None,
        help="file name not including extension. (eg: en_50k)"
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
        CustomArgs(flags=['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(flags=['--bs', '--batch_size'], type=int, target=('batch_size'))
    ]

    config = ConfigParser(args, options)
 
    main_train(config)



    ################################
    # Test run 
    # python3 train.py -c parameters.json -d data/processed_data --lang en --data_prefix en_50k -s result_emb/