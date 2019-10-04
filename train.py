import argparse 
import collections

import torch 
import numpy as np

from parse_config import ConfigParser
from utils.train_utils import load_data, oversampling, mini_batchfy
from utils.data_loader import SentenceDataset, batchfy

def main_train(config): 
    
    logger = config.get_logger('train')

    vocabs = []
    datasets = []
    langs = [] 

    logger.info("SRC LANG : %s", config["src_data_prefix"])
    logger.info("TGT LANG : %s", config["tgt_data_prefix"])

    logger.debug("-- loading dataset --")
   
    src_data, src_vocab = load_data(config["src_data"], config["src_data_prefix"])
    tgt_data, tgt_vocab = load_data(config["tgt_data"], config["tgt_data_prefix"])

    src_data.vectorized_corpus = np.asarray(src_data.vectorized_corpus) 
    tgt_data.vectorized_corpus = np.asarray(tgt_data.vectorized_corpus)

    src_data.length = np.asarray(src_data.length) 
    tgt_data.length = np.asarray(tgt_data.length)

    logger.debug("-- finished loading dataset --")

    # check for need of oversampling 
    if len(src_data.tokenized_corpus) != len(tgt_data.tokenized_corpus):
        logger.debug("-- oversampling will be performed --")
        src_data, tgt_data = oversampling(src_data, tgt_data)
        logger.debug("-- finished oversampling -- ")

    logger.debug("-- generating mini-batches of size %d -- ", config["batch_size"])
    
    src_dataset = SentenceDataset(src_data.vectorized_corpus, src_data.length, config["batch_size"], src_vocab.special_tokens)
    tgt_dataset = SentenceDataset(tgt_data.vectorized_corpus, tgt_data.length, config["batch_size"], tgt_vocab.special_tokens)

    # src_batches = batchfy(dataset=src_dataset, batch_size=1) 
    # tgt_batches = batchfy(dataset=tgt_dataset, batch_size=1)

    logger.debug("-- total of %d batches are created --", len(src_dataset.bacth_idx_list))

    logger.debug("-- building model: %s --", config["name"])

    logger.debug("-- model is ready -- ")

    if config["gpu_id"]:
        logger.debug("-- GPU is used for the training --")

    logger.debug("++ starting the training ++")
    # model

    # train
    
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
        "--src_data",
        type=str,
        default=None,
        required=True,
        help="directory that contains preprocessed source dataset"
    )

    args.add_argument(
        "--tgt_data",
        type=str,
        default=None,
        required=True,
        help="directory that contains preprocessed target dataset"
    )

    args.add_argument(
        "--src_data_prefix",
        type=str,
        default=None,
        help="src file name not including extension. (eg: en_50k)"
    )

    args.add_argument(
        "--tgt_data_prefix",
        type=str,
        default=None,
        help="tgt file name not including extension. (eg: ja_50k)"
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