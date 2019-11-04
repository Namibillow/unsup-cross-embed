from model import BiLSTM
from parse_config import ConfigParser
from trainer import Trainer
from utils.train_utils import load_data, oversampling, save_embedding
from utils.data_loader import SentenceDataset, batchfy

import argparse 
import collections

import torch 
import numpy as np

def main_train(config): 
    
    logger = config.get_logger('train')

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
        logger.info("-- oversampling will be performed --")
        src_data, tgt_data = oversampling(src_data, tgt_data)
        logger.info("-- finished oversampling -- ")

    logger.debug("-- generating mini batches of size %d -- ", config["batch_size"])
    
    src_dataset = SentenceDataset(src_data.vectorized_corpus, src_data.length, config["batch_size"], src_vocab.special_tokens)
    tgt_dataset = SentenceDataset(tgt_data.vectorized_corpus, tgt_data.length, config["batch_size"], tgt_vocab.special_tokens)

    src_batches = batchfy(dataset=src_dataset, batch_size=1, shuffle=True) 
    tgt_batches = batchfy(dataset=tgt_dataset, batch_size=1, shuffle=True)
    
    total_batches = len(src_dataset.bacth_idx_list)
    logger.debug("-- total of %d batches are created --", total_batches)
 
    logger.debug("-- building model: %s --", config["name"])

    model = BiLSTM(src_vocab, tgt_vocab, config)

    logger.debug("-- model is ready -- ")

    trainer = Trainer(model=model,
                      config=config,
                      total_batches=total_batches,
                      src_data_loader=src_batches,
                      tgt_data_loader=tgt_batches,
                      src_vocab=src_vocab,
                      tgt_vocab=tgt_vocab
                    )
    
    logger.debug("++ starting the training ++")

    best_model = trainer.train()

    logger.debug("++ training is done ++")
    
    logger.debug("-- saving the embedding --")

    # Save the embedding 
    save_embedding(best_model.embedding_src, config["emb_dim"], src_vocab, config.save_dir, config["src_data_prefix"])
    save_embedding(best_model.embedding_tgt, config["emb_dim"], tgt_vocab, config.save_dir, config["tgt_data_prefix"])

    logger.debug("-- completed --")


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
        help="file path to  save the output"
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
    #  python3 train.py -c parameters.json --src_data data/processed_data/low_resource/ja --src_data_prefix ja_50K --tgt_data data/processed_data/low_resource/en --tgt_data_prefix en_50K -s results
