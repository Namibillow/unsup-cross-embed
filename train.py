import argparse 
import collections

import torch 
import numpy as np

from parse_config import ConfigParser
from utils.train_utils import load_data, oversampling
from utils.data_loader import SentenceDataset, batchfy
from model import BiLSTM

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

    src_batches = batchfy(dataset=src_dataset, batch_size=1, shuffle=True) 
    tgt_batches = batchfy(dataset=tgt_dataset, batch_size=1, shuffle=True)

    logger.debug("-- total of %d batches are created --", len(src_dataset.bacth_idx_list))
 
    ####### Needs to be done in trainer #############################
    logger.debug("-- building model: %s --", config["name"])

    if config["gpu_id"]:
        logger.debug("-- GPU is used for the training --")
        device = torch.device("cuda")
    else:
        logger.debug("-- CPU is used for the training --")
        device = torch.device("cpu")

    # model = BiLSTM().device() 

    # logger.debug("-- model is ready -- ")

    # criterion = nn.CrossEntropyLoss()

    # optimizer = optim.ASGD(model.parameters(), lr=lr_rate)

    # trainer = Trainer(model, criterion, optimizer,
    #                   config=config,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   lr_scheduler=lr_scheduler)

    # trainer.train()



    # src_fwd_inputs, src_fwd_outputs, src_bwd_inputs, src_bwd_outputs = next(iter(src_dataset))
    # print(type(src_fwd_inputs))
    # print(type(src_fwd_inputs[0]))
    # print(type(src_fwd_inputs[0][0]))
    # print(src_fwd_inputs)
    # src_fwd_inputs.to(device)

    logger.debug("++ starting the training ++")
    for e in range(1, config["epoch"]+1):
        for (src, tgt) in zip(src_batches, tgt_batches):

            src_fwd_inputs, src_fwd_outputs, src_bwd_inputs, src_bwd_outputs = src
            tgt_fwd_inputs, tgt_fwd_outputs, tgt_bwd_inputs, tgt_bwd_outputs = tgt
        
            # Send data to the GPU
            src_fwd_inputs, src_fwd_outputs = src_fwd_inputs.to(device), src_fwd_outputs.to(device)
            src_bwd_inputs, src_bwd_outputs = src_bwd_inputs.to(device), src_bwd_outputs.to(device)
            tgt_fwd_inputs, tgt_fwd_outputs = tgt_fwd_inputs.to(device), tgt_fwd_outputs.to(device)
            tgt_bwd_inputs, tgt_bwd_outputs = tgt_bwd_inputs.to(device), tgt_bwd_outputs.to(device)
            
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()        # compute gradients of all variables wrt loss

            optimizer.step()       # perform updates using calculated gradients

        logger.info("epoch %d/%d  - loss %d")

    #####################################################################
    logger.debug("+"* 30)
    
    logger.debug("-- saving the embedding --")
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