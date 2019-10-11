import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

class BiLSTM(nn.Module):
    """
    - Bidirectional LSTM module 
    """
    def __init__(self, src_vocab, tgt_vocab, config):
        """
        input:
            src_vocab
            tgt_vocab
            config
        """
        super(BiLSTM, self).__init__() 

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.hidden_dim = config["hidden_dim"]
        self.embedding_dim = config["emb_dim"]
        self.num_layers = config["num_layer"]
        self.drop_prob = config["dropout_rate"]
       

        ################# LAYERS ########################################      
        self.embedding_src = nn.Embedding(num_embeddings=self.src_vocab.vocab_len, embedding_dim=self.embedding_dim, padding_idx=self.src_vocab.special_tokens["PAD"])
        self.embedding_tgt = nn.Embedding(num_embeddings=self.tgt_vocab.vocab_len, embedding_dim=self.embedding_dim, padding_idx=self.tgt_vocab.special_tokens["PAD"])

        self.dropout = nn.Dropout(self.drop_prob)

        self.forward_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          dropout=self.drop_prob, batch_first=True)
        self.back_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          dropout=self.drop_prob, batch_first=True)

        self.output_layer_src = nn.Linear(in_features=self.hidden_dim, out_features=self.src_vocab.vocab_len, bias=False)
        self.output_layer_tgt = nn.Linear(in_features=self.hidden_dim, out_features=self.tgt_vocab.vocab_len, bias=False)
        

    def switch_lstm(self, type):
        if (type == "fwd"):
            self.lstm = self.forward_lstm

        elif (type == "bkw"):
            self.lstm = self.back_lstm

        else:
            raise Exception("Invalid type")

    def swithc_lang(self, type):
        
        if type == 0:
            self.output_layer = self.output_layer_src
            self.embedding = self.embedding_src
            self.curr = "src"
        elif type == 1:
            self.output_layer = self.output_layer_tgt
            self.embedding = self.embedding_tgt
            self.curr = "tgt"
        else:
            raise Exception("Invalid")

    def forward(self, inputs, sent_lengths):
        """
        input:
            lang: 
                - language name (string)
            sentences:
                - tensor (batch, seq_length, embedding)
        returns:
            forward_output:
                -
            backward_output:
                - 
        """
        # Weights for EOS shared 
        # Also the embedding of BOS and EOS needs to be shared 
        if self.curr == "src":
            self.embedding.weight.data[self.src_vocab.special_tokens["BOS_FWD"]] = self.embedding_tgt.weight.data[self.src_vocab.special_tokens["BOS_FWD"]]
            self.embedding.weight.data[self.src_vocab.special_tokens["BOS_BWD"]] = self.embedding_tgt.weight.data[self.src_vocab.special_tokens["BOS_BWD"]]
            self.output_layer.weight.data[self.src_vocab.special_tokens["EOS"]] = self.output_layer_tgt.weight.data[self.src_vocab.special_tokens["EOS"]]
        elif self.curr == "tgt":
            self.embedding.weight.data[self.src_vocab.special_tokens["BOS_FWD"]] = self.embedding_src.weight.data[self.src_vocab.special_tokens["BOS_FWD"]]
            self.embedding.weight.data[self.src_vocab.special_tokens["BOS_BWD"]] = self.embedding_src.weight.data[self.src_vocab.special_tokens["BOS_BWD"]]
            self.output_layer.weight.data[self.src_vocab.special_tokens["EOS"]] = self.output_layer_src.weight.data[self.src_vocab.special_tokens["EOS"]]
            
        inputs = self.embedding(inputs)
        h_t, (h_last, c_last) = self.lstm(inputs)

        score = self.output_layer(self.dropout(h_t))

        return score 

