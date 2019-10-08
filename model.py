import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F

class BiLSTM(nn.Module):
    """
    - Bidirectional LSTM module 
    """
    def __init__(self, src_vocab, tgt_vocab, config):
        """
        input:
            hidden_dim: 300 according to the paper 
            embedding_dim: 300 according to the paper 
            num_layers: number of hidden layers. 2 according to the paper.
            en_vocab_size: total vocab numbers
            zh_vocab_size: total vocab numbers
            output_size:
            batch_size:
            drop_prob: drop rate 
        """
        super(BiLSTM, self).__init__() 

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.hidden_dim = config["hidden_dim"]
        self.embedding_dim = config["emb_dim"]
        self.num_layers = config["num_layers"]
        self.drop_prob = config["dropout_rate"]
        self.weight_init_range = config["weight_init_range"]

        ################# LAYERS ########################################
        
        # src embedding 
        self.embedding_src = nn.Embedding(num_embeddings=self.src_vocab.vocab_len, embedding_dim=self.embedding_dim, padding_idx=self.src_vocab.special_tokens["PAD"])
        # tgt embedding 
        self.embedding_tgt = nn.Embedding(num_embeddings=self.tgt_vocab.vocab_len, embedding_dim=self.embedding_dim, padding_idx=self.tgt_vocab.special_tokens["PAD"])

        # Initialize all of the parameters using an uniform distribution
        for param in self.model.parameters():
            param.data.uniform_(self.weight_init_range[0], self.weight_init_range[1])

        # Unknown set to 0 vector
        self.embedding_src.weight.data[self.src_vocab.special_tokens["UNK"]] *= 0
        self.embedding_tgt.weight.data[self.tgt_vocab.special_tokens["UNK"]] *= 0

        # Dropout 
        self.dropout = nn.Dropout(self.drop_prob)

        # LSTM forward and backward
        self.forward_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          dropout=self.drop_prob, batch_first=True)
        self.back_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          dropout=self.drop_prob, batch_first=True)

        # take all timestep
        self.output_layer_src = nn.Linear(in_features=hidden_dim, out_features=self.src_vocab.vocab_len, bias=False)
        self.output_layer_tgt = nn.Linear(in_features=hidden_dim, out_features=self.tgt_vocab.vocab_len, bias=False)
        
    def init_hidden(self, bs):
        """
        - initialize the hiddent states 
        """
        hidden = Variable(next(self.parameters()).data.new(self.num_layers, bs, self.hidden_dim))
        cell =  Variable(next(self.parameters()).data.new(self.num_layers, bs, self.hidden_dim))
        return hidden.zero_(), cell.zero_()

    def forward(self, lang, sentences, sent_lengths):
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
        
        batch_size, seq_len = sentences.size()
        
        # Check which embedding to use
        if lang == "en":
            # Copy the embedding of the <BOS>
            self.embedding_en.weight.data[2] = self.embedding_zh.weight.data[2]
            
            # Copy the linear mapping weight of the <EOS> 
            self.output_layer_en.weight.data[3] = self.output_layer_zh.weight.data[3]
            
            # Get the embedding 
            # EMBEDS: (batch_size, number of sequences, embedding_dim)
            embeds = self.embedding_en(sentences)
            
        elif lang == "zh":
            # Copy the embedding of the <BOS>
            self.embedding_zh.weight.data[2] = self.embedding_en.weight.data[2]
            
            # Copy the linear mapping weight of the <EOS> 
            self.output_layer_zh.weight.data[3] = self.output_layer_en.weight.data[3]
            
            # Get the embedding 
            embeds = self.embedding_zh(sentences) 
            
        

        # Initialize the hidden states 
        fh_t, fc_t = self.init_hidden(batch_size)
        bh_t, bc_t = self.init_hidden(batch_size)

        embeds_pad_forward = nn.utils.rnn.pack_padded_sequence(embeds, sent_lengths, batch_first=True)
    
        # Reverse the input and feed it to the lstm 
        for i, s in enumerate(embeds):
            reverse = s[:sent_lengths[i]]
            padding = s[sent_lengths[i]:]
            
            # reverse the sentence (only non padding part)
            reverse = torch.flip(reverse, [0])

            embeds[i] = torch.cat((reverse, padding),0)

            # Swap back the BOS and EOS
            embeds[i][0], embeds[i][sent_lengths[i]-1] = embeds[i][sent_lengths[i]-1], embeds[i][0]

        embeds_pad_backward = nn.utils.rnn.pack_padded_sequence(embeds, sent_lengths, batch_first=True)

        """
        inputs:
            input : (batch, seq_len, embedding_dim)
            h_0 : (num_layers, batch, hidden_size)
            c_0 : (num_layers, batch, hidden_size)

        returns:
            output : (batch, seq_len, embedding_dim) 
            h_n : (num_layers, batch, hidden_size)
            c_n : (num_layers, batch, hidden_size)
        """
        # Forward
        forward_lstm_out, (fh_t, fc_t)  = self.forward_lstm(embeds_pad_forward, (fh_t, fc_t))
        # Backward
        backward_lstm_out, (bh_t, bc_t) = self.back_lstm(embeds_pad_backward,(bh_t,bc_t))
        
        forward_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(forward_lstm_out, batch_first=True, total_length=seq_len)
        backward_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(backward_lstm_out, batch_first=True, total_length=seq_len)

        # Change the shape to (batch_size * seq_len, hidden_dim) 
        forward_lstm_out = forward_lstm_out.contiguous()
        forward_lstm_out = forward_lstm_out.view(-1, forward_lstm_out.shape[2])

        backward_lstm_out = backward_lstm_out.contiguous()
        backward_lstm_out = backward_lstm_out.view(-1, backward_lstm_out.shape[2])

        # Dropout 
        forward_lstm_out = self.dropout(forward_lstm_out)
        backward_lstm_out = self.dropout(backward_lstm_out)

        # Check which linear mapping to use
        output_layer = self.output_layer_en if lang == "en" else self.output_layer_zh

        forward_output =  output_layer(forward_lstm_out)
        backward_output = output_layer(backward_lstm_out)

        # shape = (batch_size * seq_len, vocab_size)
        forward_output, backward_output = F.log_softmax(forward_output, dim=1), F.log_softmax(backward_output, dim=1)

        # shape = (batch_size, seq_len, vocab_size)
        # return forward_output.view(batch_size, seq_len, -1), backward_output.view(batch_size, seq_len, -1)
        
        return forward_output, backward_output

