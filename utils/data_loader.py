import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch 

class SentenceDataset(Dataset):
    def __init__(self, X, length, batch_size,special_tokens):
        self.input = X
        self.length = length # sort by length 
        self.batch_size = batch_size
        self.special_tokens = special_tokens

        self.fwd_input = []
        self.fwd_output = []
        self.bwd_input = []
        self.bwd_output = []

        self.sort_data_by_length()

        self.bacth_idx_list = self.generate_batch_idx()
        
        self.build_output() 

        self.to_tensor = lambda x: torch.LongTensor(x)

        assert len(self.fwd_input) == len(self.fwd_output) == len(self.bwd_input) == len(self.bwd_output), "length is wrong!"
    
    def __getitem__(self, index):
        fwd_inputs = self.to_tensor(self.fwd_input[index])
        fwd_outputs = self.to_tensor(self.fwd_output[index])
        bwd_inputs = self.to_tensor(self.bwd_input[index])
        bwd_outputs = self.to_tensor(self.bwd_output[index])
        # batch_lengths = self.length[self.bacth_idx_list[index]]

        return fwd_inputs, fwd_outputs, bwd_inputs, bwd_outputs
    
    def __len__(self):
        return len(self.fwd_input)
    
    def sort_data_by_length(self):
        idx = np.argsort(self.length)[::-1] # sort by descending order
        # apply sorted order
        self.input = np.array([self.input[i] for i in idx])
        self.length = self.length[idx]

    def generate_batch_idx(self):
        bacth_idx_list = []

        for i in range(0, len(self.length), self.batch_size):
            batch_idx = list(range(i, min(i + self.batch_size, len(self.length))))
            bacth_idx_list.append(batch_idx)

        return bacth_idx_list

    def build_output(self):
        """
        - creates output and also do padding and appending special tokens
        """
        for batch_idx in self.bacth_idx_list:
            sent_lengths = self.length[batch_idx]
            max_sent_length = max(sent_lengths)
            num_pad_len = max_sent_length - sent_lengths

            fwd_input_b = []
            fwd_output_b = []
            bwd_input_b = []
            bwd_output_b = []

            for i, index in enumerate(batch_idx):
                sentence = self.input[index]

                fwd_input_b.append([self.special_tokens["BOS_FWD"]] + sentence + num_pad_len[i] * [self.special_tokens["PAD"]])
                fwd_output_b.append(sentence + [self.special_tokens["EOS"]] + num_pad_len[i] * [self.special_tokens["PAD"]])
                bwd_input_b.append([self.special_tokens["BOS_BWD"]] + sentence[::-1] + num_pad_len[i] * [self.special_tokens["PAD"]])
                bwd_output_b.append(sentence[::-1] + [self.special_tokens["EOS"]] + num_pad_len[i] * [self.special_tokens["PAD"]])

            self.fwd_input.append(fwd_input_b)
            self.fwd_output.append(fwd_output_b)
            self.bwd_input.append(bwd_input_b)
            self.bwd_output.append(bwd_output_b)


def batchfy(dataset, batch_size, shuffle=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)