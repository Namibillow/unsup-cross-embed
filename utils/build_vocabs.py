from collections import defaultdict, Counter, OrderedDict
from itertools import chain

import numpy as np
import pandas as pd 

"""
builds vocabulary 
"""

SPECIAL_TOKENS = {"BOS_FWD": 0, "BOS_BWD": 1, "EOS":2, "PAD":3, "UNK":4}

class Dictionary:
    def __init__(self, lang,  max_vocab, min_freq, corpus):
        """
        """
        self.lang = lang 
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.corpus = corpus

        self.vocab = []
        self.word2index = dict()
        self.index2word = dict()

    def build_vocab_dict(self):
        """
        """

        # Count the frequency and sort them 
        wordCounter = Counter(chain.from_iterable(self.corpus))
        wordFreq = OrderedDict(wordCounter.most_common())

        if(len(wordFreq) < self.max_vocab):
            print("Less vocabs")

        vocab = list(wordFreq.keys())[:self.max_vocab+1]
                

        # Predefine some 
        self.word2index['<PAD>'] = 0
        self.word2index["<UNK>"] = 1
        self.word2index["<BOS>"] = 2
        self.word2index["<EOS>"] = 3

        self.index2word[0] = "<PAD>"
        self.index2word[1] = "<UNK>"
        self.index2word[2] = "<BOS>"
        self.index2word[3] = "<EOS>"

        index = len(self.word2index)

        for word in vocab:
            self.word2index[word] = index
            self.index2word[index] = word
            index+=1

        print(f"Vocaburaly length for {self.lang} is {len(self.word2index)}")
    def sentence2ids(self):
        pass

    def senntences2id(self):
        pass

    def ids2sentence(self):
        pass

    def ids2sentences(self):
        pass

    def size(self):
        return len(self.id2word) - 1

    def vectorize(self):
        # get sentnce length also 
        pass


def pad