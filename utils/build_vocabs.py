from collections import defaultdict, Counter, OrderedDict
from itertools import chain, dropwhile

import numpy as np
import pandas as pd 

"""
builds vocabulary 
"""

PAD, UNK, BOS_FWD, BOS_BWD, EOS = 0,1,2,3,4

class Dictionary:
    def __init__(self, lang,  max_vocab, min_freq, corpus):
        self.lang = lang 
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.corpus = corpus

        self.dataset = Dataset()
        self.vocabulary = Vocabulary()

    def build_vocab_dict(self):
        """
        - 

        """

        word2index = OrderedDict()
        index2word = OrderedDict()

        # Predefine for special tokens
        word2index['<PAD>'] = PAD
        word2index["<UNK>"] = UNK
        word2index["<BOS_FWD>"] = BOS_FWD
        word2index["<BOS_BWD>"] = BOS_BWD
        word2index["<EOS>"] = EOS

        index2word[PAD] = "<PAD>"
        index2word[UNK] = "<UNK>"
        index2word[BOS_FWD] = "<BOS_FWD>"
        index2word[BOS_BWD] = "<BOS_BWD>"
        index2word[EOS] = "<EOS>"

        # Count the frequency and sort them 
        wordCounter = Counter(chain.from_iterable(self.corpus))
        wordFreq = OrderedDict(wordCounter.most_common())

        # Remove objects whose counts are less than threshold in counter 
        words = np.array(list(wordFreq.keys()))
        freq = np.array(list(wordFreq.values()))

        idx = freq >= self.min_freq
        vocab = words[idx].tolist()

        print(f"Vocabulary count: {len(vocab)}/{self.max_vocab}")

        vocab = vocab[:self.max_vocab+1]
                
        index = len(word2index)

        for word in vocab:
            word2index[word] = index
            index2word[index] = word
            index+=1

        print(f"Vocaburaly length for {self.lang} including special tokens is {len(word2index)}")

        special_tokens = {"PAD": PAD, "UNK": UNK, "BOS_FWD": BOS_FWD, "BOS_BWD": BOS_BWD, "EOS": EOS}
        
        vocab_len = len(word2index)

        # update
        self.vocabulary.update(word2index, index2word, vocab_len, special_tokens)
    
    
    def sentence2idxs(self, sentence):
        idxs = [self.vocabulary.word2index[word] if word in self.vocabulary.word2index else UNK for word in sentence]

        return idxs

    def vectorize(self, tokenized_sentences):
        """
        - returns vectorized (converts from words of lists to numbers of lists)
        """
        ids = [self.sentence2idxs(sentence) for sentence in tokenized_sentences]
        lengths = [len(s) for s in ids]

        # update 
        self.dataset.update(ids, tokenized_sentences, lengths)


class Dataset():
    def __init__(self):
        """
        variables:
            vectorized_corpus:
            tokenized_corpus:
            length: a list of integers where each number represents length of the sentences
        """
        self.vectorized_corpus = []
        self.tokenized_corpus = []
        self.length = []

    def update(self ,vectorized_corpus, tokenized_corpus, length):
        self.vectorized_corpus = vectorized_corpus
        self.tokenized_corpus = tokenized_corpus
        self.length = length 


class Vocabulary():
    def __init__(self):
        """
        variables:
            word2index:
            index2word:
            special_tokens:
            vocab_len:
        """
        self.word2index = {}
        self.index2word = {}
        self.special_tokens = {}
        self.vocab_len = 0

    def update(self, word2index, index2word, vocab_len, special_tokens):
        self.word2index = word2index
        self.index2word = index2word
        self.special_tokens = special_tokens
        self.vocab_len = vocab_len 