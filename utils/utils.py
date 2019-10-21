from collections import OrderedDict, defaultdict
import json 
from pathlib simport Path 
import random

import numpy as np

"""
stores bunch of small utility functions 
"""

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def load_emb(emb_file, dtype='float'):
    """
    Load the embedding 

    returns:
        words: (list) contains word
        matrix: (np.array) containes vector for each word [count, dim]
    """
    embfile = open(emb_file, errors='surrogateescape')
    
    header = embfile.readline().split(" ")
    count = int(header[0])
    dim = int(header[1])

    words = []
    matrix = np.empty((count, dim), dtype=dtype)

    for i in range(count):
        word, vec = embfile.readline().split(" ", 1)

        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
    
    return (words, matrix)
    
def load_dict(dict_file, v_limit, src_word2ind, tgt_word2ind):
    """
    - Read dictionary and compute coverage
    
    return: 
        src: (list) a list of indexes of src words
        src2tgt (dict) key is src word index, value is a set of tgt word index
    """

    src2trg = defaultdict(set)
    oov = set()
    vocab = set()
    reached = 0

    with open(dict_file, errors='surrogateescape') as f: 
        lines = list(f)
    
    random.seed(30)
    random.shuffle(lines)

    for line in lines:
        src, trg = line.split()
        if reached == limit:
            break
        try:
            src_ind = src_word2ind[src]
            trg_ind = tgt_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
            reached+=1
        except KeyError:
            oov.add(src)
    src = list(src2trg.keys())

    print(f"{len(src2trg)}/{limit} pairs will be considered")

    return src, src2trg