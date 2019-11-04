from collections import OrderedDict, defaultdict
import json 
from pathlib import Path 
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


def load_emb(directory, emb_file, dtype='float'):
    """
    Load the embedding 

    returns:
        word2emb: (dict) key -> word, value ->  embedding value
    """
    path = directory / emb_file

    assert path.is_file(), "file does not exist "

    embfile = open(path, errors='surrogateescape')
    
    header = embfile.readline().split(" ")
    count = int(header[0])
    dim = int(header[1])

    word2emb = dict()

    for i in range(count):
        try:
            word, vec = embfile.readline().split(" ", 1)
        except:
            print("Failed")

        word2emb[word] = np.fromstring(vec, sep=' ', dtype=dtype)
    
    return word2emb
    
def load_dict(dict_file):
    """
    - Read dictionary
    
    return: 
        
    """
    src_word_list = []
    tgt_word_list = [] 

    with open(dict_file, errors='surrogateescape') as f: 
        lines = list(f)
    
    # random.seed()
    # random.shuffle(lines)

    for line in lines:
        src, tgt = line.split()
        src_word_list.append(src)
        tgt_word_list.append(tgt)

    print(f"total of {len(lines)} pairs of words in the dictionary.")

    return src_word_list, tgt_word_list