from utils.utils import load_dict, load_emb

import argparse
from collections import defaultdict
from pathlib import Path
import datetime

import numpy as np

def calc_score(translation, src_words, tgt_words):

    # translation => dict of list: key src words, value a list of indexes
    num_src_words = len(translation)
    recip_rank = 0
    num_correct = 0
    correct = []

    for i in range(num_src_words):
        top_k_words_ind = translation[i]

        for j, ind in enumerate(top_k_words_ind):
            if i == ind:
                correct.append((i, ind, j+1))
                recip_rank += 1 / (j+1)
                num_correct+=1
            
    prec = recip_rank / num_src_words
    num_correct/= num_src_words
    
    return (prec, num_correct), correct 

def length_normalize(matrix):
    """
    -  inplace normalization
    """
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, np.newaxis]

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train for CLWE")
    parser.add_argument(
        "-d", 
        "--dict", 
        type=str,
        default=None,
        help="path where the dictionary is stored"
        )

    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="number of vocaburay to take in account for evaluation "
    )

    parser.add_argument( 
        "--src_emb",
        type=str,
        default=None,
        required=True,
        help="directory that contains preprocessed source dataset"
    )

    parser.add_argument(
        "--tgt_emb",
        type=str,
        default=None,
        required=True,
        help="directory that contains preprocessed target dataset"
    )

    parser.add_argument(
        "-k", 
        "--num_ranks",
        type=int,
        default=10,
        help="a number of k ranks to be considered."
    )

    parser.add_argument(
        "-dir", 
        "--directory",
        type=str,
        default=None,
        help="directory that contains all the necessary data"
    )

    args = parser.parse_args()

    directory = Path(args.directory)
    assert directory.is_dir(), "directory doesn't exist"

    # read embedding 
    print("Reading embeddings")
    src_word2emb = load_emb(directory, args.src_emb)
    tgt_word2emb = load_emb(directory, args.tgt_emb)

    # read dictionary 
    print("Reading dictionary")
    src_word_list, tgt_word_list = load_dict(args.dict)

    num_words = len(src_word_list) if len(src_word_list) > len(src_word2emb) else len(src_word2emb)
    if not args.limit:
        args.limit = num_words
    limit = args.limit if args.limit < num_words else num_words

    oov = 0
    reached = 0 

    src_covered_words = []
    tgt_covered_words = []

    src_covered_emb_list = []
    tgt_covered_emb_list = []

    num_dict_words = len(src_word_list)

    for i in range(num_dict_words):
        if reached == limit:
            break
        
        # Encode the word here
        if (src_word_list[i] in src_word2emb and tgt_word_list[i] in tgt_word2emb):
            src_emb = src_word2emb[src_word_list[i]]
            tgt_emb = tgt_word2emb[tgt_word_list[i]]

            src_covered_emb_list.append(src_emb)
            tgt_covered_emb_list.append(tgt_emb)

            src_covered_words.append(src_word_list[i])
            tgt_covered_words.append(tgt_word_list[i])
            
            reached+=1
        else:
            oov+=1

    print(f"{reached}/{limit} pairs will be considered")
    
    assert len(src_covered_emb_list) == len(tgt_covered_emb_list)

    src_covered_emb_list = np.array(src_covered_emb_list)
    tgt_covered_emb_list = np.array(tgt_covered_emb_list)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    length_normalize(src_covered_emb_list)
    length_normalize(tgt_covered_emb_list)

    print("Performing Cross-domain similarity local scaling ")
    ### knn_sim_bwd => [src_words_limit, dim] * [dim, tgt_words_limit] => [src_words_limit, tgt_words_limit]
    knn_sim_bwd = np.matmul(src_covered_emb_list, tgt_covered_emb_list.T)
    ### cossim_sorted => [src_words_limit, tgt_words_limit]
    cossim_sorted = np.sort(knn_sim_bwd, axis=1)[:,::-1]
    ### cossim_sorted_T => [tgt_words_limit, src_words_limit]
    cossim_sorted_T = np.sort(knn_sim_bwd.T, axis=1)[:,::-1]

    K = 10 # Default
    ### rT => [src_word_limit, 1]
    rT = np.array([np.mean(x[0:K]) for x in cossim_sorted]).reshape(-1, 1)
    ### rS => [1, tgt_word_limit]
    rS = np.array([np.mean(x[0:K]) for x in cossim_sorted_T]).reshape(1, -1) 
    ### similarities => [src_words_limit, tgt_words_limit]
    similarities = 2* knn_sim_bwd - np.broadcast_to(rT, knn_sim_bwd.shape) - np.broadcast_to(rS, knn_sim_bwd.shape)
    ### nn => [src_words_limit, num_ranks]
    nn = similarities.argsort(axis=1)[:,-args.num_ranks:][:,::-1]

    acc, correct = calc_score(nn, src_covered_words, tgt_covered_words)

    for (s, t, rank) in correct[:10]:
        print(f"RANK: {rank}  | SRC: {src_covered_words[s]} => TGT: {tgt_covered_words[t]}")
    
    print(f"{len(correct)}/{reached} words were correct.")


    save = directory / "evaluated.txt"

    append_write = 'a' if save.is_file() else 'w'

    with save.open(mode=append_write) as f:
        f.write(f"-- {(datetime.datetime.now()).strftime('%Y-%m-%d %H:%M')} -- \n")
        f.write(f"Dictinary pairs used: {reached}/{len(src_word_list)}\n")
        f.write(f"Accuracy with MAP@{args.num_ranks}: {acc[0]} \n")
        f.write(f"Accuracy with P@{args.num_ranks}: {acc[1]} \n")
    
    print(f"Source: {args.src_emb}")
    print(f"Target: {args.tgt_emb}")
    print(f"Accuracy with MAP@{args.num_ranks}: {acc[0]}")
    print(f"Accuracy with P@{args.num_ranks}: {acc[1]}")

    # Sample run
    # python3 eval_translate.py -d data/eval_dict/ -l 1000 -src_emb ja_50K.vec -tgt_emb en_1000K.vec
    # -k 10 -dir results/low_resource/cs_50K_en_1000K/embed/Bi_LSTM/1017_105806/ 