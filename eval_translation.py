from utils.utils import load_dict, load_emb

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

def mean_reciprocal_rank(translation):
    top_k = 0 
    # translation = dict of list: key src, value a list 
    src_words = len(translation)
    recip_rank = 0
    for i in range(src_words):
        top_k_words_ind = translation[i]

        for i, ind in enumerate(top_k_words_ind):
            if ind in src2tgt[i]:
                recip_rank + = 1 / i+1  
            
    avg = recip_rank / src_words

        # get the index of the translated word if exist else 0 
        # get average precision. Taken in account of the precision 
        # so for eg if p@1 got correct (index 0) then precision is 1 
        # if p@4 was correct (index 3) than 1/4 precision 0.25 
        # sum all the precision for words and divide by src_words
    
    return avg

def top_k_mean(m, k, inplace=False): 
    """
    sum up to top k closest words and take the average

    return:
        ans: [batch_size,]
    """
    # m => [batch_size, num_src_words]
    n = m.shape[0] # n => batch_size
    ans = np.zeros(n, dtype=m.dtype) # ans => [batch_size, ]
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n) # ind0 => [batch_size, ]
    ind1 = np.empty(n, dtype=int) # ind1 => [batch_size, ]
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum

    return ans / k

def length_normalize(matrix):
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
        default=1000,
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
        nparser='+',
        default=10,
        help="a number of k ranks to be considered."
    )

    parser.add_argument(
        "-s", 
        "--save",
        type=str,
        default=None,
        help="file path to save the output"
    )

    args = parser.parse_args()

    # read embedding 
    src_emb_words, src_emb = load_emb(args.src_emb)
    tgt_emb_words, tgt_emb = load_emb(args.tgt_emb)
    
    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    length_normalize(emb_x)
    length_normalize(emb_z)


    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_emb_words)}
    tgt_word2ind = {word: i for i, word in enumerate(tgt_emb_words)}

    v_limit = args.limit if args.limit < len(src_emb_words) else len(src_emb_words)

    # read dictionary 
    src, src2tgt = load_dict(args.dict, v_limit, src_word2ind, tgt_emb_words)

    translation = collections.defaultdict(int)
    batch_size = 1024 

    ### knn_sim_bwd => [num_tgt_words, ]
    knn_sim_bwd = np.zeros(tgt_emb.shape[0])
    for i in range(0, tgt_emb.shape[0], batch_size):
        j = min(i + batch_size, tgt_emb.shape[0])
        # tgt_emb[i:j].dot(src_emb.T) => [batch_size, dim] * [dim, num_src_words] => [batch_size, num_src_words]
        knn_sim_bwd[i:j] = top_k_mean(tgt_emb[i:j].dot(src_emb.T), k=args.num_ranks, inplace=True)
    
    for i in range(0, len(src), batch_size):
        j = min(i + batch_size, len(src))
        # src_emb[src[i:j]].dot(tgt_emb.T) => [batch_size, dim] * [dim, num_tgt_words] => [batch_size, num_tgt_words]
        # similarities => [batch_size, num_tgt_words]
        similarities = 2*src_emb[src[i:j]].dot(tgt_emb.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
        # Sort instead of argmax and get top K 
        # nn = similarities.argmax(axis=1).tolist()
        # nn => [batch_size, num_k]
        nn = similarities.argsort(axis=1)[:,-args.num_ranks:][:,::-1]
        for k in range(j-i):
            translation[src[i+k]] = nn[k].tolist()  # store top K tgt words

    mean_reciprocal_rank(translation)

    # get save path 
    save_path = Path(args.save)
