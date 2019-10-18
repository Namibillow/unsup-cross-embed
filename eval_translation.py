import argparse
from pathlib import Path
import numpy as np

# Perform Bilingual lexicon extraction
def cslc():
    knn_sim_bwd = xp.zeros(z.shape[0])
    for i in range(0, z.shape[0], BATCH_SIZE):
        j = min(i + BATCH_SIZE, z.shape[0])
        knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
    for i in range(0, len(src), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(src))
        similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
        # Sort instead of argmax and get top K 
        nn = similarities.argmax(axis=1).tolist()
        for k in range(j-i):
            translation[src[i+k]] = nn[k]  # store top K words

def mean_reciprocal_rank()):
    top_k = 0 

    src_words = len(translation)

    for i in range(src_words):
        top_k_word_ind = translation[i]

        # get the index of the translated word if exist else 0 
        # get average precision. Taken in account of the precision 
        # so for eg if p@1 got correct (index 0) then precision is 1 
        # if p@4 was correct (index 3) than 1/4 precision 0.25 
        # sum all the precision for words and divide by src_words
    
    return sum of the precision / src_words 


     

def top_k_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    print(ans)
    print(ans / k)
    print((ans/k).shape)
    return ans / k


a = np.random.rand(16, 100)

top_k_mean(a, 10, inplace=True)
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
        default=[1,5,10],
        help="a list of k ranks to be considered."
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
    src, x = read_emb(args.src_emb)
    tgt, y = read_emb(args.tgt_emb)
    
    # read dictionary 
    src2tgt = read_dict(args.dict)

    # get save path 
    save_path = Path(args.save)

    cslc()

    mean_avg_precision()

    # write to a file or get logger 