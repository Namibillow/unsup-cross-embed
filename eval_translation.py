import argparse
from pathlib import Path
from evaluation.word_translation import 

# Perform Bilingual lexicon extraction
def cslc():
    pass 

def mean_avg_precision():
    pass 



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