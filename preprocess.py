import argparse
from collections import namedtuple 

from utils.preprocess_utils import *

"""
Handles text processing 
"""

languages = {"en": English, "deu": German, "fra": French, "rus": Russian, "ces": Czech, "fin": Finnish, "jpn": Japanese, "spa": Spanish}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-LANG",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help='name of the language'
    )
    
    parser.add_argument(
        "-FILE_PATH",
        type=str,
        nargs='+',
        required=True,
        help="file pathes to the data"
    )

    parser.add_argument(
        "-MIN_FREQ",
        type=int,
        nargs="+",
        help="minimum frequency of vocabulary words"
    )

    parser.add_argument(
        "-MAX_WORDS",
        type=int,
        default=200000,
        help="maximum count of vocabulary"
    )

    parser.add_argument(
        "-SAVE_PATH",
        type=str,
        required=True,
        help="path to save the preprocessed data"
    )

    args = parser.parse_args()

    assert len(args.LANG) == len(args.FILE_PATH) == len(args.MIN_FREQ)

    Data = namedtuple("Data","language_name file_path min_freq max_words save_path") 
    
    data = [ Data(language_name=lang, file_path=args.FILE_PATH[ind], min_freq=args.MIN_FREQ[ind], \
        max_words=args.MAX_WORDS, save_path=args.SAVE_PATH+"/"+lang) for ind, lang in enumerate(args.LANG)]

    for d in data:
        languages[d.language_name](*d)


    #############################\
    # Example run: 
    # python3 preprocess.py -LANG en sp -FILE_PATH data/hello data/hola -MIN_FREQ 4 4 -MAX_WORDS 200 -SAVE_PATH new