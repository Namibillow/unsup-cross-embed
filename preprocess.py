import argparse
from collections import namedtuple 

from utils.preprocess_utils import Universal, Japanese

"""
Handles text processing 
"""

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
        "-SENT_LEN",
        type=int,
        default=50000,
        help="sentences sizes"
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

    Data = namedtuple("Data","language_name file_path min_freq max_words num_sent save_path") 
    
    data = [ Data(language_name=lang, file_path=args.FILE_PATH[ind], min_freq=args.MIN_FREQ[ind], \
        max_words=args.MAX_WORDS, num_sent=args.SENT_LEN, save_path=args.SAVE_PATH+"/"+lang) for ind, lang in enumerate(args.LANG)]

    for d in data:
        if language_name == "jpn":
            language = Japanese(*d)
        else:
            language = Universal(*d)

        corpus = language.read_corpus()
        tokenized_corpus = language.tokenize(corpus)

        # later tokenized_corpus needs to be saved in a file
        language.save_data()

    #############################
    # Example run: 
    # python3 preprocess.py -LANG en sp -FILE_PATH data/en data/sp -MIN_FREQ 4 4 -MAX_WORDS 200 -SAVE_PATH data/processed_data