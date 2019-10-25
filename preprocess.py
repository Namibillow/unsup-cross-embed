from utils.build_vocabs import Dictionary
from utils.preprocess_utils import Universal, Japanese, Thai, Korean

import argparse
from collections import namedtuple 

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
        default=70000,
        help="maximum count of vocabulary"
    )

    parser.add_argument(
        "-SAVE_PATH",
        type=str,
        required=True,
        help="path to save the preprocessed data"
    )

    args = parser.parse_args()

    assert len(args.LANG) == len(args.FILE_PATH) == len(args.MIN_FREQ), "given number of languages, file_pathes, min_freq must be equal"

    Data = namedtuple("Data","lang file_path min_freq max_words num_sent save_path") 
    
    data = [ Data(lang=lang, file_path=args.FILE_PATH[ind], min_freq=args.MIN_FREQ[ind], \
        max_words=args.MAX_WORDS, num_sent=args.SENT_LEN, save_path=args.SAVE_PATH+"/"+lang) for ind, lang in enumerate(args.LANG)]

    print("*"*70)
    print(f"Start processing the text... ")

    for d in data:
        
        if d.lang == "ja": # japanese
            language = Japanese(*d)
            tokenizer = None
        elif d.lang == "th": # thai
            language = Thai(*d)
            tokenizer = None 
        elif d.lang == "ko":
            language = Korean(*d)
            tokenizer = None 
        else:
            language = Universal(*d)
            # If language is Tamil or Turkish
            tokenizer = "polyglot" if d.lang in ["ta","tr"] else "moses"
                

        corpus = language.read_corpus()

        tokenized_corpus = language.tokenize(corpus, tokenizer)
        
        lang_dictionary = Dictionary(d.lang, d.max_words, d.min_freq, tokenized_corpus)

        lang_dictionary.build_vocab_dict()
        lang_dictionary.vectorize(tokenized_corpus)

        # later tokenized_corpus needs to be saved in a file
        language.save_data(tokenized_corpus, lang_dictionary.dataset, lang_dictionary.vocabulary)

        print("*"*70)

    print("Successfully processed the text.")

    
    #############################
    # Example run: 
    # python3 preprocess.py -LANG en sp -FILE_PATH data/en data/sp -MIN_FREQ 4 4 -MAX_WORDS 200 -SAVE_PATH data/processed_data
