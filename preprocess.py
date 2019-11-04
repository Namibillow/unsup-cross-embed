from utils.build_vocabs import Dictionary
from utils.preprocess_utils import Universal, Japanese, Thai, Korean, save_data_only

import argparse
from collections import namedtuple 
from pathlib import Path

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

    parser.add_argument(
        "-BPE",
        action='store_true',
        help="process already processed bpe applied text"
    )

    parser.add_argument(
        "-VOCAB",
        type=str,
        nargs="+",
        help="path to the vocab file"
    )

    parser.add_argument(
        "-PREFIX",
        type=str,
        nargs="+",
        help="Prefix of the data files to be saved"
    )

    args = parser.parse_args()

    if args.BPE:
        assert args.VOCAB, "vocab file is missing"
        assert args.FILE_PATH, "data is missing"

        print("Tokenized text and vocabulary are provided")

        Data= namedtuple("Data", "lang file_path vocab_file save_path prefix")

        data = [ Data(lang=lang, file_path=Path(args.FILE_PATH[ind]), vocab_file=Path(args.VOCAB[ind]), save_path=Path(args.SAVE_PATH+"/"+lang), prefix=args.PREFIX[ind]) for ind, lang in enumerate(args.LANG)]

        print("*"*70)
        print(f"Start processing the text... ")

        for d in data:
            print(f"-- procesisng {d.lang}: {d.file_path} --")
            with d.file_path.open(mode="r") as f:
                sentences = [sentence.strip() for sentence in f]
        
            tokenized_corpus = [sent.split() for sent in sentences]

            with d.vocab_file.open(mode="r") as f:
                vocabs = [v.strip() for v in f]

            lang_dictionary = Dictionary(d.lang, args.MAX_WORDS, None, tokenized_corpus)

            lang_dictionary.build_vocab_dict(vocabs)
            lang_dictionary.vectorize(tokenized_corpus)

            # later tokenized_corpus needs to be saved in a file
            save_data_only(d.save_path, d.prefix, lang_dictionary.dataset, lang_dictionary.vocabulary)

    else:

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
