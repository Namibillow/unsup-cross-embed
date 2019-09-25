from collections import Counter 
from pathlib import Path
import pickle
import random 
import re 

import regex

# Tokenizers
import MeCab # Japanese 
from polyglot.text import Text # Tamil, Turkish
from sacremoses import MosesTokenizer # rest of languages

"""
Utility class for preprocessing text
"""

class Universal:
    def __init__(self, lang, file_path, min_freq, max_words, num_sent, save_path):
        self.lang = lang 
        self.file_path = Path(file_path)
        self.min_freq = min_freq
        self.max_words = max_words
        self.num_sent = num_sent
        self.save_path = Path(save_path)

        self.save_path.mkdir(parents=True, exist_ok=True)

        print(f"building a dictionary for {self.lang}...")



    def read_corpus(self, seed=22):
        """
        - read corpus from a given file path 

        return:
            sentences: a list of sentences read from a txt file. 
        """
        with self.file_path.open(mode="r") as file:
            sentences = [sentence.strip() for sentence in file]
        
        # shuffle the list to randomize
        random.seed(seed)
        random.shuffle(sentences)
        
        print(f"{self.lang} has total of {len(sentences)}/{self.num_sent} sentences")

        
        return sentences[:self.num_sent]

    def save_data(self, tokenized_corpus, dataset, vocabulary):
        """
        - save given data to a specified path 

        input:
            tokenized_corpus: a list of lists of tokenized corpus
            dataset: instance of Dataset obj
            vocabulary: instance of Vocab obj
        """
        num_sent = str(self.num_sent // 1000)

        # example: en_50K_info.txt
        info_file_name = self.lang + "_" + num_sent + "K_info.txt"
        info_file_path = self.save_path / info_file_name

        print(f"1/4: Saving {info_file_name}")
        with info_file_path.open(mode="w") as f:
            f.write(f"Language: {self.lang} \n")
            f.write(f"Original file path: {self.file_path} \n")
            f.write(f"Number of sentences: {len(tokenized_corpus)}\n")
            f.write(f"Total Vocabulary: {(vocabulary.vocab_len)}\n")
            f.write(f"Minimum word frequency was set to {self.min_freq}\n")
            f.write(f"Maximum word count was set to {self.max_words}\n")
        
        # example: en_50K.vocab_dict
        vocab_file_name = self.lang + "_" + num_sent + "K.vocab_dict"
        vocab_file_path = self.save_path / vocab_file_name

        print(f"2/4: Saving {vocab_file_name}")
        with vocab_file_path.open(mode="wb") as f:
            pickle.dump(vocabulary,f)

        # example: en_50K_processed.txt 
        processed_file_name = self.lang + "_" + num_sent + "K_processed.txt"
        processed_file_path = self.save_path / processed_file_name

        print(f"3/4: Saving {processed_file_name}")
        with processed_file_path.open(mode="w") as f:
            for tokenized_sent in tokenized_corpus:
                f.write(" ".join(tokenized_sent))
                f.write("\n")

        # exampl: en_50K.dataset
        dataset_file_name = self.lang + "_" + num_sent + "K.dataset"
        dataset_file_path = self.save_path / dataset_file_name

        print(f"4/4: Saving {dataset_file_name}")
        with dataset_file_path.open(mode="wb") as f:
            pickle.dump(dataset,f)
            


    def text_preprocess(self, sentence):
        """
        - perform a text processing to clean text
        
        input:
            sentence: a str
        """
        # Remove URLs
        sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',sentence)

        # Remove Emails 
        sentence = re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", "", sentence)

        # Lower the sentence
        sentence = sentence.lower()

        if self.lang in ["ja"]: # japanese
            sentence = regex.sub(u"[^\r\n\p{Han}\p{Hiragana}\p{Katakana}ー。！？]", "", sentence)
        elif self.lang in ["ta"]: # tamile
            sentence = regex.sub(u"[^ \r\n\p{Tamil}.?!\-]", " ", sentence)
        elif self.lang in ["ru"]: # russian
            sentence = regex.sub(u"[^ \r\n\p{Cyrillic}.?!\-]", " ", sentence)
        else: # english, german, spanish, french, finnish, czech, turkish
            sentence = regex.sub(u"[^ \r\n\p{Latin}\-'‘’.?!]", " ", sentence)        
    
        # Common
        sentence = regex.sub("[ ]{2,}", " ", sentence) # Squeeze spaces.

        return sentence 

    def tokenize(self, corpus, tokenizer="moses"):
        """
        - tokenize sentences 

        input:
            corpus: list of sentences

        """
        tokenized_corpus = []
        if tokenizer=="moses":
            mt = MosesTokenizer(lang=self.lang)
            for sentence in corpus:
                sentence = self.text_preprocess(sentence)
                # return a list of tokenized words
                tokenized_sent = mt.tokenize(sentence)
                tokenized_corpus.append(tokenized_sent)
        elif tokenizer=="polyglot":
            for sentence in corpus:
                sentence = self.text_preprocess(sentence)
                text = Text(sentence, hint_language_code=self.lang)
                tokenized_corpus.append(list(text.words))

        return tokenized_corpus

class Japanese(Universal):
    """ tokenize Japanese corpus """
    def tokenize(self, corpus, tokenizer=None):

        mt = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

        tokenized_corpus = []
        for sentence in corpus:
            sentence = self.text_preprocess(sentence)
            
            res = mt.parseToNode(sentence)
            tokenized_sent = []
            while res:
                tokenized_sent.append(res.surface)
                res = res.next
            
            tokenized_corpus.append([x for x in tokenized_sent if x != ""])

        return tokenized_corpus
    
