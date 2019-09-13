from collections import Counter 
from pathlib import Path
import random 
import re 

from sacremoses import MosesTokenizer
import MeCab

class Universal:
    def __init__(self, lang, file_path, min_freq, max_words, num_sent, save_path):
        self.lang = lang 
        self.file_path = Path(file_path)
        self.min_freq = min_freq
        self.max_words = max_words
        self.num_sent = num_sent
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        print(f"building dictionary for {self.lang}...")

    def read_corpus(self, seed=22):
        """
        output:
            sentences 
                - a list of sentences read from a txt file
        """
        count = 0
        with file_path.open(self.file_path) as file:
            sentences = [sentence.strip() for sentence in file]
        
        # shuffle the list to randomize
        random.seed(seed)
        random.shuffle(sentences)
        
        return sentences[:self.num_sent]

    def remove_unicode(self, sentence):
        pass

    def remove_integers(self, sentence):
        pass

    def save_data(self):
        info_file_name = self.lang + "_info.txt"
        info_file_path = self.save_path / info_file_name

        with info_file_path.open(mode="w") as f:
            f.write(f"Language: {self.lang} \n")
            f.write(f"Original file path: {self.file_path}")
        
        vocab_file_name = self.lang + ".vocab_dict"
        vocab_file_path = self.save_path / vocab_file_name

    def text_process(self, senntence):
        # Remove URLs
        sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',sentence)

        # Remove Emails 
        sentence = re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", "", sentence)

        # Lower Sentence
        sentence = sentence.lower()

        sentence = remove_unicode(sentence)
        
        sentence = remove_integers(sentence)

        return sentence 

    def tokenize(self, corpus):
        mt = MosesTokenizer(lang=self.lang)

        for sentence in corpus:
            sentence = 
            tokenized = mt.tokenize(sentence, return_str=True)

class Japanese(Universal):
    def tokenize(self, corpus):
        self.tagger = MeCab.Tagger()
    