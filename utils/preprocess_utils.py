from collections import Counter 
from pathlib import Path
import re 

from sacremoses import MosesTokenizer
import MeCab

class Universal:
    def __init__(self, lang, file_path, min_freq, max_words, save_path):
        self.lang = lang 
        self.file_path = Path(file_path)
        self.min_freq = min_freq
        self.max_words = max_words
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        print(f"building dictionary for {self.lang}...")

    def read_corpus(self):
        with file_path.open(self.file_path) as file:
            # Change split if the dataset is not indexed per sentence
            sentences = [sentence.split("\t",1)[1].strip() for sentence in file]
 
    def remove_tags(self):
        pass

    def remove_whitespace(self):
        pass

    def remove_unicode(self):
        pass

    def remove_integers(self):
        pass

    def lowercase(self, text):
        pass

    def save_data(self):
        pass

class English(Universal):
    def tokenize(self):
        self.mt = MosesTokenizer(lnag=en)

class German(Universal):
    def tokenize(self):
        mt = MosesTokenizer(lang=de)

class French(Universal):
    def tokenize(self):
        mt = MosesTokenizer(lang=fr) 

class Czech(Unversal):
    def tokenize(self):
        mt = MosesTokenizer(lang=cs)

class Finnish(Universal):
    def tokenize(self):
        mt = MosesTokenizer(lang=fi)

class Japanese(Universal):
    def tokenize(self):
        self.tagger = MeCab.Tagger()
    


class Russian(Universal):
    def tokenize(self):
        mt = MosesTokenizer(lang=ru)

class Spanish(Universal):
    def tokenize(self):
        mt = MosesTokenizer(lang=es) 