from collections import Counter 
from pathlib import Path
import random 
import re 

from sacremoses import MosesTokenizer
import MeCab

class Universal:
    def __init__(self, lang, file_path, min_freq, max_words, num_sent, save_path):
        """
        input:
            lang: 
            file_path:
            min_freq:
            max_words:
            num_sent:
            save_path:
        """
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
                - total sentences is less or equal to the num_sent specified
        """
        count = 0
        with file_path.open(self.file_path) as file:
            sentences = [sentence.strip() for sentence in file]
        
        # shuffle the list to randomize
        random.seed(seed)
        random.shuffle(sentences)
        
        return sentences[:self.num_sent]

    def save_data(self):
        """
        """
        num_sent //= 1000
        # example: en_50K_info.txt
        info_file_name = self.lang + "_" + num_sent + "K_info.txt"
        info_file_path = self.save_path / info_file_name

        with info_file_path.open(mode="w") as f:
            f.write(f"Language: {self.lang} \n")
            f.write(f"Original file path: {self.file_path}")
            f.write(f"Number of sentences: {self.num_sent}")
        
        # example: en.vocab_dict
        vocab_file_name = self.lang + "_" + num_sent + "K.vocab_dict"
        vocab_file_path = self.save_path / vocab_file_name

        # example: en_processed.txt 
        processed_file_name = self.lang + "_" + num_sent + "K_processed.txt"
        processed_file_path = self.save_path / processed_file_name

    def text_process(self, senntence):
        """
        """
        # Remove URLs
        sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',sentence)

        # Remove Emails 
        sentence = re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", "", sentence)

        # Lower Sentence
        sentence = sentence.lower()

        sentence = remove_unicode(sentence)

        return sentence 

    def tokenize(self, corpus):
        """
        """
        mt = MosesTokenizer(lang=self.lang)
        tokenized_corpus = []
        for sentence in corpus:
            sentence = self.text_process(sentence)
            tokenized_sent = mt.tokenize(sentence)
            tokenized_corpus.append(tokenized_sent)

        return tokenized_corpus

class Japanese(Universal):
    def tokenize(self, corpus):
        self.tagger = MeCab.Tagger()
    