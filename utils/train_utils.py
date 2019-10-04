from pathlib import Path 
import pickle
import random
import numpy as np 

def load_data(data_path, data_prefix):
	"""
		- Load the data from given data_path and data_prefix
	"""

	file_path = Path(data_path)

	data_path = data_prefix + ".dataset"
	vocab_path = data_prefix + ".vocab_dict"

	data_path = (file_path / data_path).open(mode="rb")
	dataset = pickle.load(data_path)

	vocab_path = (file_path / vocab_path).open(mode="rb")
	vocab = pickle.load(vocab_path)

	# for i in range(5):

	# 	print(dataset.vectorized_corpus[i])
	# 	print(dataset.tokenized_corpus[i])
	# 	assert dataset.length[i] == len(dataset.tokenized_corpus[i])

	# print(f"Vocabulary length: {vocab.vocab_len}")
	# print(vocab.special_tokens)
	return dataset, vocab 

def get_minibatches():
	pass

def pad_sequence():
	pass

def pad_special_tokens():
	pass # add <BOS> <EOS> 


def oversampling(src, tgt):
	"""
	- Randomly repeat some minority samples and balance the number of samples between the dataset 
	"""
	max_sent, less_sent = (src, tgt) if len(src.vectorized_corpus) > len(tgt.vectorized_corpus) else (tgt, src)

	max_num_sent, less_num_sent = len(max_sent.vectorized_corpus), len(less_sent.vectorized_corpus)

	logger.debug("Max sentence:  %d", max_num_sent)
	logger.debug("Less sentence: %d", less_num_sent)

	repeat = max_num_sent // less_num_sent
	remainder = max_num_sent % less_num_sent
	random_idx = random.sample(range(less_num_sent), remainder)

	less_sent.vectorized_corpus = augment_data(less_sent.vectorized_corpus, repeat, remainder, random_idx)
	less_sent.length = augment_data(less_sent.length, rep, ramdom_idx)

	return (max_sent, less_sent) if max_sent == src else (less_sent, max_sent)


def augment_data(self,lines, rep, ramdom_idx):
        out = lines.copy()
        out = out * rep # repeat 
        out += [lines[idx] for idx in ramdom_idx] # add remainder
        return out

def save_embedding():
	pass

def mini_batchfy():
	pass