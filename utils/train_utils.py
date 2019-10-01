from pathlib import Path 
import pickle
import random
import numpy as np 

def load_data(data_path, data_lang, data_prefix):
	"""
		- Load the data from given data_path and data_prefix
	"""

	file_path = Path(data_path)

	data_path = data_prefix + ".dataset"
	vocab_path = data_prefix + ".vocab_dict"

	data_path = (file_path / data_lang / data_path).open(mode="rb")
	dataset = pickle.load(data_path)

	vocab_path = (file_path / data_lang / vocab_path).open(mode="rb")
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


def oversampling(datasets):
	"""
	- Randomly repeat some minority samples and balance the number of samples between the dataset 
	"""

	largest_corpus = np.argmax([len(datasets[i].tokenized_corpus) for i in range(len(datasets))])
	max_sentence_num = len(datasets[largest_corpus].tokenized_corpus)

	for i in range(len(datasets)):
		sentence_num = len(datasets[i].tokenized_corpus)
		if max_sentence_num != sentence_num:
			print("Perform oversampling")
			print("max_sentence_num: ", max_sentence_num)
			print("src lang" + str(i) + ": ", sentence_num)
			rep = max_sentence_num // sentence_num
			remainder = max_sentence_num % sentence_num
			ramdom_idx = random.sample(range(sentence_num), remainder)
			datasets[i].tokenized_corpus = augment_data(dataset[i].tokenized_corpus, rep, ramdom_idx)
			datasets[i].lengths = augment_data(dataset[i].lengths, rep, ramdom_idx)

		dataset[i].lengths = np.array(dataset[i].lengths) # list -> numpy

	return datasets[0], datasets[1]

def augment_data(self,lines, rep, ramdom_idx):
        out = lines.copy()
        out = out * rep # repeat 
        out += [lines[idx] for idx in ramdom_idx] # add remainder
        return out

def save_embedding():
	pass

def mini_batchfy():
	pass