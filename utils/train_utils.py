from pathlib import Path 
import pickle

def load_data(data_path, data_lang, data_prefix):
	"""
		- Load the data from given data_path and data_prefix
	"""
	print(f"Loading {data_prefix} dataset...")

	file_path = Path(data_path)

	data_path = data_prefix + ".dataset"
	vocab_path = data_prefix + ".vocab_dict"

	data_path = (file_path / data_lang / data_path).open(mode="rb")
	dataset = pickle.load(data_path)

	vocab_path = (file_path / data_lang / vocab_path).open(mode="rb")
	vocab = pickle.load(vocab_path)

	print("Loading completed.")

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