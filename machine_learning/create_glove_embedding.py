import bcolz
import pickle
import numpy as np

GLOVE_PATH = "glove"

words = []
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{GLOVE_PATH}/6B.50.dat', mode='w')

with open(f'{GLOVE_PATH}/glove.6B.50d.txt', 'rb') as file:
    for idx, line in enumerate(file):
        line_arr = line.decode().split()
        word = line_arr[0]
        words.append(word)
        word2idx[word] = idx
        word_vec = np.array(line_arr[1:]).astype(np.float)
        vectors.append(word_vec)

# Add an extra word for unknown vocab. It is the mean of all vectors
# https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
avg_vec = np.mean(vectors[1:].reshape((400001, 50)), axis=0)
unknown_word = "[unk]" # Note: "unk" and "<unk>" are actually words in GloVe

word2idx[unknown_word] = len(words)
words.append(unknown_word)
vectors.append(avg_vec)

# Save the results
vectors = bcolz.carray(vectors[1:].reshape((400002, 50)), rootdir=f'{GLOVE_PATH}/6B.50.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{GLOVE_PATH}/6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{GLOVE_PATH}/6B.50_idx.pkl', 'wb'))
