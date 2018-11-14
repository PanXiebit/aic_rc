#!/usr/bin/python
# coding:utf-8

#!/usr/bin/python
# coding:utf-8
import pickle
import numpy as np
from tqdm import tqdm
import os

def char_embedding(word2id, char2id, char_embedding, word_len=3, char_embed_size=200,
                   word_char_vec_path=None):
    word_char_vec = np.random.normal(0, 0.1, (len(word2id), char_embed_size))
    if not os.path.exists(word_char_vec_path):
        for i, (word, id) in tqdm(enumerate(word2id.items())):
            word = word[:word_len]
            if len(word) < word_len:
                word = list(word) + ["PAD"] * (word_len - len(word))
            # print(word, len(word))
            chars_id = [char2id[x] if x in char2id else char2id['UNK'] for x in word]
            word_vec = [np.reshape(char_embedding[charid], (1, char_embed_size))
                        for charid in chars_id] # list, [1,768]
            word_vec = np.concatenate(word_vec, axis=0)
            word_vec = np.sum(word_vec, axis=0)
            # print(word_vec.shape)
            word_char_vec[id] = word_vec
        print(word_char_vec.shape)
        np.save(word_char_vec_path, word_char_vec)
    else:
        word_char_vec = np.load(word_char_vec_path)
    return word_char_vec


if __name__ == "__main__":
    data_path = "/home/haixiao.liu/xiepan/project/aic_data/"
    word_char_vec_path = "/home/haixiao.liu/xiepan/project/aic_data/word_char_vec.npy"

    with open(data_path + 'word2id.pickle', 'rb') as f1:
        word2id = pickle.load(f1)

    char2id_path = "/home/haixiao.liu/xiepan/project/aic_data/char2id.pickle"
    with open(char2id_path, "rb") as f:
        char2id = pickle.load(f)
    # print(char2id["好"])

    charvec_npy = "/home/haixiao.liu/xiepan/project/aic_data/charvec_200.npy"
    char_wordvec = np.load(charvec_npy)
    # print(char_wordvec.shape)
    # print(char_wordvec[1000])
    # print(np.mean(char_wordvec, axis=1)[:10])
    # print(np.var(char_wordvec, axis=1)[:10])
    word_char_embedding = char_embedding(word2id, char2id, char_wordvec, word_char_vec_path=word_char_vec_path)
    print(word_char_embedding.shape)
