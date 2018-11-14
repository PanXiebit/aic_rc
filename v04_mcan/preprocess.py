#!/usr/bin/python
# coding:utf-8

import pickle
import json
from tqdm import tqdm
import jieba
import numpy as np
import os

def seg_line(line):
    return list(jieba.cut(line))


def seg_data(path):
    print ('\nstart process {}'.format(path))
    data = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            dic = json.loads(line, encoding='utf-8')
            if len(dic["alternatives"].split("|")) != 3:
                continue
            question = dic['query']
            doc = dic['passage']
            alternatives = dic['alternatives']
            data.append([seg_line(question), seg_line(doc), alternatives.split('|'), dic['query_id']])
    return data


def build_word_count(data):
    wordCount = {}

    def add_count(lst):
        for word in lst:
            if word not in wordCount:
                wordCount[word] = 0
            wordCount[word] += 1

    for one in data:
        [add_count(x) for x in one[0:3]]
    print ('word type size ', len(wordCount))
    return wordCount


def build_word2id(wordCount, threshold=10):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    for word in wordCount:
        if wordCount[word] >= threshold:
            if word not in word2id:
                word2id[word] = len(word2id)
        else:
            chars = list(word)
            for char in chars:
                if char not in word2id:
                    word2id[char] = len(word2id)
    print ('processed word size ', len(word2id))
    return word2id


def transform_data_to_id(raw_data, word2id):
    data = []

    def map_word_to_id(word):
        output = []
        if word in word2id:
            output.append(word2id[word])
        else:
            chars = list(word)
            for char in chars:
                if char in word2id:
                    output.append(word2id[char])
                else:
                    output.append(1)
        return output

    def map_sent_to_id(sent):
        output = []
        for word in sent:
            output.extend(map_word_to_id(word))
        return output

    for one in raw_data:
        question = map_sent_to_id(one[0])
        doc = map_sent_to_id(one[1])
        candidates = [map_word_to_id(x) for x in one[2]]
        length = [len(x) for x in candidates]
        max_length = max(length)
        if max_length > 1:
            pad_len = [max_length - x for x in length]
            candidates = [x[0] + [0] * x[1] for x in zip(candidates, pad_len)]
        data.append([question, doc, candidates, one[-1]])
    return data


def process_data(data_path, threshold, embedding_path=None, wordvec_path=None, embed_size=200):
    train_file_path = data_path + 'trainingset/train.json'
    dev_file_path = data_path + 'validationset/valid.json'
    test_a_file_path = data_path + 'testa/testa.json'
    # test_b_file_path = data_path + 'ai_challenger_oqmrc_testb_20180816/ai_challenger_oqmrc_testb.json'
    path_lst = [train_file_path, dev_file_path, test_a_file_path] #, test_b_file_path]
    output_path = [data_path + x for x in ['train.pickle', 'dev.pickle','testa.pickle']] # , 'testb.pickle']]
    return _process_data(path_lst, threshold, output_path, embedding_path, wordvec_path, embed_size)


def _process_data(path_lst, word_min_count=5, output_file_path=[],
                  embedding_path=None, wordvec_path=None, embed_size=200):

    raw_data = []
    for path in path_lst:
        raw_data.append(seg_data(path))

    word_count = build_word_count([y for x in raw_data for y in x])
    with open(data_path + 'word-count.pickle', 'wb') as f:
        pickle.dump(word_count, f)

    word2id = build_word2id(word_count, word_min_count)

    with open(data_path + 'word2id.pickle', 'wb') as f:
        pickle.dump(word2id, f)
    for one_raw_data, one_output_file_path in zip(raw_data, output_file_path):
        with open(one_output_file_path, 'wb') as f:
            one_data = transform_data_to_id(one_raw_data, word2id)
            pickle.dump(one_data, f)

    embedding = np.random.randn(len(word2id), embed_size)
    embedding[0] = 0.0
    if os.path.exists(wordvec_path):
        print("load embedding vector from {}".format(embedding_path))
        with open(embedding_path, "r") as f:
            count = 0
            for i, line in tqdm(enumerate(f)):
                if i > 1000000:
                    break
                content = line.strip().split()
                tokens = content[0]
                if tokens not in word2id:
                    continue
                else:
                    print(tokens)
                count += 1
                embedding[word2id[tokens]] = np.asarray(list(map(float, content[1:])))
            print("pre-trained words {}".format(count))
            np.save(wordvec_path, embedding)
    else:
        print("load wordvec from the npy file {}".format(wordvec_path))
        embedding = np.load(wordvec_path)
    return len(word2id), embedding

if __name__ == "__main__":
    data_path = "/home/zhengyinhe/xie_data/data/"
    embedding_path = "/home/zhengyinhe/xie_data/data/Tencent_AILab_ChineseEmbedding.txt"
    wordvec_apth = "/home/zhengyinhe/xie_data/data/tencent_wordvec_200.npy"
    process_data(data_path, 10, embedding_path, wordvec_apth, embed_size=200)

"""
threshold = 5:
word type size  332265
processed word size  96973
load embedding vector from /home/panxie/Document/aic_rc/data/Tencent_AILab_ChineseEmbedding.txt
990082it [00:11, 85898.50it/s]pre-trained words 88197

threshold = 10:
word type size  332265
processed word size  64140
load embedding vector from /home/panxie/Document/aic_rc/data/Tencent_AILab_ChineseEmbedding.txt
995471it [00:11, 90491.58it/s]pre-trained words 59942
"""
