#!/usr/bin/python
# -*- encoding = utf-8 -*-
# coding:utf-8

import pickle
import json
from tqdm import tqdm
import jieba
import numpy as np
import os

# def _encode(char):
#     return unicode(char, encoding="utf-8")

def seg_line(line, word_len=3):
    char_line = []
    word_list = list(jieba.cut(line))
    # word_list = [_encode(x) for x in word_list]
    for word in word_list:
        word = word[:word_len]
        if len(word) < word_len:
            word = list(word) + ["PAD"] * (word_len-len(word))
        char_line.append(word)
    char_line = [" ".join(x) for x in char_line]
    return " ".join(char_line)


def seg_data(path):
    print ('\nstart process {}'.format(path))
    data = []
    with open(path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            dic = json.loads(line, encoding='utf-8')
            if len(dic["alternatives"].split("|")) != 3:
                continue
            question = dic['query']
            # print(question)
            doc = dic['passage']
            alternatives = dic['alternatives'].split("|")
            query = seg_line(question)
            ans1 = seg_line(alternatives[0])
            ans2 = seg_line(alternatives[1])
            ans3 =seg_line(alternatives[2])
            data.append([query, seg_line(doc), [ans1, ans2, ans3], dic['query_id']])
    return data


def transform_data_to_id(raw_data, char2id):
    data = []

    def map_word_to_id(word):
        output = []
        if word in char2id:
            output.append(char2id[word])
        else:
            output.append(char2id["UNK"])
        return output

    def map_sent_to_id(sent):
        output = []
        for word in sent.split():
            output.extend(map_word_to_id(word))
        return output

    for one in raw_data:
        question = map_sent_to_id(one[0])
        doc = map_sent_to_id(one[1])
        candidates = [map_sent_to_id(x) for x in one[2]]
        length = [len(x) for x in candidates]
        max_length = max(length)
        if max_length > 1:
            pad_len = [max_length - x for x in length]
            candidates = [x[0] + [0] * x[1] for x in zip(candidates, pad_len)]
        data.append([question, doc, candidates, one[-1]])
    return data


def process_data(data_path, threshold=5):
    train_file_path = data_path + 'trainingset/train.json'
    dev_file_path = data_path + 'validationset/valid.json'
    test_a_file_path = data_path + 'testa/testa.json'
    # test_b_file_path = data_path + 'ai_challenger_oqmrc_testb_20180816/ai_challenger_oqmrc_testb.json'
    path_lst = [train_file_path, dev_file_path, test_a_file_path] #, test_b_file_path]
    output_path = [data_path + x for x in ['char_train.pickle', 'char_dev.pickle','char_testa.pickle']] # , 'testb.pickle']]
    return _process_data(path_lst, threshold, output_path)


def _process_data(path_lst, word_min_count=5, output_file_path=[]):
    raw_data = []
    for path in path_lst:
        raw_data.append(seg_data(path))


    print("load word2id from {}".format(data_path + "char2id.pickle"))
    char2id = pickle.load(open(data_path + "char2id.pickle", "rb"))

    for one_raw_data, one_output_file_path in zip(raw_data, output_file_path):
         with open(one_output_file_path, 'wb') as f:
            one_data = transform_data_to_id(one_raw_data, char2id)
            pickle.dump(one_data, f)



if __name__ == "__main__":
    data_path = "/home/haixiao.liu/xiepan/project/aic_data/"
    process_data(data_path)
