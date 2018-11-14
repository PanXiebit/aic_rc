import json
import tqdm
import jieba
from collections import Counter
import os
import tensorflow as tf
import numpy as np

START_WORD = ['_PAD', '_UNK']

def simple_process(sent):
    # jieba.add_word("无法确定")
    sent = sent.strip()
    sent = sent.replace(" ", "")
    return list(jieba.cut(sent))

class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    """
    def __init__(self, trainset_path=None, embedding_path=None, vocab_path=None,
                 min_count=10, embed_size=500, use_pretrained=False):
        self.trainset_path = trainset_path
        self.vocab_path = vocab_path
        self.embed_size = embed_size
        self.embedding_path = embedding_path
        self.min_count = min_count

        # get the vocabulary from the trainset file
        self.get_vocab()

        self.randomly_get_embedding(embed_size=self.embed_size)
        if use_pretrained:
            self.load_pretrained_embedding(self.embedding_path)
        for token in ["_PAD", "_UNK"]:
            self.embedding[self.get_id(token)] = np.zeros([embed_size])
        self.embedding = tf.cast(self.embedding, tf.float32)

    def size(self):
        return len(self.worddict)

    def get_id(self, token):
        try:
            return self.word2idx[token]
        except KeyError:
            return self.word2idx["_UNK"]

    def get_word(self, idx):
        return self.idx2word[idx]

    def randomly_get_embedding(self, embed_size):
        """
        randomly initializes the embeddings for each token
        :param
        embed_size: the size of the embedding for each token
        """
        self.embedding = np.random.normal(size=(self.size(), embed_size))

    def load_pretrained_embedding(self, embedding_path):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered
        """
        if self.embedding_path is None:
            print("no pretrained embedding!")
        else:
            wordvec_path = "/home/zhengyinhe/xie_data/data/tencent_wordvec_200.npy"
            if not os.path.exists(wordvec_path):
                print("load embedding vector from {}".format(embedding_path))
                with open(embedding_path, "r", encoding="utf-8") as f:
                    count = 0
                    for i, line in tqdm.tqdm(enumerate(f)):
                        content = line.strip().split()
                        tokens = content[0]
                        if tokens not in self.worddict:
                            continue
                        count += 1
                        self.embedding[self.word2idx[tokens]] = np.array(list(map(float, content[1:])))
                    np.save(wordvec_path, self.embedding)
            else:
                print("load wordvec from the npy file {}".format(wordvec_path))
                self.embedding = np.load(wordvec_path)
            # print("{} words been pretrained".format(count))

    def get_vocab(self):
        if not os.path.exists(self.vocab_path):
            print("load vocab from the {} file!".format(self.trainset_path))
            vocab = []
            max_passage = 0
            max_query = 0
            max_answer = 0
            passage_len = []
            query_len = []
            answer_len = []
            with open(self.trainset_path) as f:
                for i, line in tqdm.tqdm(enumerate(f)):
                    content = json.loads(line)
                    passage = content["passage"]
                    query = content['query']
                    answer = content["answer"]
                    alternatives = content["alternatives"]
                    passage = simple_process(passage)
                    query = simple_process(query)
                    answer = simple_process(answer)
                    alternatives = simple_process(alternatives)
                    vocab.extend(passage)
                    vocab.extend(query)
                    vocab.extend(answer)
                    passage_len.append(len(passage))
                    query_len.append(len(query))
                    answer_len.append(len(answer))
                    if len(passage) + 1 > max_passage:
                        max_passage = len(passage) + 1
                    if len(query) + 1 > max_query:
                        max_query = len(query) + 1
                    if len(answer) > max_answer:
                        max_answer = len(answer)
                        print(answer)
                        print(alternatives)
                    if len(answer) > 10:
                        print(answer, len(answer))  # only one example is longer than 10
                        print(alternatives)
            print("the number of query is {}".format(i))
            print("max length of passage {}".format(max_passage))  # 14015
            print("max length of query {}".format(max_query))  # 29
            print("max length of answer {}".format(max_answer))  # 20

            passage_len_pairs = Counter(passage_len)
            print("passage length {}".format(passage_len_pairs))
            query_len_pairs = Counter(query_len)
            print("query length {}".format(query_len_pairs))
            answer_len_pairs = Counter(answer_len)
            print("answer length {}".format(answer_len_pairs))

            word_pairs = Counter(vocab)
            self.worddict = []
            for word, word_num in word_pairs.items():
                if word_num > self.min_count:
                    self.worddict.append(word)

            # * de zuo yong: https://blog.csdn.net/xiaoqu001/article/details/78823498
            self.worddict = START_WORD + self.worddict
            # self.size = len(self.worddict)
            print("the size of vocabulary {}".format(len(self.worddict)))

            with open(self.vocab_path, "w") as f:
                for word in self.worddict:
                    f.write(word + "\n")

            self.word2idx = dict(zip(self.worddict, range(len(self.worddict))))
            self.idx2word = dict(zip(range(len(self.worddict)), self.worddict))
            # print(word2idx)
        else:
            print("load vocabulay and word2idx from {}".format(self.vocab_path))
            self.worddict = []
            with open(self.vocab_path, "r") as f:
                for line in tqdm.tqdm(f):
                    self.worddict.append(line.strip())
                self.word2idx = dict(zip(self.worddict, range(len(self.worddict))))
                self.idx2word = dict(zip(range(len(self.worddict)), self.worddict))


if __name__ == "__main__":
    infile = "/home/zhengyinhe/xiepan/aic_rc/trainingset/trainingset.json"
    # infile = "/home/zhengyinhe/xiepan/aic_rc/validationset/validationset.json"
    vocab_path = "/home/zhengyinhe/xiepan/aic_rc/data/vocab.txt"
    embed_path = "/home/zhengyinhe/xiepan/aic_rc/data/model_500_sg.vec"
    vocab = Vocab(trainset_path=infile, vocab_path=vocab_path, embedding_path=embed_path)
    # vocab.get_vocab()
    index1 = vocab.get_id("维生素")
    index2 = vocab.get_id("问题")
    index3 = vocab.get_id("_EOS")

    word1 = vocab.get_word(17)
    word2 = vocab.get_word(29)
    word3 = vocab.get_word(2)
    print(index1, index2, index3)
    print(word1, word2, word3)
    # # print(simple_process("无法确定"))


