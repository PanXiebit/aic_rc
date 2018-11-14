# -*- coding: utf-8 -*-
import argparse
import pickle
import codecs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from utils import *

from preprocess import seg_data, transform_data_to_id

parser = argparse.ArgumentParser(description='inference procedure, '
                                             'note you should train the data at first')

parser.add_argument('--data', type=str,
                    default='/home/zhengyinhe/xie_data/data/testa/testa.json',
                    help='location of the test data')

parser.add_argument('--valid_data', type=str,
                    default='/home/zhengyinhe/xie_data/data/validationset/valid.json',
                    help='location of the test data')

parser.add_argument('--word_path', type=str, default='/home/zhengyinhe/xie_data/data/word2id.pickle',
                    help='location of the word2id.obj')

parser.add_argument('--output', type=str, default='prediction.a_1.txt',
                    help='prediction path')

parser.add_argument('--valid', type=str, default='valid_1.txt',
                    help='prediction path')

parser.add_argument('--model', type=str, default='model_3.pt',
                    help='model path')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_true',default=True,
                    help='use CUDA')

args = parser.parse_args()

with open(args.model, 'rb') as f:
    model = torch.load(f)
if args.cuda:
    model.cuda()
    print(model)

with open(args.word_path, 'rb') as f:
    word2id = pickle.load(f)
    print(len(word2id))

raw_data = seg_data(args.data)
transformed_data = transform_data_to_id(raw_data, word2id)
data = [x + [y[2]] for x, y in zip(transformed_data, raw_data)]
data = sorted(data, key=lambda x: len(x[1]))
print ('test data size {:d}'.format(len(data)))

raw_data_valid = seg_data(args.valid_data)
transformed_data_valid = transform_data_to_id(raw_data_valid, word2id)
dev_data = [x + [y[2]] for x, y in zip(transformed_data_valid, raw_data_valid)]
dev_data = sorted(dev_data, key=lambda x: len(x[1]))
print ('valid data size {:d}'.format(len(dev_data)))

def inference():
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            one = data[i:i + args.batch_size]
            query, _ = padding([x[0] for x in one], max_len=50)
            passage, _ = padding([x[1] for x in one], max_len=300)
            answer = pad_answer([x[2] for x in one])
            str_words = [x[-1] for x in one]
            ids = [x[3] for x in one]
            query, passage = torch.LongTensor(query), torch.LongTensor(passage),
            answer = torch.LongTensor(answer)
            if args.cuda:
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = model([query, passage, answer, False])
            for q_id, prediction, candidates in zip(ids, output, str_words):
                prediction_answer = u''.join(candidates[prediction])
                predictions.append(str(q_id) + '\t' + prediction_answer)
    outputs = u'\n'.join(predictions)
    print(outputs)
    with codecs.open(args.output, 'w',encoding='utf-8') as f:
        f.write(outputs)
    print ('done!')



print('dev data size {:d}'.format(len(dev_data)))
def dev_infer():
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(dev_data), args.batch_size):
            one = dev_data[i:i + args.batch_size]
            query, _ = padding([x[0] for x in one], max_len=50)
            passage, _ = padding([x[1] for x in one], max_len=300)
            answer = pad_answer([x[2] for x in one])
            str_words = [x[-1] for x in one]
            ids = [x[3] for x in one]
            query, passage = torch.LongTensor(query), torch.LongTensor(passage),
            answer = torch.LongTensor(answer)
            if args.cuda:
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = model([query, passage, answer, False])
            for q_id, prediction, candidates in zip(ids, output, str_words):
                prediction_answer = u''.join(candidates[prediction])
                predictions.append(str(q_id) + '\t' + prediction_answer)
    outputs = u'\n'.join(predictions)
    print(outputs)
    with codecs.open(args.valid, 'w',encoding='utf-8') as f:
        f.write(outputs)
    print ('done!')


if __name__ == '__main__':
    inference()
    dev_infer()
