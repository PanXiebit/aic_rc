# -*- coding: utf-8 -*-
import argparse
import pickle
import torch
import logging

from model2 import MwAN
from preprocess import process_data
from utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('aic2_v2.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]   >> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
logger = get_logger()

parser = argparse.ArgumentParser(description='PyTorch implementation')

parser.add_argument('--data', type=str, default='/home/zhengyinhe/xiepan/aic_rc/model_73/data/',
                    help='location directory of the data corpus')
parser.add_argument('--threshold', type=int, default=5,
                    help='threshold count of the word')
parser.add_argument('--epoch', type=int, default=6,
                    help='training epochs')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=256,
                    help='hidden size of the model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--log_interval', type=int, default=300,
                    help='# of batches to see the training error')
parser.add_argument('--test_interval', type=int, default=600,
                    help='of batches to see the validation result')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model2_2.pt',
                    help='path to save the final model')

args = parser.parse_args()

# vocab_size = process_data(args.data, args.threshold)
word_embedding = np.load(args.data + "wordvec_200.npy")
char_embedding = np.load(args.data + "word_char_vec.npy")
print(word_embedding.shape, char_embedding.shape)

model = MwAN(word_embedding, char_embedding, embedding_size=args.emsize, encoder_size=args.nhid, drop_out=args.dropout)
logger.info(model)
logger.info("path to save the model {}".format(args.save))
print('Model total parameters:', get_model_parameters(model))
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adamax(model.parameters())
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
with open(args.data + 'train.pickle', 'rb') as f:
    train_data = pickle.load(f)
print(len(train_data))
with open(args.data + 'dev.pickle', 'rb') as f:
    dev_data = pickle.load(f)
dev_data = sorted(dev_data, key=lambda x: len(x[1]))
print(len(dev_data))
logger.info('train data size {:d}, dev data size {:d}'.format(len(train_data), len(dev_data)))


def train(epoch, best):
    #print(best)
    #model.train()
    data = shuffle_data(train_data, 1)
    total_loss = 0.0
    r, a = 0.0, 0.0
    for num, i in enumerate(range(0, int(len(data)), args.batch_size)):
        model.train()
        one = data[i:i + args.batch_size]
        query, _ = padding([x[0] for x in one], max_len=50)
        passage, _ = padding([x[1] for x in one], max_len=350)
        answer = pad_answer([x[2] for x in one])
        query, passage, answer= torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
        # print(label, type(label))
        if args.cuda:
            query = query.cuda()
            passage = passage.cuda()
            answer = answer.cuda()
        optimizer.zero_grad()
        loss, pred = model([query, passage, answer,True])
        r += torch.eq(pred,0).sum().item()
        a += len(one)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if (num + 1) % args.log_interval == 0:
            logger.info( '|---epoch {:d} train error is {:f}  acc is {:f}  eclipse {:.2f}%---|'
                         .format(epoch, total_loss / args.log_interval, r * 100.0 / a, i * 100.0 / int(len(data))))
            total_loss = 0
            r, a = 0.0, 0.0

        if (num + 1) % args.test_interval == 0:
            acc = test()
            if acc > best:
                best = acc
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    f.close()
                    logger.info('|------epoch {:d} eclipse {:2f} validation acc {:f}'.format(epoch, i * 100.0 / (len(data)), best) )
    return best

def test():
    model.eval()
    r, a = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(dev_data), args.batch_size):
            one = dev_data[i:i + args.batch_size]
            query, _ = padding([x[0] for x in one], max_len=50)
            passage, _ = padding([x[1] for x in one], max_len=500)
            answer = pad_answer([x[2] for x in one])
            query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
            if args.cuda:
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = model([query, passage, answer, False])
            r += torch.eq(output, 0).sum().item()
            a += len(one)
    return r * 100.0 / a


def main():
    best = 0.0
    for epoch in range(args.epoch):
        acc = train(epoch, best)
        if acc > best:
            best = acc
        acc = test()
        if acc > best:
            best = acc
            with open(args.save, 'wb') as f:
                torch.save(model, f)
                f.close()
        logger.info('epcoh {:d} validation acc is {:f}, best validation  acc {:f}'.format(epoch, acc, best))


if __name__ == '__main__':
    main()
