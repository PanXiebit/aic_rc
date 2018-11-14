#!/usr/bin/python
# coding:utf-8
import tensorflow as tf


class Config(object):
    def __init__(self):
        flags = tf.app.flags
        # vocab
        flags.DEFINE_string("trainset_path", "/home/zhengyinhe/xie_data/trainingset/trainingset.json",
                            "the path of trainset json file")
        flags.DEFINE_string("validset_path", "/home/zhengyinhe/xie_data/validationset/trainset.json",
                            "the path of trainset json file")
        flags.DEFINE_string("embedding_path", "/home/zhengyinhe/xie_data/data/Tencent_AILab_ChineseEmbedding.txt",
                            "the path of embedding vector")
        flags.DEFINE_string("vocab_path", "/home/zhengyinhe/xie_data/data/vocab.txt",
                            "the vocab file path")
        flags.DEFINE_integer("embed_size", 200, "the embedding size")
        flags.DEFINE_boolean("use_pretrained", True, "wheather to use the pretrained wordvec")
        flags.DEFINE_list("use_gpus", [0, 1], "the gpu id to use")

        # dataset file
        flags.DEFINE_string("train_tfrecord_file", "/home/zhengyinhe/xie_data/data/trainset.tfrecord",
                            "the path of trainset tfrecord file")
        flags.DEFINE_string("valid_tfrecord_file", "/home/zhengyinhe/xie_data/data/validset.tfrecord",
                            "the path of trainset tfrecord file")
        flags.DEFINE_string("test_tfrecord_file", "/home/zhengyinhe/xie_data/data/test.tfrecord",
                            "the path of trainset tfrecord file")
        flags.DEFINE_integer("train_batch_size", 20, "train batch size")
        flags.DEFINE_integer("valid_batch_size", 20, "valid batch size")
        flags.DEFINE_integer("num_epochs", 20, "epoch number")
        flags.DEFINE_boolean("is_training", True, "train/inference")
        flags.DEFINE_float("dropout_keep_prob", 0.9, "dropout probability to keep")

        # model save and graph save
        flags.DEFINE_string("log_dir", "/home/zhengyinhe/xie_data/v6_cnn/model/log_v1/", "the path of graph to save")
        flags.DEFINE_string("train_dir", "/home/zhengyinhe/xie_data/v6_cnn/model/train_v1",
                            "the path of train graph to save")
        flags.DEFINE_string("valid_dir", "/home/zhengyinhe/xie_data/v6_cnn/model/valid_v1",
                            "the path of valid graph to save")
        flags.DEFINE_string("best_test_save", "/home/zhengyinhe/xie_data/v6_cnn/model/best_test_v1",
                            "best test save path", )
        flags.DEFINE_string("result_path", "/home/zhengyinhe/xie_data/v6_cnn/model/result_v1",
                            "the path of result path")

        # length limits
        flags.DEFINE_integer("max_passage", 500, "the max length of passage")
        flags.DEFINE_integer("max_query", 20, "the max length of query")
        flags.DEFINE_integer("max_answer", 5, "the max length of answer")

        # QANet
        flags.DEFINE_integer("num_blocks", 3, "the number of blocks during encoder")
        flags.DEFINE_integer("conv_layer_num", 4, "the number of conv layers")
        flags.DEFINE_integer("kernel_size", 5, "the kernel size of convolution")
        flags.DEFINE_boolean("mask", True, "to mask padding during passage/query encoding")
        flags.DEFINE_integer("num_filters", 128, "the number of channels")
        flags.DEFINE_integer("num_heads", 8, "the heads of self-attention")

        # RNN
        flags.DEFINE_integer("hidden_size", 200, "the size of hidden state")
        flags.DEFINE_integer("num_units", 200, "the size of num_units of attention pooling")
        flags.DEFINE_string("rnn_type", "bi-gru", "the encoder and fusion rnn tyep")
        flags.DEFINE_integer("layer_num", 2, "the number of multiRNN layers")

        # optimizer
        flags.DEFINE_string("optim", "adam", "adam/sgd/rprop/adagrad")
        flags.DEFINE_float("learning_rate", 0.001, "the learning rate")
        flags.DEFINE_float("learning_rate_decay_factor", 0.9, "learning_rate_decay_factor")
        flags.DEFINE_float("weight_decay", 3e-6, "the weight decay")

        # test
        tf.app.flags.DEFINE_integer("save_every_n_iteration", 50, "save_every_n_iteration")
        flags.DEFINE_string("test_tfrecord", "/home/zhengyinhe/xiepan/aic_rc/data/test.tfrecord",
                            "the path of test tfrecord file")
        flags.DEFINE_integer("test_size", 20, "the size of test")
        self.FLAGS = flags.FLAGS
