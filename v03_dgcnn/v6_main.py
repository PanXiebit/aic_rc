#!/usr/bin/python
# coding:utf-8

import os
import time
import tensorflow as tf
from cnn_model import RCModel
from dataset import get_dataset
from vocab import Vocab
from config import Config
import logging
import numpy as np
#
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = Config()
FLAGS = config.FLAGS
def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('aic_rc_v1.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] \
        >> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
logger = get_logger()

def train():
    # session info
    sess_config = tf.ConfigProto(
        # log_device_placement=True,
        allow_soft_placement=True,
    )
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # vocab
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            vocab = Vocab(FLAGS.trainset_path, FLAGS.embedding_path, FLAGS.vocab_path,
                          min_count=10, embed_size=FLAGS.embed_size, use_pretrained=False)
            rcmodel = RCModel(vocab, FLAGS)
            logger.info("Reading model parameters from %s" % FLAGS.train_dir)
            rcmodel.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            vocab = Vocab(FLAGS.trainset_path, FLAGS.embedding_path, FLAGS.vocab_path,
                          min_count=10, embed_size=FLAGS.embed_size, use_pretrained=FLAGS.use_pretrained)
            print("creat new parameters")
            rcmodel = RCModel(vocab, FLAGS)
            logger.info("the third version of model!")
            sess.run(tf.global_variables_initializer())


        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'), graph=tf.get_default_graph())
        valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'valid'))

        sess.run(rcmodel.train_iterator.initializer, feed_dict={rcmodel.train_size:FLAGS.train_batch_size})
        sess.run(rcmodel.valid_iterator.initializer, feed_dict={rcmodel.valid_size:FLAGS.valid_batch_size})
        train_handle = sess.run(rcmodel.train_iterator.string_handle())
        valid_handle = sess.run(rcmodel.valid_iterator.string_handle())

        test_loss_placeholder = tf.placeholder(dtype=tf.float32)
        test_loss_summary_op = tf.summary.scalar("test_loss", test_loss_placeholder)

        time_step = 0
        best_test_loss = 1e6
        test_prev_loss = [1e6] * 5
        while True:
            try:
                start_time = time.time()
                # run one step
                train_feed_dict = {rcmodel.data_handle:train_handle,
                                   rcmodel.dropout_keep_prob:FLAGS.dropout_keep_prob}

                train_summary, _, train_total_loss, train_acc, step, learning_rate, = sess.run(
                    fetches=[merged, rcmodel.train_op, rcmodel.total_loss, rcmodel.accuracy,rcmodel.global_step, rcmodel.learning_rate],
                    feed_dict=train_feed_dict)
                train_writer.add_summary(train_summary, global_step=step)

                if step % FLAGS.save_every_n_iteration == 0:
                    # validation
                    valid_feed_dict = {rcmodel.data_handle: valid_handle,
                                       rcmodel.dropout_keep_prob: 1.0}

                    valid_summary, valid_total_loss, valid_acc, valid_logits, valid_labels = sess.run(
                        fetches=[merged, rcmodel.total_loss, rcmodel.accuracy, rcmodel.logits, rcmodel.labels],
                        feed_dict=valid_feed_dict
                    )
                    valid_writer.add_summary(valid_summary, global_step=step)

                    if step % 50 == 0:
                        time_step += (time.time() - start_time)
                        logger.info("step:{}, step-time:{:.4f}\n"
                                            "train_total_loss:{:.4f}, train_acc:{:.4f}\n"
                                            "valid_total_loss:{:.4f}, valid_acc:{:.4f}, learning_rate:{:.6f}\n"
                                            # "valid_logits:{}\n "
                                            .format(step, time_step, train_total_loss, train_acc,
                                                    valid_total_loss, valid_acc, learning_rate,
                                                    ))
                    if step % 200 == 0:
                        rcmodel.saver.save(sess, save_path="%s/model.ckpt" % FLAGS.train_dir, global_step=step)

                if step % 500 == 0:
                    test_acc_epoch = 0
                    test_loss = 0
                    num = 0
                    while True:
                        try:
                            num += 1
                            test_feed_dict = {rcmodel.data_handle: valid_handle,
                                              rcmodel.dropout_keep_prob: 1.0}
                            test_total_loss, test_acc = sess.run(
                                fetches=[rcmodel.total_loss, rcmodel.accuracy],
                                feed_dict=test_feed_dict)
                            test_num = int(30000/FLAGS.valid_batch_size)
                            if num > test_num:
                                break
                            test_loss += test_total_loss
                            test_acc_epoch += test_acc
                        except tf.errors.OutOfRangeError:
                            break
                    test_acc_epoch /= num
                    test_loss /= num
                    if test_loss > max(test_prev_loss):
                        break
                    test_prev_loss[1:] += test_loss
                    logger.info("after step:{}, test_loss:{:.4f}, test_acc:{:.4f}"
                                .format(step, test_loss, test_acc_epoch))

                    [test_loss_summ] = sess.run([test_loss_summary_op],
                                                 feed_dict={test_loss_placeholder: test_loss})
                    valid_writer.add_summary(test_loss_summ, global_step=step)
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        rcmodel.saver.save(sess, "%s/model.ckpt" % FLAGS.best_test_save, global_step=step)
                if step > 100000:
                    break

            except tf.errors.OutOfRangeError:
                break

def test():
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = False

    with tf.Session(config=sess_config) as sess:
        # test and inference
        print("testing and reading model parameters from %s" % FLAGS.train_dir)
        vocab = Vocab(FLAGS.trainset_path, FLAGS.embedding_path, FLAGS.vocab_path,
                      min_count=10, embed_size=FLAGS.embed_size, use_pretrained=False)
        rcmodel = RCModel(vocab, FLAGS)
        print(FLAGS.train_dir)
        restore_time = time.time()
        model_path = "/home/zhengyinhe/xie_data/v4/model/train_v1/model.ckpt-00079950"
        rcmodel.saver.restore(sess, save_path=model_path)
        logger.info("restore done!, spend time :{}".format(time.time()-restore_time))
        # test_dataset = get_dataset(vocab.word2idx, FLAGS.test_tfrecord_file, FLAGS.test_size,
        #                            repeat_num=1, shuffle_bufer=500, prefetch=-1)
        # test_iterator = test_dataset.make_initializable_iterator()
        # sess.run(test_iterator.initializer)
        # test_handle = sess.run(test_iterator.string_handle())
        sess.run(rcmodel.train_iterator.initializer, feed_dict={rcmodel.train_size: FLAGS.train_batch_size})
        sess.run(rcmodel.valid_iterator.initializer, feed_dict={rcmodel.valid_size: FLAGS.valid_batch_size})
        train_handle = sess.run(rcmodel.train_iterator.string_handle())
        valid_handle = sess.run(rcmodel.valid_iterator.string_handle())

        def show_word(idx_list):
            return [vocab.idx2word[idx_list[i]] if idx_list[i] in vocab.idx2word
                    else "_UNK" for i in range(len(idx_list))]

        def cut_pad(sentence):
            if sentence.find("_PAD") != -1:
                return sentence[:sentence.find("_PAD")]
            return sentence

        result = {}
        acc = []
        id = 0
        while True:
            try:
                id += 1
                test_feed_dict = {rcmodel.data_handle:train_handle}
                test_ans = sess.run(
                    fetches=[rcmodel.predict,
                             rcmodel.query_id,
                             rcmodel.alter0,
                             rcmodel.alter1,
                             rcmodel.alter2,
                             rcmodel.accuracy],
                    feed_dict=test_feed_dict)
                batch_size = test_ans[0].shape[0]
                acc.append(test_ans[5])
                if id % 50 == 0:
                    logger.info("   processing the sample {}".format(id * batch_size))
                for i in range(batch_size):
                    query_id = test_ans[1][i]
                    if test_ans[0][i] == 0:
                        result[query_id] = cut_pad("".join(show_word(test_ans[2][i])))
                    elif test_ans[0][i] == 1:
                        result[query_id] = cut_pad("".join(show_word(test_ans[3][i])))
                    else:
                        result[query_id] = cut_pad("".join(show_word(test_ans[4][i])))
            except tf.errors.OutOfRangeError:
                logger.info(np.mean(acc))
                result = sorted(result.items(), key=lambda item: item[0])
                with open("/home/zhengyinhe/xie_data/v4/model/result_v3_valid.txt", "w") as f:
                    for query_id, answer_text in result:
                        f.write("{}\t{}\n".format(query_id, answer_text))
                logger.info("Done!")
                break


if __name__ == "__main__":
    if FLAGS.is_training:
        train()
    else:
        test()

#     sen = "xiepan"
#     print(sen.find("_PAD"))
