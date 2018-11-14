# get tfrecord
import tensorflow as tf
import logging
from vocab import simple_process, Vocab
import tqdm
import json
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_record(word2index, infile, outfile, max_passage=500, max_query=20, max_answer=5):
    """
    generate tfrecord files based on the file given by infile
    passages and query are padded using PAD_ID
    Only the index of each word is saved.

    :param word2index: dict from word to index
    :param infile: input file
    :param outfile: output tfrecord file
    :return:
    Returns nothing, but generate a outfile file for the generated tfrecord
    """
    if not os.path.exists(outfile):
        print("begin to generate tfrecord file!")
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _byte_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        with tf.python_io.TFRecordWriter(outfile) as tfrecord_writer:
            print("loading {}".format(infile))
            with open(infile) as f:
                for i, line in tqdm.tqdm(enumerate(f)):
                    content = json.loads(line)
                    if len(content["alternatives"].split("|")) != 3:
                        continue
                    passage = simple_process(content['passage'])
                    query = simple_process(content['query'])
                    query_id = content['query_id']
                    if "test" not in infile:
                        answer = simple_process(content["answer"])
                    else:
                        answer = simple_process("_PAD")
                    alter0 = simple_process(content["alternatives"].split("|")[0])
                    alter1 = simple_process(content["alternatives"].split("|")[1])
                    alter2 = simple_process(content["alternatives"].split("|")[2])
                    if i % 10000 == 0:
                        print("   processing line {}".format(i))
                        print(passage)
                        print(query)
                        print(answer)

                    # word to index
                    passage_idx = [word2index[word] if word in word2index else word2index["_UNK"] for word in passage]
                    query_idx = [word2index[word] if word in word2index else word2index['_UNK'] for word in query]
                    answer_idx = [word2index[word] if word in word2index else word2index['_UNK'] for word in answer]
                    alter0_idx = [word2index[word] if word in word2index else word2index['_UNK'] for word in alter0]
                    alter1_idx = [word2index[word] if word in word2index else word2index['_UNK'] for word in alter1]
                    alter2_idx = [word2index[word] if word in word2index else word2index['_UNK'] for word in alter2]

                    # valid length
                    passage_len = len(passage_idx)
                    query_len = len(query_idx)
                    answer_len = len(answer_idx)
                    alter0_len = len(alter0_idx)
                    alter1_len = len(alter1_idx)
                    alter2_len = len(alter2_idx)

                    # truncate but no padding
                    passage_idx = passage_idx[:max_passage]
                    query_idx = query_idx[:max_query]
                    answer_idx = answer_idx[:max_answer]
                    alter0_idx = alter0_idx[:max_answer]
                    alter1_idx = alter1_idx[:max_answer]
                    alter2_idx = alter2_idx[:max_answer]

                    # to numpy ndarray
                    answer_idx = np.array(answer_idx)
                    passage_idx = np.array(passage_idx)   # (len, )
                    query_idx = np.array(query_idx)       # (len, )
                    alter0_idx = np.array(alter0_idx)
                    alter1_idx = np.array(alter1_idx)
                    alter2_idx = np.array(alter2_idx)

                    feature = dict()
                    # tf.compat.as_bytes:
                    #   Converts either bytes or unicode to `bytes`, using utf-8 encoding for text.
                    feature["passage"] =  _byte_feature(tf.compat.as_bytes(passage_idx.flatten().tostring()))
                    feature["query"] = _byte_feature(tf.compat.as_bytes(query_idx.flatten().tostring()))
                    feature["answer"] = _byte_feature(tf.compat.as_bytes(answer_idx.flatten().tostring()))
                    feature["alter0"] = _byte_feature(tf.compat.as_bytes(alter0_idx.flatten().tostring()))
                    feature["alter1"] = _byte_feature(tf.compat.as_bytes(alter1_idx.flatten().tostring()))
                    feature["alter2"] = _byte_feature(tf.compat.as_bytes(alter2_idx.flatten().tostring()))
                    feature['query_id'] = _int64_feature([query_id])
                    feature['passage_len'] = _int64_feature([passage_len])
                    feature["query_len"] = _int64_feature([query_len])
                    feature["answer_len"] = _int64_feature([answer_len])
                    feature["alter0_len"] = _int64_feature([alter0_len])
                    feature["alter1_len"] = _int64_feature([alter1_len])
                    feature["alter2_len"] = _int64_feature([alter2_len])

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    tfrecord_writer.write(example.SerializeToString())
                print('Finished, generated tfrecord_files: {}'.format(outfile))
    else:
        print("tfrecord file had been generate {}".format(outfile))


def get_dataset(word2index, tfrecord_file, batch_size=60, repeat_num=None, shuffle_bufer=1000, prefetch=1000):
    def _parse(example):
        feature = {"passage":tf.FixedLenFeature([], tf.string),
                   "passage_len":tf.FixedLenFeature([], tf.int64),
                   "query":tf.FixedLenFeature([], tf.string),
                   "query_id": tf.FixedLenFeature([], tf.int64),
                   "query_len":tf.FixedLenFeature([], tf.int64),
                   "answer":tf.FixedLenFeature([], tf.string),
                   "answer_len": tf.FixedLenFeature([], tf.int64),
                   "alter0":tf.FixedLenFeature([], tf.string),
                   "alter1": tf.FixedLenFeature([], tf.string),
                   "alter2": tf.FixedLenFeature([], tf.string),
                   "alter0_len": tf.FixedLenFeature([], tf.int64),
                   "alter1_len": tf.FixedLenFeature([], tf.int64),
                   "alter2_len": tf.FixedLenFeature([], tf.int64)}
        data = tf.parse_single_example(example, features=feature)

        return data

    def _str2np(example):
        # r"""Reinterpret the bytes of a string as a vector of numbers.
        example["passage"] = tf.decode_raw(example["passage"], tf.int64)
        example["query"] = tf.decode_raw(example["query"], tf.int64)
        example["answer"] = tf.decode_raw(example["answer"], tf.int64)
        example["alter0"] = tf.decode_raw(example["alter0"], tf.int64)
        example["alter1"] = tf.decode_raw(example["alter1"], tf.int64)
        example["alter2"] = tf.decode_raw(example["alter2"], tf.int64)
        return example

    def _change_shape(example):
        example["passage"] = tf.reshape(example['passage'], (-1,))
        example["query"] = tf.reshape(example['query'], (-1,))
        example["answer"] = tf.reshape(example['answer'], (-1,))
        example["answer_len"] = tf.reshape(example['answer_len'], (-1,))
        example["query_id"] = tf.reshape(example['query_id'], (-1,))
        example["passage_len"] = tf.reshape(example['passage_len'], (-1,))
        example["query_len"] = tf.reshape(example['query_len'], (-1,))
        example["alter0"] = tf.reshape(example["alter0"], (-1,))
        example["alter1"] = tf.reshape(example["alter1"], (-1,))
        example["alter2"] = tf.reshape(example["alter2"], (-1,))
        example["alter0_len"] = tf.reshape(example['alter0_len'], (-1,))
        example["alter1_len"] = tf.reshape(example['alter1_len'], (-1,))
        example["alter2_len"] = tf.reshape(example['alter2_len'], (-1,))
        return example

    def _change_shape2(example):
        example["passage_len"] = tf.reshape(example['passage_len'], (-1,))
        example["query_len"] = tf.reshape(example['query_len'], (-1,))
        example["answer_len"] = tf.reshape(example['answer_len'], (-1,))
        example["query_id"] = tf.reshape(example['query_id'], (-1,))
        example["alter0_len"] = tf.reshape(example['alter0_len'], (-1,))
        example["alter1_len"] = tf.reshape(example['alter1_len'], (-1,))
        example["alter2_len"] = tf.reshape(example['alter2_len'], (-1,))
        return example

    # (e.g. `tf.Dimension(None)` in a `tf.TensorShape` or `-1` in a
    #         tensor-like object) will be padded to the maximum size of that
    #         dimension in each batch.
    shape = {'passage':[None],
             'query':[None],
             'answer':[None],
             'alter0':[None],
             'alter1': [None],
             'alter2': [None],
             'passage_len': [None],
             'query_len':[None],
             'answer_len':[None],
             'query_id':[None],
             'alter0_len':[None],
             'alter1_len': [None],
             'alter2_len': [None],
             }

    padding_value = {'passage':tf.constant(word2index["_PAD"], tf.int64),
                     'query': tf.constant(word2index["_PAD"], tf.int64),
                     'answer': tf.constant(word2index["_PAD"], tf.int64),
                     'alter0': tf.constant(word2index["_PAD"], tf.int64),
                     'alter1': tf.constant(word2index["_PAD"], tf.int64),
                     'alter2': tf.constant(word2index["_PAD"], tf.int64),
                     'passage_len': tf.constant(0, tf.int64),
                     'answer_len': tf.constant(0, tf.int64),
                     'query_len': tf.constant(0, tf.int64),
                     'query_id': tf.constant(0, tf.int64),
                     'alter0_len': tf.constant(0, tf.int64),
                     'alter1_len': tf.constant(0, tf.int64),
                     'alter2_len': tf.constant(0, tf.int64)
                     }

    dataset = tf.data.TFRecordDataset([tfrecord_file])
    dataset = dataset.map(_parse)
    dataset = dataset.map(_str2np)

    # if no this line, will get the error
    #       "All elements in a batch must have the same rank as the padded shape for component4:
    #       expected rank 1 but got element with rank 0"
    #       output_shapes=[[?,?], [?,?], [?,?], [?,?], [?], [?,?], [?], [?,?], [?]],
    dataset = dataset.map(_change_shape)

    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=shape,
                                   padding_values=padding_value)

    dataset = dataset.map(_change_shape2)

    # random shuffle dataset
    if shuffle_bufer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_bufer)

    # the epochs to repeat
    if repeat_num:
        dataset = dataset.repeat(repeat_num)

    # Creates a `Dataset` that prefetches elements from this dataset
    if prefetch > 0:
        dataset = dataset.prefetch(buffer_size=prefetch)

    return dataset

if __name__ == "__main__":
    data_path = "/home/zhengyinhe/xiepan/aic_rc"
    train_path = "/home/zhengyinhe/xiepan/aic_rc/trainingset/trainingset.json"
    valid_path = "/home/zhengyinhe/xiepan/aic_rc/validationset/validationset.json"
    test_path = "/home/zhengyinhe/xiepan/aic_rc/testa/testa.json"
    vocab_path = "/home/zhengyinhe/xiepan/aic_rc/data/vocab.txt"
    trainset_tfrecord = "/home/zhengyinhe/xiepan/aic_rc/data/trainset.tfrecord"
    validset_tfrecord = "/home/zhengyinhe/xiepan/aic_rc/data/validset.tfrecord"
    test_tfrecord = "/home/zhengyinhe/xiepan/aic_rc/data/test.tfrecord"
    vocab = Vocab(trainset_path=train_path, vocab_path=vocab_path, use_pretrained=False)

    # get_record
    # get_record(word2index=vocab.word2idx, infile=train_path, outfile=trainset_tfrecord)
    # get_record(vocab.word2idx, infile=valid_path, outfile=validset_tfrecord)
    get_record(vocab.word2idx, infile=test_path, outfile=test_tfrecord)

    # dataset
    # train_dataset = get_dataset(vocab.word2idx, trainset_tfrecord, batch_size=2, repeat_num=1, prefetch=1000)
    # valid_dataset = get_dataset(vocab.word2idx, validset_tfrecord, batch_size=5, repeat_num=1, prefetch=1000)
    test_dataset = get_dataset(vocab.word2idx, test_tfrecord, batch_size=5, repeat_num=1, prefetch=1000)
    #
    #
    # data_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()
    data_handle = tf.placeholder(tf.string, shape=[])
    data_iter = tf.data.Iterator.from_string_handle(data_handle,
                                                    test_dataset.output_types,
                                                    test_dataset.output_shapes)
    # next_batch = data_iterator.get_next()
    test_batch = test_iterator.get_next()
    batch_data = test_batch

    passage = batch_data["passage"]
    query = batch_data["query"]
    query_id = batch_data["query_id"]
    answer = batch_data["answer"]
    passage_len = batch_data["passage_len"]
    query_len = batch_data['query_len']
    answer_len = batch_data["answer_len"]
    alter0 = batch_data["alter0"]
    alter1 = batch_data["alter1"]
    alter2 = batch_data["alter2"]
    alter0_len = batch_data["alter0_len"]
    alter1_len = batch_data["alter1_len"]
    alter2_len = batch_data["alter2_len"]

    def show_word(list):
        return [vocab.idx2word[i] for i in list]

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = False
    with tf.Session(config=sess_config) as sess:
        sess.run(test_iterator.initializer)
        test_handle = sess.run(test_iterator.string_handle())
        id = 0
        while True:
            try:
                id += 1
                if id > 5:
                    break
                print(id)
                _passage, _query, _query_id, _passage_len, _query_len, _answer, _answer_len,\
                    _alter0, _alter1, _alter0_len, _alter1_len= sess.run(
                    fetches = [passage, query, query_id, passage_len, query_len, answer, answer_len,
                               alter0, alter1, alter0_len, alter1_len],
                    feed_dict={data_handle:test_handle})
                print(_passage.shape)
                print(_query.shape)
                print(_answer.shape)
                print("--------------")
                # print(_passage_len)
                # print(_query_len)
                # print(_answer_len)
                print(_alter0_len.shape)
                print(_alter1_len)
                print('--------------')
                print(_query_id[0])
                print("".join(show_word(_alter0[0])))
                # print("".join(show_word(_passage[0])))
                # print("".join(show_word(_query[0])))
                # print("".join(show_word(_answer[0])))
                print("--------------")
            except tf.errors.OutOfRangeError:
                break





