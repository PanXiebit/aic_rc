import os
import time
import tensorflow as tf
from layers.match_layer import match2
from dataset import get_dataset
# from layers.basic_rnn import _rnn
from layers.cnn_encoder import conv_block
from layers.attention_pooling import attention_pooling
class RCModel(object):
    """
    Implements the main reading comprehension model.
    """
    def __init__(self, vocab, FLAGS):
        # config
        # dataset
        self.train_batch_size = FLAGS.train_batch_size
        self.valid_batch_size = FLAGS.valid_batch_size
        self.train_tfrecord_file = FLAGS.train_tfrecord_file
        self.num_epochs = FLAGS.num_epochs
        self.valid_tfrecord_file = FLAGS.valid_tfrecord_file
        self.test_tfrecord_file = FLAGS.test_tfrecord_file
        self.is_training = FLAGS.is_training

        # length limits
        self.max_passage = FLAGS.max_passage
        self.max_query = FLAGS.max_query
        self.max_answer = FLAGS.max_answer

        # passage and query encoder
        self.num_blocks = FLAGS.num_blocks
        self.conv_layer_num = FLAGS.conv_layer_num
        self.kernel_size = FLAGS.kernel_size
        self.mask = FLAGS.mask
        self.num_filters = FLAGS.num_filters
        self.num_heads = FLAGS.num_heads
        self.initializer = lambda:tf.contrib.layers.variance_scaling_initializer(
            factor=2.0,
            mode='FAN_AVG',
            uniform=False,
            dtype=tf.float32)

        # answer encode
        self.hidden_size = FLAGS.hidden_size
        self.rnn_type = FLAGS.rnn_type
        self.layer_num = FLAGS.layer_num
        self.embed_size = FLAGS.embed_size
        self.num_units = FLAGS.num_units       # attention pooling

        # optimizer
        self.optim_type = FLAGS.optim
        self.weight_decay = FLAGS.weight_decay
        self.learning_rate = FLAGS.learning_rate
        self.learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
        self.use_gpus = FLAGS.use_gpus

        # the vocab
        self.vocab = vocab
        self._build_graph()


    def _build_graph(self):
        """
        build the computation with tensorflow
        """
        start_t = time.time()
        with tf.device('cpu:0'):
            self._setup_placeholders()
            self._make_input()
            self._embed()
            self._optimizer()

            losses = []
            acces = []
            tower_grads = []
            reuse_variables = False
            for i in self.use_gpus:
                with tf.device("/gpu:%d"%i):
                    self._encode(reuse=reuse_variables)
                    self._match(reuse=reuse_variables)
                    self._fuse(reuse=reuse_variables)
                    self._get_logits(reuse=reuse_variables)
                    cur_loss, cur_acc = self._computer_loss_and_acc(reuse=reuse_variables)
                    reuse_variables = True
                    losses.append(cur_loss)
                    acces.append(cur_acc)
                    var_and_grads = self.optimizer.compute_gradients(cur_loss) # pairs (gradient, variable)
                    tower_grads.append(var_and_grads)
            avg_grads_and_vars = self.average_gradients(tower_grads)

            self.mean_loss = tf.reduce_mean(tf.convert_to_tensor(losses), name="mean_losses")
            tf.summary.scalar("mean_losses", self.mean_loss)
            self.accuracy = tf.reduce_mean(tf.convert_to_tensor(acces), name="mean_accuracy")
            tf.summary.scalar("mean_accuracy", self.accuracy)

            # clipped_gradients, _ = tf.clip_by_global_norm(avg_grads_and_vars, self.max_gradient_norm)
            self.train_op = self.optimizer.apply_gradients(avg_grads_and_vars, global_step=self.global_step)

            print("Time to build graph: {} s".format(time.time() - start_t))
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                        max_to_keep=30, pad_step_number=True,
                                        keep_checkpoint_every_n_hours=1.0)
        for var in tf.trainable_variables():
            print(var)

    def _setup_placeholders(self):
        self.train_size = tf.placeholder(tf.int64, [], name="train_size")
        self.valid_size = tf.placeholder(tf.int64, [], name="valid_size")
        self.test_size = tf.placeholder(tf.int64, [], name="test_size")
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="dropout_keep")
        self.dropout = 1.0 - self.dropout_keep_prob
        self.data_handle = tf.placeholder(tf.string, shape=[], name="data_handle")

    def _make_input(self):
        train_dataset = get_dataset(self.vocab.word2idx, self.train_tfrecord_file, self.train_size,
                                    repeat_num=-1, shuffle_bufer=1000, prefetch=1000)
        valid_dataset = get_dataset(self.vocab.word2idx, self.valid_tfrecord_file, self.valid_size,
                                    repeat_num=-1, shuffle_bufer=1000)
        test_dataset = get_dataset(self.vocab.word2idx, self.test_tfrecord_file, self.test_size,
                                   repeat_num=1, shuffle_bufer=1000)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.valid_iterator = valid_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()

        data_iter = tf.data.Iterator.from_string_handle(self.data_handle,
                                                        train_dataset.output_types,
                                                        train_dataset.output_shapes)
        batch_data = data_iter.get_next()
        self.passage = tf.cast(batch_data["passage"],tf.int32, name="passage")
        self.query = tf.cast(batch_data["query"],tf.int32, name="query")
        self.answer = tf.cast(batch_data['answer'],tf.int32, name="query")
        self.passage_len = tf.cast(batch_data["passage_len"], tf.int32, name="passage_len")
        self.query_len = tf.cast(batch_data["query_len"], tf.int32, name="query_len")
        self.answer_len = tf.cast(batch_data["answer_len"], tf.int32, name="answer_len")
        self.query_id = tf.cast(batch_data["query_id"], tf.int32, name="query_id")

        # make labels and predict
        self.alter0 = tf.cast(batch_data["alter0"], tf.int32, name="alter0")
        self.alter1 = tf.cast(batch_data["alter1"], tf.int32, name="alter1")
        self.alter2 = tf.cast(batch_data["alter2"], tf.int32, name="alter2")
        self.alter0_len = tf.cast(batch_data["alter0_len"], tf.int32, name="alter0_len")
        self.alter1_len = tf.cast(batch_data["alter1_len"], tf.int32, name="alter1_len")
        self.alter2_len = tf.cast(batch_data["alter2_len"], tf.int32, name="alter2_len")

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device("/cpu:0"), tf.variable_scope("embedding"):
            self.word_embedding = tf.get_variable(
                name="word_embedding",
                initializer=self.vocab.embedding,
                trainable=True
            )
            # tf.summary.histogram("word-embedding", self.word_embedding)

            self.passage_emb = tf.nn.embedding_lookup(self.word_embedding, self.passage)
            self.query_emb = tf.nn.embedding_lookup(self.word_embedding, self.query)
            self.alter0_emb = tf.nn.embedding_lookup(self.word_embedding, self.alter0)
            self.alter1_emb = tf.nn.embedding_lookup(self.word_embedding, self.alter1)
            self.alter2_emb = tf.nn.embedding_lookup(self.word_embedding, self.alter2)

    def _encode(self, reuse=None):
        with tf.variable_scope("encoder_layer", reuse=reuse):
            self.enc_passage = conv_block(
                inputs=self.passage_emb, num_conv_block=3, kernel_size=3,
                dilation=2, num_filters=96, scope="encoder",
                reuse=reuse, dropout=self.dropout)
            self.enc_passage = tf.identity(self.enc_passage, name="enc_passage")

            self.enc_query = conv_block(
                inputs=self.query_emb, num_conv_block=3, kernel_size=3,
                dilation=2, num_filters=96, scope="encoder",
                reuse=True, dropout=self.dropout)
            self.enc_query = tf.identity(self.enc_query, name="enc_query")

            def _pad_alter(alter):
                alter = tf.pad(alter, [[0, 0], [0, self.max_answer - tf.shape(alter)[1]]])
                return alter

            answers = tf.concat([_pad_alter(self.alter0),
                                 _pad_alter(self.alter1),
                                 _pad_alter(self.alter2)], axis=0)  # [batch*3, max_answer]

            self.answers_emb = tf.nn.embedding_lookup(self.word_embedding, answers)  # [batch*3, max_answer, embed_size]
            self.enc_answers = conv_block(
                inputs=self.answers_emb, num_conv_block=3, kernel_size=3,
                dilation=2, num_filters=96, scope="encoder",
                reuse=True, dropout=self.dropout)
            self.enc_answers = tf.identity(self.enc_answers, name="enc_answers")  # # [batch*3, max_answer, num_filters]

    def _match(self, reuse=None):
        """
        co-attention mechanism
        """
        with tf.variable_scope("match_layer", reuse=reuse):
            sim_matrix = match2(self.enc_passage, self.enc_query, self.passage_len, self.query_len) # [batch, passage_len, query_len]
            self.mask_passage = tf.cast(self.passage, tf.bool)                   # [batch, passage_len]
            mask_p = tf.cast(tf.expand_dims(self.mask_passage, axis=2), tf.float32)  # [batch, passage_len, 1]
            S_T = tf.transpose(tf.nn.softmax(sim_matrix + -1e30 * (1-mask_p), axis=1), (0,2,1)) # [batch, query_len, passage_len]

            self.mask_query = tf.cast(self.query, tf.bool)
            mask_q = tf.cast(tf.expand_dims(self.mask_query, axis=1), tf.float32)    # [batch, 1, query_len]
            S_ = tf.nn.softmax(sim_matrix + -1e30 * (1-mask_q), axis=2)      # [batch, passage_len, query_len]

            self.p2q = tf.matmul(S_, self.enc_query)
            self.q2p = tf.matmul(tf.matmul(S_, S_T), self.enc_passage)   # [batch, passage_len, num_filters]
            self.match_out = tf.concat([self.enc_passage, self.p2q, self.enc_passage * self.p2q, self.enc_passage*self.q2p], axis=-1)
            self.match_out = tf.nn.dropout(self.match_out, self.dropout_keep_prob, name="match_out")
            self.match_out = tf.identity(self.match_out, "match_out_idt")

    def _fuse(self, reuse=None):
        with tf.variable_scope("fusion_layer", reuse=reuse):
            self.fusion_out = conv_block(
                inputs=self.match_out, num_conv_block=6, kernel_size=3,
                dilation=2, num_filters=96, scope="fusion",
                reuse=reuse, dropout=self.dropout)
            self.fusion_out = tf.identity(self.fusion_out, name="fusion_out")

    def _get_logits(self, reuse=None):
        # make labels
        with tf.variable_scope("compute_loss", reuse=reuse):
            self.alters = tf.concat([tf.reduce_sum(self.alter0, 1, keepdims=True),
                                     tf.reduce_sum(self.alter1, 1, keepdims=True),
                                     tf.reduce_sum(self.alter2, 1, keepdims=True)], axis=1)     # [batch, 3]
            self.labels = tf.cast(tf.equal(
                self.alters, tf.reshape(tf.reduce_sum(self.answer, axis=1), (-1,1))), tf.float32)  # [batch ,3]

            def labels_smoothing(inputs, epsilon=0.1):
                K = inputs.get_shape().as_list()[-1]
                return ((1.0-epsilon)*inputs) + (epsilon/K)
            self.labels = tf.identity(labels_smoothing(self.labels), "labels")

            self.logits = attention_pooling(self.enc_query, self.fusion_out, self.enc_answers,
                                            self.num_filters, self.initializer())

    def _computer_loss_and_acc(self, reuse=None):
        with tf.variable_scope("computer_losses", reuse=reuse):
            # cross entropy
            self.cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.labels, name="cross_ent")
            self.loss = tf.reduce_mean(self.cross_ent, name="loss")
            tf.summary.scalar("loss", self.loss)
            if self.weight_decay:
                self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                         if "bias" and "embedding" not in v.name])
            self.total_loss = self.loss + self.l2_loss * self.weight_decay
            tf.summary.scalar("total_loss", self.total_loss)
            self.batch_size = tf.shape(self.passage)[0]
            self.predict = tf.argmax(self.logits, axis=1)                           #  [batch]
            self.acc = tf.cast(tf.equal(tf.argmax(self.labels, 1), self.predict), tf.float32)
            self.accuracy = tf.identity(tf.reduce_sum(self.acc) / tf.to_float(self.batch_size), name="accuracy")
            tf.summary.scalar("accuracy", self.accuracy)
        return self.total_loss, self.accuracy

    def _optimizer(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        # learning rate decay
        # self.learning_rate = tf.Variable(tf.to_float(self.learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learning_rate = tf.minimum(self.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1 ))
        # self.learning_rate_decay_op = self.learning_rate.assign(
        #     self.learning_rate * self.learning_rate_decay_factor)
        tf.summary.scalar("learning-rate", self.learning_rate)
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.max_gradient_norm = 5.0
        # self.global_step = tf.Variable(0, trainable=False, name="global-step")
        # params = tf.trainable_variables()
        # gradients = tf.gradients(self.total_loss, params)
        # clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    @staticmethod
    def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
