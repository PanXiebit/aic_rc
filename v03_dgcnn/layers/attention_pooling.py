import tensorflow as tf

def attention_pooling(enc_query, fusion_out, enc_answers, num_filters, initializer):
    with tf.variable_scope("query-attention"):
        Wq = tf.layers.dense(enc_query, units=num_filters,
                             kernel_initializer=initializer)
        # Wq = layer_norm(Wq)
        Wq = tf.nn.tanh(Wq)
        sj = tf.transpose(tf.layers.dense(Wq, units=1,
                                          kernel_initializer=initializer), (0, 2, 1))  # [batch, 1, max_query]
        weights_q = tf.nn.softmax(sj, axis=-1, name="weights_q")
        attened_query = tf.matmul(weights_q, enc_query)  # [batch, 1, num_filter]
        attened_query = tf.identity(attened_query, name="attened_auery")

    with tf.variable_scope("fusion-attention"):
        sf = tf.layers.dense(fusion_out, num_filters,
                             kernel_initializer=initializer) \
             + tf.layers.dense(attened_query, num_filters,
                               kernel_initializer=initializer)
        # sf = layer_norm(sf)
        sf = tf.nn.tanh(sf)
        sf = tf.transpose(tf.layers.dense(sf, units=1, kernel_initializer=initializer),
                          (0, 2, 1))  # [batch, 1, max_passage]
        weights_f = tf.nn.softmax(sf, axis=-1, name="weights_f")
        attened_fusion = tf.matmul(weights_f, fusion_out)  # [batch, 1, num_filters]
        attened_fusion = tf.identity(attened_fusion, name="attened_fusion")

    with tf.variable_scope("answer-attention"):
        # answer attention pooling
        answers_enc = tf.layers.dense(enc_answers,
                                      units=num_filters,
                                      kernel_initializer=initializer)  # [batch*3, max_answer, num_filters]
        # answers_enc = layer_norm(answers_enc)
        answers_enc = tf.nn.tanh(answers_enc)
        answer_score = tf.nn.softmax(tf.layers.dense(answers_enc, units=1,
                                                     kernel_initializer=initializer),
                                     axis=1)                                          # [batch*3, max_answer, 1]
        answers_attened = tf.matmul(answer_score, enc_answers, transpose_a=True)
        answers_attened = tf.concat(tf.split(answers_attened, 3, axis=0), axis=1)     # [-1, 3, num_filters]
        attened_answer = tf.identity(answers_attened, name="attened_answer")          # [batch, 3, num_filters]

    cosine = tf.squeeze(tf.matmul(attened_answer, attened_fusion, transpose_b=True))  # [batch, 3]
    denom = (tf.reduce_sum(tf.square(attened_answer), axis=2, keepdims=False) *
             tf.reduce_sum(tf.square(attened_fusion), axis=2, keepdims=False) + 1e-6)
    logits = tf.identity(cosine/denom, name="logits")                                 # [batch, 3]
    return logits
