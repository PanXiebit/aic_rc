
import tensorflow as tf
from tensorflow.python.ops import nn_ops

def match(passage_enc, query_enc, passage_len, query_len, mask=True):
    """
    Match the passage_encodes with question_encodes using Attention Flow Match algorithm
    :param passage_enc:  [batch, max_passage, hidden_size]
    :param query_enc: [batch, max_query, hidden_size]
    :param passage_len:
    :param query_len:
    :return:
    """
    # Dureader's implement, didn't ponder the padding while attention.
    # share the same similary matrix
    sim_matrix = tf.matmul(passage_enc, query_enc, transpose_b=True) # [batch, max_passage, max_query]
    # content-to-query
    query_aware_weight = tf.nn.softmax(sim_matrix, axis=-1)
    content2query_att = tf.matmul(query_aware_weight, query_enc)     # [batch, max_passage, hidden_size]

    # query-to-content
    # find the most relevant word in content
    content_aware_weight = tf.expand_dims(tf.nn.softmax(tf.reduce_max(sim_matrix, axis=2), axis=-1), axis=1) # [batch, 1, max_passage]
    query2content_att = tf.matmul(content_aware_weight, passage_enc)                  # [batch, 1, hidden_size]
    query2content_att = tf.tile(query2content_att, [1, tf.shape(passage_enc)[1], 1])  # [batch, max_passage, hidden_size]
    concat_outputs = tf.concat([passage_enc, content2query_att,
                                passage_enc * content2query_att,
                                passage_enc * query2content_att], axis=-1)
    return concat_outputs


def match2(passage_enc, query_enc, passage_len=None, query_len=None,
           regularizer=None, kernel_initializer=None):
    """

    :param passage_enc: [batch, passage_len, hidden_size]
    :param query_enc:   [batch, query_len, hidden_size]
    :param passage_len: [batch]
    :param query_len:   [batch]
    :param passage_enc_state:  [batch, hidden_size]
    :param query_enc_state:    [batch, hidden_size]
    :return:
    """
    # concat
    passage_shape = passage_enc.get_shape().as_list()
    query_shape = query_enc.get_shape().as_list()
    if len(passage_shape) != 3 or len(query_shape) != 3:
        raise ValueError("`passage_enc and query_enc` must be 3 dims (batch_size, len, dimension)")
    if passage_shape[2] != query_shape[2]:
         raise ValueError("the last dimension of `args` must equal")
    dim = passage_shape[-1]
    passage_len = tf.shape(passage_enc)[1]
    query_len = tf.shape(query_enc)[1]
    with tf.variable_scope("bi-attention"):
        weights4arg0 = tf.get_variable(
                        "linear_kernel4arg0", [dim, 1],
                        regularizer=regularizer,
                        initializer=kernel_initializer
        )
        weights4arg1 = tf.get_variable(
                        "linear_kernel4arg1", [dim, 1],
                        regularizer=regularizer,
                        initializer=kernel_initializer
        )
        weights4mul = tf.get_variable(
                        "linear_kernel4mul", [1, 1,dim],
                        regularizer=regularizer,
                        initializer=kernel_initializer
        )
        biases = tf.get_variable(
                        "linear_bias", [1],
                        regularizer=regularizer,
                        initializer=tf.zeros_initializer()
        )
        subres0 = tf.tile(tf.einsum('aij,jk->aik',passage_enc,  weights4arg0), (1,1, query_len))
        subres1 = tf.transpose(tf.tile(tf.einsum('aij,jk->aik',query_enc, weights4arg1), (1,1, passage_len)),(0,2,1))
        subres2 = tf.matmul(passage_enc*weights4mul, query_enc, transpose_b=True)
        res = subres0 + subres1 + subres2
        # res = tf.nn.bias_add(res, biases)
        return res

if __name__ == "__main__":
    passage = tf.random_normal([5, 10, 32])
    query = tf.random_normal([5, 3, 32])
    output = match2(passage, query)
    print(output)