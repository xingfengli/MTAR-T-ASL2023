import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
import numpy as np


# 初始化权重
def weight_variable(shape, name='weights'):
    # initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.get_variable(name, shape=shape, initializer=tf.initializers.he_normal())


# 初始化偏置
def biases_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)


# 卷积
def conv2d(x, filter_shape, padding='SAME'):
    return tf.nn.conv2d(x, filter_shape, strides=[1, 1, 1, 1], padding=padding)


# 池化层
def avg_pool(x, shape):
    return tf.nn.avg_pool(x, ksize=shape, strides=shape, padding='SAME')


# gelu激活函数
def gelu(inputs):
    cdf = 0.5*(1.0 + tf.erf(inputs/tf.sqrt(2.0)))
    return inputs*cdf


def asp_layer(inputs, keep_prob=0.5, training=True):
    inputs_shape = inputs.get_shape().as_list()
    if len(inputs_shape) == 4:
        inputs = tf.reshape(inputs, [-1, inputs_shape[1], inputs_shape[2] * inputs_shape[3]])
    inputs_shape = inputs.get_shape().as_list()

    w_u = tf.get_variable('w_u', [inputs_shape[-1], inputs_shape[-1]],
                          initializer=tf.initializers.truncated_normal(stddev=0.1))
    w_v = tf.get_variable('w_v', [inputs_shape[-1], ], initializer=tf.initializers.truncated_normal(stddev=0.1))
    b_u = tf.get_variable('b_u', [inputs_shape[-1], ], initializer=tf.initializers.constant(0.1))

    att_states = tf.nn.tanh(tf.tensordot(inputs, w_u, axes=1) + b_u)
    # att_states = tf.nn.dropout(att_states, keep_prob=keep_prob)
    att_coef = tf.tensordot(att_states, w_v, axes=1)
    att_coef = tf.nn.softmax(att_coef)
    # att_coef = tf.nn.dropout(att_coef, keep_prob=keep_prob)
    att_coef = tf.reshape(att_coef, [-1, inputs_shape[1], 1])
    outputs_attention_coef = tf.multiply(att_states, att_coef)
    outputs_attention = tf.reduce_sum(outputs_attention_coef, axis=1)
    outputs_attention = tf.reshape(outputs_attention, [-1, 1, inputs_shape[-1]])
    outputs_attention_std = tf.sqrt(tf.reduce_sum(tf.square(outputs_attention_coef - outputs_attention), axis=1))
    outputs_attention_std = tf.layers.batch_normalization(outputs_attention_std, training=training)

    return outputs_attention_std


def sp_layer(inputs):
    inputs_shape = inputs.get_shape().as_list()
    if len(inputs_shape) == 4:
        inputs = tf.reshape(inputs, [-1, inputs_shape[1], inputs_shape[2] * inputs_shape[3]])
    inputs_shape = inputs.get_shape().as_list()
    outputs_mean = tf.reduce_mean(inputs, axis=1)
    outputs_mean_re = tf.reshape(outputs_mean, [-1, 1, inputs_shape[-1]])
    outputs_std = tf.sqrt(tf.reduce_sum(tf.square(inputs-outputs_mean_re), axis=1))
    return outputs_std, outputs_mean


def attention(inputs):
    inputs_shape = inputs.get_shape().as_list()
    if len(inputs_shape) == 4:
        inputs = tf.reshape(inputs, [-1, inputs_shape[1], inputs_shape[2] * inputs_shape[3]])
    inputs_shape = inputs.get_shape().as_list()

    w_u = tf.get_variable('w_u', [inputs_shape[-1], inputs_shape[-1]],
                          initializer=tf.initializers.truncated_normal(stddev=0.1))
    w_v = tf.get_variable('w_v', [inputs_shape[-1], ], initializer=tf.initializers.truncated_normal(stddev=0.1))
    b_u = tf.get_variable('b_u', [inputs_shape[-1], ], initializer=tf.initializers.constant(0.1))

    att_states = tf.nn.tanh(tf.tensordot(inputs, w_u, axes=1) + b_u)
    att_coef = tf.tensordot(att_states, w_v, axes=1)
    att_coef = tf.nn.softmax(att_coef)

    att_coef = tf.reshape(att_coef, [-1, inputs_shape[1], 1])
    outputs_attention_coef = tf.multiply(att_states, att_coef)
    outputs_attention = tf.reduce_sum(outputs_attention_coef, axis=1)

    return outputs_attention


def ff(inputs, num_units, keep_prob=0.8, scope="feedforward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.elu)
        outputs = tf.nn.dropout(outputs, keep_prob)
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs += inputs

        # Normalize
        outputs = layer_normalize(outputs)

    return outputs


def layer_normalize(inputs, epsilon=1e-8, scope="ln"):
    # layer normalization
    with tf.variable_scope(scope):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta

    return outputs


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def multihead_attention(queries, keys, values, keep_prob=0.8, num_heads=8, num_units=None):

    # Set the fall back option for num_units
    if num_units is None:  # 如果num_units未指定，则使用query的最后一个维度作为num_units
        num_units = queries.get_shape().as_list()[-1]

    # Linear projections    线性映射 生成Q、K、V三个attention matrix
    Q = tf.layers.dense(queries, num_units, use_bias=True)  # (N, T_q, C)
    # Q = tf.nn.dropout(Q, keep_prob=keep_prob)
    K = tf.layers.dense(keys, num_units, use_bias=True)  # (N, T_k, C)
    # K = tf.nn.dropout(K, keep_prob=keep_prob)
    V = tf.layers.dense(values, num_units, use_bias=True)  # (N, T_k, C)
    # V = tf.nn.dropout(V, keep_prob=keep_prob)

    # Split and concat  将上步生成的attention matrix切分为num_heads个小矩阵，并在第一维度进行拼接
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # ###########################Key Masking,作用是什么
    # key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
    # key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)#，对key_masks在第一维度上进行扩展
    # # tf.expand_dims(key_masks, 1)增加一个维度，1代表第二维度
    # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
    #
    # paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    # # f.where(a,b,c)函数：
    # # 功能：当a输出结果为true时，tf.where(a,b,c)函数会选择b值输出。
    # # 当a输出结果为false时，tf.where(a,b,c)函数会选择c值输出。
    # outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

    # ###########################Query Masking,这个的作用又是什么
    # query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
    # query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    # outputs *= query_masks  # broadcasting. (N, T_q, C)

    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
    outputs = tf.nn.dropout(outputs, keep_prob)
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
    outputs += queries
    outputs = layer_normalize(outputs)  # (N, T_q, C)
    return outputs


def cnn(x, feature_map=[32, 64, 96, 128], training=True, keep_prob=0.5, name="CNN"):
    with tf.variable_scope(name):
        with tf.variable_scope("conv1"):
            # with tf.device('/gpu:1'):
            w_conv1 = weight_variable([5, 5, 1, feature_map[0]])
            b_conv1 = biases_variable([feature_map[0]])
            h_conv1 = conv2d(x, w_conv1) + b_conv1
            h_conv1 = tf.layers.batch_normalization(h_conv1, training=training)
            h_conv1 = tf.nn.leaky_relu(h_conv1)
            h_conv1 = tf.nn.dropout(h_conv1, keep_prob)
            h_pool1 = avg_pool(h_conv1, [1, 2, 2, 1])
            print(h_pool1)

        with tf.variable_scope("conv2"):
            # with tf.device('/gpu:1'):
            w_conv2 = weight_variable([3, 3, feature_map[0], feature_map[1]])
            b_conv2 = biases_variable([feature_map[1]])
            h_conv2 = conv2d(h_pool1, w_conv2) + b_conv2
            h_conv2 = tf.layers.batch_normalization(h_conv2, training=training)
            h_conv2 = tf.nn.leaky_relu(h_conv2)
            h_conv2 = tf.nn.dropout(h_conv2, keep_prob)
            h_pool2 = avg_pool(h_conv2, shape=[1, 2, 2, 1])
            print(h_pool2)

        with tf.variable_scope("conv3"):
            # with tf.device('/gpu:1'):
            w_conv3 = weight_variable([3, 3, feature_map[1], feature_map[2]])
            b_conv3 = biases_variable([feature_map[2]])
            h_conv3 = conv2d(h_pool2, w_conv3) + b_conv3
            h_conv3 = tf.layers.batch_normalization(h_conv3, training=training)
            h_conv3 = tf.nn.leaky_relu(h_conv3)
            h_conv3 = tf.nn.dropout(h_conv3, keep_prob)
            h_pool3 = avg_pool(h_conv3, shape=[1, 2, 2, 1])
            print(h_pool3)

        with tf.variable_scope("conv4"):
            w_conv4 = weight_variable([3, 3, feature_map[2], feature_map[3]])
            b_conv4 = biases_variable([feature_map[3]])
            h_conv4 = conv2d(h_pool3, w_conv4) + b_conv4
            h_conv4 = tf.layers.batch_normalization(h_conv4, training=training)
            h_conv4 = tf.nn.leaky_relu(h_conv4)
            h_conv4 = tf.nn.dropout(h_conv4, keep_prob)
            h_pool4 = avg_pool(h_conv4, shape=[1, 2, 2, 1])
            print(h_pool4)

        # with tf.variable_scope("conv5"):
        #     w_conv5 = weight_variable([3, 3, feature_map[3], feature_map[4]])
        #     b_conv5 = biases_variable([feature_map[4]])
        #     h_conv5 = conv2d(h_pool4, w_conv5) + b_conv5
        #     h_conv5 = tf.layers.batch_normalization(h_conv5, training=training)
        #     h_conv5 = tf.nn.relu(h_conv5)
        #     h_conv5 = tf.nn.dropout(h_conv5, keep_prob)
        #     print(h_conv5)
    return h_pool4


def ann(x_global, keep_prob=0.5, training=True):

    global_fc1 = tf.layers.dense(x_global, units=300, activation=None)
    global_fc1 = tf.layers.batch_normalization(global_fc1, training=training)
    global_fc1 = tf.nn.relu(global_fc1)
    global_fc1 = tf.nn.dropout(global_fc1, keep_prob)

    global_fc2 = tf.layers.dense(global_fc1, units=300, activation=None)
    global_fc2 = tf.layers.batch_normalization(global_fc2, training=training)
    global_fc2 = tf.nn.relu(global_fc2)
    global_fc2 = tf.nn.dropout(global_fc2, keep_prob)

    global_fc3 = tf.layers.dense(global_fc2, units=300, activation=None)
    global_fc3 = tf.layers.batch_normalization(global_fc3, training=training)
    global_fc3 = tf.nn.relu(global_fc3)
    global_fc3 = tf.nn.dropout(global_fc3, keep_prob)

    return global_fc2


def conv1d_block(inputs, sa_filters=[128, 128, 128], keep_prob=0.5, dilation_rate=[1, 2, 4], training=True, name="conv1"):
    inputs_shape = inputs.get_shape().as_list()
    if len(inputs_shape) == 4:
        inputs = tf.reshape(inputs, [-1, inputs_shape[1], inputs_shape[2] * inputs_shape[3]])

    # mel、spectrogram、vggish需要加dense层，不然效果会下降，proposed、mfcc不需要
    inputs = tf.layers.dense(inputs, units=sa_filters[0]*2, activation=None)
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

    with tf.variable_scope(name+"_1"):
        conv1_1 = tf.layers.conv1d(inputs=inputs, filters=sa_filters[0], kernel_size=5, padding="SAME",
                                   activation=None, dilation_rate=dilation_rate[0])
        conv1_1 = layer_normalize(conv1_1, scope="ln" + "_1" + name)
        conv1_1 = tf.nn.elu(conv1_1)
        conv1_1 = tf.keras.layers.SpatialDropout1D(1 - keep_prob)(conv1_1, training=training)
    with tf.variable_scope(name+"_2"):
        conv1_2 = tf.layers.conv1d(inputs=conv1_1, filters=sa_filters[1], kernel_size=3, padding="SAME",
                                   activation=None, dilation_rate=dilation_rate[1])
        conv1_2 = layer_normalize(conv1_2, scope="ln" + "_2" + name)
        conv1_2 = tf.nn.elu(conv1_2)
        conv1_2 = tf.keras.layers.SpatialDropout1D(1 - keep_prob)(conv1_2, training=training)
    with tf.variable_scope(name+"_3"):
        conv1_3 = tf.layers.conv1d(inputs=conv1_2, filters=sa_filters[2], kernel_size=3, padding="SAME",
                                   activation=None, dilation_rate=dilation_rate[2])
        conv1_3 = layer_normalize(conv1_3, scope="ln" + "_3" + name)
        conv1_3 = tf.nn.elu(conv1_3)
        conv1_3 = tf.keras.layers.SpatialDropout1D(1 - keep_prob)(conv1_3, training=training)

    if inputs.get_shape().as_list()[-1] != sa_filters[2]:
        conv1 = tf.layers.conv1d(inputs=inputs, filters=sa_filters[2], kernel_size=1, padding="SAME",
                                 activation=None)
        conv1 = layer_normalize(conv1, scope="ln" + name)
        conv1 = tf.nn.elu(conv1)
        return conv1_3+conv1
    else:
        return conv1_3


def transformer(inputs, num_blocks=2, keep_prob=0.5, num_heads=8, num_units=[512, 256]):
    inputs_shape = inputs.get_shape().as_list()
    if len(inputs_shape) == 4:
        inputs = tf.reshape(inputs, [-1, inputs_shape[1], inputs_shape[2] * inputs_shape[3]])

    inputs = tf.layers.dense(inputs, units=num_units[1], activation=tf.nn.elu)
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
    # inputs += positional_encoding(inputs, inputs_shape[1], masking=False)

    global_feature = inputs
    for i in range(num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
            # self-attention
            global_feature = multihead_attention(queries=global_feature,
                                                 keys=global_feature,
                                                 values=global_feature,
                                                 keep_prob=keep_prob,
                                                 num_heads=num_heads)
            # feed forward
            global_feature = ff(global_feature, num_units=num_units, keep_prob=keep_prob)
    return global_feature
