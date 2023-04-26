# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:42:57 2021

@author: Administrator
"""

import tensorflow as tf
import numpy as np
from utils import batch_iter, plot_lost, get_spectorm_data_2
from nets import cnn, conv1d_block, sp_layer
from concordance_correlation_coefficient import concordance_correlation_coefficient
import os


tf.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

lr = 0.0005
epochs = 40
batch_size = 32
cnn_dropout_lr = 0.8

heights = 300
widths = 120

tcn_filter = [196, 196, 196]
tcn_dropout_lr = 0.8

c1_units = 200
dnn_dropout_lr = 0.8
# c2_units=200

speech_nums = 9946
n_classes = 3
random_seed = 10
feature_map = [32, 64, 96, 128]
# l2_rate=0.001

lr_v = tf.Variable(lr, dtype=tf.float32, trainable=False)

data_path = "./IEMOCAP_mel_VAD.csv"
x_data, Valence, Activation, Dominance = get_spectorm_data_2(data_path, heights=heights, widths=widths)

x = tf.placeholder(tf.float32, [None, heights, widths, 1])
y = tf.placeholder(tf.float32, [None, 1])
bn_training = tf.placeholder(dtype=tf.bool)
cnn_keep_prob = tf.placeholder(tf.float32)
tcn_keep_prob = tf.placeholder(tf.float32)
dnn_keep_prob = tf.placeholder(tf.float32)

cnn_output = cnn(x, feature_map, bn_training, cnn_keep_prob)
tcn_output = conv1d_block(cnn_output, tcn_filter, tcn_keep_prob, dilation_rate=[1, 2, 4], training=bn_training, name="conv1")
_, outputs_mean = sp_layer(tcn_output)

fc1 = tf.layers.dense(outputs_mean, units=c1_units, activation=tf.nn.elu)
fc1 = tf.nn.dropout(fc1, dnn_keep_prob)

output = tf.layers.dense(fc1, units=1, activation=None)

# regularzer=tf.contrib.layers.l2_regularizer(l2_rate)
# tf.contrib.layers.apply_regularization(regularzer,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
# regularizatuin_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

loss = tf.losses.mean_squared_error(y, output)
# loss += regularizatuin_loss

# 增加BN层配合使用代码
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # train_step = tf.train.MomentumOptimizer(lr_v,0.9).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr_v).minimize(loss)
# y_pred = tf.arg_max(prediction, 1)
# correct_prediction = tf.equal(y_pred, tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# per_process_gpu_memory_fraction=0.4
gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

with tf.Session(config=config) as sess:
    x_train_1, x_val_1 = x_data[850:], x_data[:850]
    y_train_1, y_val_1 = Activation[850:], Activation[:850]
    x_train_2, x_val_2 = np.concatenate([x_data[:850], x_data[1774:]], axis=0), x_data[850:1774]
    y_train_2, y_val_2 = np.concatenate([Activation[:850], Activation[1774:]], axis=0), Activation[850:1774]
    x_train_3, x_val_3 = np.concatenate([x_data[:1774], x_data[2628:]], axis=0), x_data[1774:2628]
    y_train_3, y_val_3 = np.concatenate([Activation[:1774], Activation[2628:]], axis=0), Activation[1774:2628]
    x_train_4, x_val_4 = np.concatenate([x_data[:2628], x_data[3577:]], axis=0), x_data[2628:3577]
    y_train_4, y_val_4 = np.concatenate([Activation[:2628], Activation[3577:]], axis=0), Activation[2628:3577]
    x_train_5, x_val_5 = np.concatenate([x_data[:3577], x_data[4620:]], axis=0), x_data[3577:4620]
    y_train_5, y_val_5 = np.concatenate([Activation[:3577], Activation[4620:]], axis=0), Activation[3577:4620]
    x_train_6, x_val_6 = np.concatenate([x_data[:4620], x_data[5703:]], axis=0), x_data[4620:5703]
    y_train_6, y_val_6 = np.concatenate([Activation[:4620], Activation[5703:]], axis=0), Activation[4620:5703]
    x_train_7, x_val_7 = np.concatenate([x_data[:5703], x_data[6684:]], axis=0), x_data[5703:6684]
    y_train_7, y_val_7 = np.concatenate([Activation[:5703], Activation[6684:]], axis=0), Activation[5703:6684]
    x_train_8, x_val_8 = np.concatenate([x_data[:6684], x_data[7797:]], axis=0), x_data[6684:7797]
    y_train_8, y_val_8 = np.concatenate([Activation[:6684], Activation[7797:]], axis=0), Activation[6684:7797]
    x_train_9, x_val_9 = np.concatenate([x_data[:7797], x_data[8815:]], axis=0), x_data[7797:8815]
    y_train_9, y_val_9 = np.concatenate([Activation[:7797], Activation[8815:]], axis=0), Activation[7797:8815]
    x_train_10, x_val_10 = x_data[:8815], x_data[8815:]
    y_train_10, y_val_10 = Activation[:8815], Activation[8815:]

    x_train_data = [x_train_1, x_train_2, x_train_3, x_train_4, x_train_5, x_train_6, x_train_7, x_train_8, x_train_9,
                    x_train_10]
    y_train_data = [y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7, y_train_8, y_train_9,
                    y_train_10]

    x_val_data = [x_val_1, x_val_2, x_val_3, x_val_4, x_val_5, x_val_6, x_val_7, x_val_8, x_val_9, x_val_10]
    y_val_data = [y_val_1, y_val_2, y_val_3, y_val_4, y_val_5, y_val_6, y_val_7, y_val_8, y_val_9, y_val_10]
    ccc_all = []
    test_y_pred_best_all = []
    for k in range(10):
        print("###########第{}折###########".format(k))
        sess.run(init)
        train_costs = []
        test_costs = []
        test_ccc_best = 0
        test_y_pred_best = []

        for epoch in range(epochs):
            if (epoch + 1) % 30 == 0:
                sess.run(tf.assign(lr_v, lr_v / 10))
            batches = batch_iter(list(zip(x_train_data[k], y_train_data[k])), batch_size, shuffle=True)
            train_lost_sum = []
            for batch in batches:
                # zip(*batch)，将zip对象变成原先组合前的数据。
                x_batch, y_batch = zip(*batch)
                train_result = sess.run([train_step, loss],
                                        feed_dict={x: np.array(x_batch).reshape(-1, heights, widths, 1),
                                                   y: np.array(y_batch),
                                                   bn_training: True,
                                                   cnn_keep_prob: cnn_dropout_lr,
                                                   tcn_keep_prob: tcn_dropout_lr,
                                                   dnn_keep_prob: dnn_dropout_lr,
                                                   })
                train_lost_sum.append(train_result[1])
            train_costs.append(sum(train_lost_sum) / len(train_lost_sum))
            batches_test = batch_iter(list(zip(x_val_data[k], y_val_data[k])), 50)
            test_lost_sum = []
            test_y_pred = []
            y_val = []
            for batch in batches_test:
                # zip(*batch)，将zip对象变成原先组合前的数据。
                test_x_batch, test_y_batch = zip(*batch)
                test_lost, test_prediction = sess.run([loss, output],
                                                      feed_dict={x: np.array(test_x_batch).reshape(-1, heights, widths, 1),
                                                                 y: np.array(test_y_batch),
                                                                 bn_training: False,
                                                                 cnn_keep_prob: 1.0,
                                                                 tcn_keep_prob: 1.0,
                                                                 dnn_keep_prob: 1.0
                                                                 })
                test_lost_sum.append(test_lost)
                test_y_pred.extend(test_prediction)
                y_val.extend(test_y_batch)
            test_costs.append(sum(test_lost_sum) / len(test_lost_sum))
            ccc = concordance_correlation_coefficient(np.array(y_val).reshape([len(y_val), ]), np.array(test_y_pred).reshape([len(test_y_pred), ]))
            if test_ccc_best < ccc:
                test_ccc_best = ccc
                test_y_pred_best = test_y_pred
            print("epoch:" + str(epoch) + " CCC: " + str(ccc))

        ccc_all.append(test_ccc_best)
        print("epoch:" + "test_acc_best: " + str(test_ccc_best))
        test_y_pred_best_all.extend(test_y_pred_best)

        plot_lost(train_costs, test_costs, epochs)
    print("十折CCC：")
    print(ccc_all)
    print("十折平均CCC：{}".format(sum(ccc_all) / len(ccc_all)))
    print("总CCC:")
    print(Activation.shape)
    print(np.array(test_y_pred_best_all).shape)
    print(concordance_correlation_coefficient(Activation.reshape([len(Activation), ]),
                                              np.array(test_y_pred_best_all).reshape([len(test_y_pred_best_all), ])))

