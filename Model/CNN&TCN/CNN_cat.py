# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:42:57 2021

@author: Administrator
"""

import tensorflow as tf
import numpy as np
from utils import get_spectorm_data, batch_iter, plot_lost
from nets import cnn, sp_layer
from sklearn import metrics
import os

tf.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

lr = 0.0005
epochs = 40
batch_size = 32
cnn_dropout_lr = 0.8

heights = 300
widths = 120
c1_units = 200
dnn_dropout_lr = 0.8
# c2_units=200

speech_nums = 5473
n_classes = 4
random_seed = 10
feature_map = [32, 64, 96, 128]
# l2_rate=0.001

lr_v = tf.Variable(lr, dtype=tf.float32, trainable=False)

data_path = "./IEMOCAP_mel_new.csv"
x_data, one_hot_y_data, _ = get_spectorm_data(data_path, heights=heights, widths=widths, n_classes=n_classes, speech_nums=speech_nums)

x = tf.placeholder(tf.float32, [None, heights, widths, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
bn_training = tf.placeholder(dtype=tf.bool)
cnn_keep_prob = tf.placeholder(tf.float32)
dnn_keep_prob = tf.placeholder(tf.float32)

cnn_output = cnn(x, feature_map, bn_training, cnn_keep_prob)
_, outputs_mean = sp_layer(cnn_output)

fc1 = tf.layers.dense(outputs_mean, units=c1_units, activation=tf.nn.relu)
fc1 = tf.nn.dropout(fc1, dnn_keep_prob)

logits = tf.layers.dense(fc1, units=n_classes, activation=None)
prediction = tf.nn.softmax(logits)

# regularzer=tf.contrib.layers.l2_regularizer(l2_rate)
# tf.contrib.layers.apply_regularization(regularzer,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
# regularizatuin_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

# loss = tf.reduce_mean(focal_loss_softmax(labels=y,logits=logits,gamma=2.5))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
# loss += regularizatuin_loss

# 增加BN层配合使用代码
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # train_step = tf.train.MomentumOptimizer(lr_v,0.9).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr_v).minimize(loss)
y_pred = tf.arg_max(prediction, 1)
correct_prediction = tf.equal(y_pred, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
# saver = tf.train.Saver()

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

with tf.Session(config=config) as sess:
    x_train_1, x_val_1 = x_data[512:], x_data[:512]
    y_train_1, y_val_1 = one_hot_y_data[512:], one_hot_y_data[:512]
    x_train_2, x_val_2 = np.concatenate([x_data[:512], x_data[1059:]], axis=0), x_data[512:1059]
    y_train_2, y_val_2 = np.concatenate([one_hot_y_data[:512], one_hot_y_data[1059:]], axis=0), one_hot_y_data[512:1059]
    x_train_3, x_val_3 = np.concatenate([x_data[:1059], x_data[1535:]], axis=0), x_data[1059:1535]
    y_train_3, y_val_3 = np.concatenate([one_hot_y_data[:1059], one_hot_y_data[1535:]], axis=0), one_hot_y_data[
                                                                                                 1059:1535]
    x_train_4, x_val_4 = np.concatenate([x_data[:1535], x_data[2075:]], axis=0), x_data[1535:2075]
    y_train_4, y_val_4 = np.concatenate([one_hot_y_data[:1535], one_hot_y_data[2075:]], axis=0), one_hot_y_data[
                                                                                                 1535:2075]
    x_train_5, x_val_5 = np.concatenate([x_data[:2075], x_data[2596:]], axis=0), x_data[2075:2596]
    y_train_5, y_val_5 = np.concatenate([one_hot_y_data[:2075], one_hot_y_data[2596:]], axis=0), one_hot_y_data[
                                                                                                 2075:2596]
    x_train_6, x_val_6 = np.concatenate([x_data[:2596], x_data[3223:]], axis=0), x_data[2596:3223]
    y_train_6, y_val_6 = np.concatenate([one_hot_y_data[:2596], one_hot_y_data[3223:]], axis=0), one_hot_y_data[
                                                                                                 2596:3223]
    x_train_7, x_val_7 = np.concatenate([x_data[:3223], x_data[3745:]], axis=0), x_data[3223:3745]
    y_train_7, y_val_7 = np.concatenate([one_hot_y_data[:3223], one_hot_y_data[3745:]], axis=0), one_hot_y_data[
                                                                                                 3223:3745]
    x_train_8, x_val_8 = np.concatenate([x_data[:3745], x_data[4248:]], axis=0), x_data[3745:4248]
    y_train_8, y_val_8 = np.concatenate([one_hot_y_data[:3745], one_hot_y_data[4248:]], axis=0), one_hot_y_data[
                                                                                                 3745:4248]
    x_train_9, x_val_9 = np.concatenate([x_data[:4248], x_data[4826:]], axis=0), x_data[4248:4826]
    y_train_9, y_val_9 = np.concatenate([one_hot_y_data[:4248], one_hot_y_data[4826:]], axis=0), one_hot_y_data[
                                                                                                 4248:4826]
    x_train_10, x_val_10 = x_data[:4826], x_data[4826:]
    y_train_10, y_val_10 = one_hot_y_data[:4826], one_hot_y_data[4826:]

    x_train_data = [x_train_1, x_train_2, x_train_3, x_train_4, x_train_5, x_train_6, x_train_7, x_train_8, x_train_9,
                    x_train_10]
    y_train_data = [y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7, y_train_8, y_train_9,
                    y_train_10]

    x_val_data = [x_val_1, x_val_2, x_val_3, x_val_4, x_val_5, x_val_6, x_val_7, x_val_8, x_val_9, x_val_10]
    y_val_data = [y_val_1, y_val_2, y_val_3, y_val_4, y_val_5, y_val_6, y_val_7, y_val_8, y_val_9, y_val_10]
    acc_all = []
    test_y_pred_best_all = []
    for k in range(10):
        print("###########第{}折###########".format(k))
        sess.run(init)
        train_costs = []
        test_costs = []
        test_acc_best = 0
        test_UA_best = 0
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
                test_lost, test_prediction = sess.run([loss, y_pred],
                                                      feed_dict={x: np.array(test_x_batch).reshape(-1, heights, widths, 1),
                                                                 y: np.array(test_y_batch),
                                                                 bn_training: False,
                                                                 cnn_keep_prob: 1.0,
                                                                 dnn_keep_prob: 1.0
                                                                 })
                test_lost_sum.append(test_lost)
                test_y_pred.extend(test_prediction)
                y_val.extend(test_y_batch)
            test_correct_prediction = np.array(test_y_pred) == np.array(y_val).argmax(1)
            test_acc = np.mean(test_correct_prediction)
            test_costs.append(sum(test_lost_sum) / len(test_lost_sum))
            UA_result = metrics.recall_score(np.array(y_val).argmax(1), np.array(test_y_pred), average='macro')
            if UA_result > test_UA_best:
                test_acc_best = test_acc
                test_UA_best = UA_result
                test_y_pred_best = test_y_pred
            print("epoch:" + str(epoch) + " Test accuracy:" + str(test_acc))
            print("epoch:" + "UA: " + str(UA_result))
        acc_all.append(test_acc_best)
        print("epoch:" + "test_acc_best: " + str(test_acc_best))
        test_y_pred_best_all.extend(test_y_pred_best)
        # 混淆矩阵
        print(metrics.confusion_matrix(np.array(y_val).argmax(1), test_y_pred_best))
        # 计算召回率
        print(metrics.recall_score(np.array(y_val).argmax(1), test_y_pred_best, average=None))
        # 计算平均召回率
        print(metrics.recall_score(np.array(y_val).argmax(1), test_y_pred_best, average='macro'))

        # plot_lost(train_costs, test_costs, epochs)
    print("十折准确率：")
    print(acc_all)
    print("十折平均准确率：{}".format(sum(acc_all) / len(acc_all)))
    print("总混淆矩阵:")
    print(metrics.confusion_matrix(np.array(one_hot_y_data).argmax(1), test_y_pred_best_all))
    print(np.array(metrics.confusion_matrix(np.array(one_hot_y_data).argmax(1), test_y_pred_best_all)).sum(axis=0).tolist())
    print(metrics.recall_score(np.array(one_hot_y_data).argmax(1), test_y_pred_best_all, average=None))
    print(metrics.recall_score(np.array(one_hot_y_data).argmax(1), test_y_pred_best_all, average='macro'))
