import numpy as np
from sklearn.preprocessing import StandardScaler
import shutil
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def get_global_data(data_path):
    data = np.loadtxt(data_path, dtype=np.float64, delimiter=',', encoding='utf-8')

    # data = data.iloc[:, ].values
    # 标准化处理
    standardScaler = StandardScaler()
    standardScaler.fit(data)
    data = standardScaler.transform(data)

    return data


def get_spectorm_data(data_path, heights=300, widths=40, n_classes=4, speech_nums=5531):
    data = np.loadtxt(data_path, dtype=np.float64, delimiter=',', encoding='utf-8')
    x_data, y_data = np.split(data, [heights * widths * 1, ], axis=1)

    # 数字标签转换成one hot标签
    y_data = y_data.astype(int) - 1
    one_hot_y_data = np.zeros(shape=(speech_nums, n_classes))
    one_hot_y_data[np.arange(0, speech_nums), y_data.reshape(speech_nums, )] = 1

    # 标准化处理
    standardScaler = StandardScaler()
    standardScaler.fit(x_data)
    x_data = standardScaler.transform(x_data)

    return x_data, one_hot_y_data


def get_spectorm_data_2(data_path, heights=300, widths=40):
    data = np.loadtxt(data_path, dtype=np.float64, delimiter=',', encoding='utf-8')
    x_data, Valence, Activation, Dominance = np.split(data, [heights * widths * 1, heights * widths + 1, heights * widths + 2], axis=1)

    # 标准化处理
    standardScaler = StandardScaler()
    standardScaler.fit(x_data)
    x_data = standardScaler.transform(x_data)

    return x_data, Valence, Activation, Dominance


def batch_iter(data, batch_size, shuffle=False):
    """
        Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    # 每个epoch的num_batch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def save_medel(sess, model_name):
    if os.path.exists(model_name):
        shutil.rmtree(model_name)

    builder = tf.saved_model.builder.SavedModelBuilder(model_name)
    builder.add_meta_graph_and_variables(sess, ["serve"])
    builder.save()


def plot_lost(train_costs=[], test_costs=[], epochs=40, filename="ABC"):
    plt.plot(np.arange(epochs), train_costs, color='r', label='train_costs')
    plt.plot(np.arange(epochs), test_costs, color='b', label='test_costs')
    plt.xlabel('epochs')
    plt.ylabel('cost_results')
    plt.legend(loc='best')
    plt.savefig(filename + ".jpg")
    plt.show()


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 01:02:58 2019

@author: jdang03
"""
import numpy as np


def concordance_correlation_coefficient(y_true, y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    # >>> from sklearn.metrics import concordance_correlation_coefficient
    # >>> y_true = [3, -0.5, 2, 7]
    # >>> y_pred = [2.5, 0.0, 2, 8]
    # >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


# n_samples = 1000
# y_true = np.arange(n_samples)
# y_true = y_true.astype(np.float32)
# y_pred = y_true + 500
# c = concordance_correlation_coefficient(y_true, y_pred)
# print(c)