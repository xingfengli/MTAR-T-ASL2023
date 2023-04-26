# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import os

ms_len = 300
n_mels = 80
zero_padding = np.zeros([50, n_mels])


if __name__ == '__main__':
    data_path = "./proposed_csvIEMOCAP_80_reZip/"
    label_path = "./label/"
    spectorm_data = []
    label = []
    label_file = pd.read_excel(label_path + "IEMOCAP_9946.xlsx")
    label_file = label_file.sort_values(by="speaker", ascending=True)

    ########### 分类数据
    # label_file = label_file[label_file["Cat_Type"] >= 1]
    # label_file = label_file[label_file["Cat_Type"] <= 5]
    # print(label_file)
    # print(label_file.shape)
    # label_file.loc[label_file["Cat_Type"] == 5, "Cat_Type"] = 2
    # label = np.array(label_file["Cat_Type"])
    # print(label)

    data_list = list(label_file["UID"].values)
    for j, data_file in enumerate(data_list):
        print(data_file)
        data = pd.read_csv(data_path + data_file + ".csv", header=None, index_col=False)
        data = np.transpose(data.values)

        while len(data) < ms_len:
             data = np.concatenate([data, zero_padding], axis=0)
        # data = data[:ms_len, :68]
        data = data[:ms_len]
        data = np.reshape(data, (ms_len*n_mels,))
        spectorm_data.append(data.tolist())
        print(len(spectorm_data))

    ########## 回归数据
    valence = np.array(label_file["Valence"])
    activation = np.array(label_file["Activation"])
    dominance = np.array(label_file["Dominance"])
    valence = np.array(valence).reshape([len(valence), 1])
    activation = np.array(activation).reshape([len(activation), 1])
    dominance = np.array(dominance).reshape([len(dominance), 1])

    spectorm_data = np.concatenate([np.array(spectorm_data), valence, activation, dominance], axis=1)
    print(spectorm_data.shape)
    np.savetxt('IEMOCAP_proposed_vad.csv', spectorm_data, delimiter=',')

    ######### 分类数据
    # label = np.array(label).reshape([len(label), 1])
    #
    # spectorm_data = np.concatenate([np.array(spectorm_data), label], axis=1)
    # print(spectorm_data.shape)
    # np.savetxt('IEMOCAP_proposed_classification.csv', spectorm_data, delimiter=',')




