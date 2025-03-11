import os
import numpy as np
import pandas as pd
import csv


base_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-02-NIA-FeatSel-SvsO/01-ThreeTasks'
save_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-02-NIA-FeatSel-SvsO/01-ThreeTasks/S189-O216-210215-threetasks.'

file_name_01 = base_path + '/img_features_SvsO-1-72.7-210215-sort-nia-same.csv'
file_name_03 = base_path + '/img_features_SvsO-3-69.3-210215-sort-nia-same.csv'
#file_name_04 = base_path + '/img_features_SvsO-4-68.1-210215-sort-nia-same.csv'
file_name_08 = base_path + '/img_features_SvsO-8-69.1-210215-sort-nia-same.csv'


dataset_array_all = []

dataset_array = []
data_pd = pd.read_csv(file_name_01)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = dataset_array
print('dataset_array_all 1',  dataset_array_all.shape)


dataset_array = []
data_pd = pd.read_csv(file_name_03)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
print('dataset_array_all 3',  dataset_array_all.shape)

"""
dataset_array = []
data_pd = pd.read_csv(file_name_04)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
print('dataset_array_all 4',  dataset_array_all.shape)
"""

dataset_array = []
data_pd = pd.read_csv(file_name_08)
dataset_array = np.array(data_pd.iloc[:, 0:4608])
print(dataset_array.shape)
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
print('dataset_array_all 8', dataset_array_all.shape)


dataset_array = np.array(data_pd.iloc[:, 4610:4611])
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)

df = pd.DataFrame(dataset_array_all)
df.to_csv(save_path + 'csv', index=False)

"""cnt = 1
for file_name in file_list:
    dataset_array = []
    if file_name.find('csv') is not -1:
        file_path = base_path + '/' + file_name
        data_pd = pd.read_csv(file_path)
        #data_pd.iloc[:, 0:4607]
        dataset_array = np.array(data_pd.iloc[:, 0:4608])
        print(dataset_array.shape)
        if cnt == 1:
            dataset_array_all = dataset_array
            print('dataset_array_all ',cnt, ' ',  dataset_array_all.shape)
        else :
            dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)
            print('dataset_array_all ',cnt, ' ',  dataset_array_all.shape)
        cnt += 1
dataset_array = np.array(data_pd.iloc[:, 4608:4611])
dataset_array_all = np.append(dataset_array_all, dataset_array, axis = 1)

df = pd.DataFrame(dataset_array_all)
df.to_csv(save_path + 'csv', index=False)"""


