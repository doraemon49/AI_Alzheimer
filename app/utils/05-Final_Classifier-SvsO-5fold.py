from keras import models
from keras.models import load_model
from keras import layers
from keras import backend as K
import numpy
import random
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

result_lst = []

# option = 1 : 학습 후 새로운 h5파일 생성, 전체 데이터를 Shuffle 하여 Train, Validation, Test로 나눔
# option = 2 : 학습 후 새로운 h5파일 생성, Train, Validation, Test 데이터를 각각 읽어서 실험 진행
# option = 3 : 학습된 h5파일을 load 후 predict

option = 3
category = 'SCI vs OTHERS'

# 반복횟수
iteration = 10000

if option == 1:
    for it in range(iteration):

        f_num = 2000
        dataset = numpy.loadtxt('block_total_features_SCI_Vs_Others.csv', delimiter=",", skiprows=1, encoding='utf-8')

        random.shuffle(dataset)

        train_index = len(dataset) * 0.6
        train_index = int(train_index)

        validation_index = len(dataset) * 0.8
        validation_index = int(validation_index)

        train_data = dataset[:train_index, :-1]
        train_labels = dataset[:train_index, -1]

        validation_data = dataset[train_index:validation_index, :-1]
        validation_labels = dataset[train_index:validation_index, -1]

        test_data = dataset[validation_index:, :-1]
        test_labels = dataset[validation_index:, -1]

        network = models.Sequential()
        network.add(layers.Dense(50, activation='relu', input_dim=f_num))
        network.add(layers.Dense(10, activation='relu'))
        network.add(layers.Dense(1, activation='sigmoid'))

        network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        network.fit(train_data, train_labels, epochs=100, verbose=0, validation_data=(validation_data, validation_labels))

        test_loss, test_acc = network.evaluate(test_data, test_labels)
        test_acc = round(test_acc, 3)
        result_lst.append(test_acc)

        print(category + ' test_acc :', test_acc)

        network.save(category + '_' + str(test_acc) + '.h5')

elif option == 2:
    flag = 1
    base_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-02-NIA-FeatSel-SvsO/01-FourTasks/Fold-5'
    while True:
        f_num = 18432

        if flag == 1:
            train_dataset = numpy.loadtxt(base_path + '/1-fold/f1_fourtasks_train_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            validation_dataset = numpy.loadtxt(base_path + '/1-fold/f1_fourtasks_val_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            test_dataset = numpy.loadtxt(base_path + '/1-fold/f1_fourtasks_test_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
        elif flag == 2:
            train_dataset = numpy.loadtxt(base_path + '/2-fold/f2_fourtasks_train_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            validation_dataset = numpy.loadtxt(base_path + '/2-fold/f2_fourtasks_val_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            test_dataset = numpy.loadtxt(base_path + '/2-fold/f2_fourtasks_test_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
        elif flag == 3:
            train_dataset = numpy.loadtxt(base_path + '/3-fold/f3_fourtasks_train_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            validation_dataset = numpy.loadtxt(base_path + '/3-fold/f3_fourtasks_val_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            test_dataset = numpy.loadtxt(base_path + '/3-fold/f3_fourtasks_test_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
        elif flag == 4:
            train_dataset = numpy.loadtxt(base_path + '/4-fold/f4_fourtasks_train_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            validation_dataset = numpy.loadtxt(base_path + '/4-fold/f4_fourtasks_val_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            test_dataset = numpy.loadtxt(base_path + '/4-fold/f4_fourtasks_test_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
        elif flag == 5:
            train_dataset = numpy.loadtxt(base_path + '/5-fold/f5_fourtasks_train_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            validation_dataset = numpy.loadtxt(base_path + '/5-fold/f5_fourtasks_val_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
            test_dataset = numpy.loadtxt(base_path + '/5-fold/f5_fourtasks_test_data.csv', delimiter=",", skiprows=1, encoding='utf-8')


        train_data = train_dataset[:, :-1]
        train_labels = train_dataset[:, -1]

        validation_data = validation_dataset[:, :-1]
        validation_labels = validation_dataset[:, -1]

        test_data = test_dataset[:, :-1]
        test_labels = test_dataset[:, -1]

        network = models.Sequential()
        network.add(layers.Dense(60, activation='relu', input_dim=f_num))
        network.add(layers.Dense(20, activation='relu'))
        network.add(layers.Dense(1, activation='sigmoid'))

        network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        network.fit(train_data, train_labels, epochs=200, verbose=0, validation_data=(validation_data, validation_labels))

        test_loss, test_acc = network.evaluate(test_data, test_labels)
        test_acc = round(test_acc, 3)
        result_lst.append(test_acc)

        print('SCI_vs_OTHERS_onlyNIA' + ' test_acc :', test_acc)

        if flag == 1:
            network.save(base_path + '/1-fold/f1_fourtasks_onlyNIA_' + str(test_acc) + '.h5')
            flag = 2
            K.clear_session()
        elif flag == 2:
            network.save(base_path + '/2-fold/f2_fourtasks_onlyNIA_' + str(test_acc) + '.h5')
            flag = 3
            K.clear_session()
        elif flag == 3:
            network.save(base_path + '/3-fold/f3_fourtasks_onlyNIA_' + str(test_acc) + '.h5')
            flag = 4
            K.clear_session()
        elif flag == 4:
            network.save(base_path + '/4-fold/f4_fourtasks_onlyNIA_' + str(test_acc) + '.h5')
            flag = 5
            K.clear_session()
        elif flag == 5:
            network.save(base_path + '/5-fold/f5_fourtasks_onlyNIA_' + str(test_acc) + '.h5')
            flag = 1
            K.clear_session()


elif option == 3:
    base_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-02-NIA-FeatSel-SvsO/01-FourTasks/Fold-5'
    f_num = 18432
    dataset = numpy.loadtxt(base_path + '/5-fold/f5_fourtasks_test_data.csv', delimiter=",", skiprows=1, encoding='utf-8')
    test_data_pd = pd.read_csv(base_path + '/5-fold/f5_fourtasks_test_data.csv')

    data = dataset[:, :-1]
    label = dataset[:, -1]

    test_model = load_model(base_path + '/5-fold/f5_fourtasks_onlyNIA_0.976.h5')

    test_loss, test_acc = test_model.evaluate(data, label)
    print(test_acc)
    r = 0
    d = 0
    for i in range((len(dataset))):
        if int(dataset[i, -1]) == test_model.predict_classes(data)[i][0] :
            r += 1
            print(int(dataset[i, -1]),test_model.predict_classes(data)[i][0], '맞았음')
        else:
            d += 1
            print(int(dataset[i, -1]), test_model.predict_classes(data)[i][0], '맞았음')
    print('맞은것 : ', r , '개 , 틀린거 : ' , d, '개')