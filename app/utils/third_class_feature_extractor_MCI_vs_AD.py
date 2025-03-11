from keras.models import load_model, Model
from keras.preprocessing import image
from keras import models
import numpy as np
import pandas as pd
import os
from keras import backend as K


def feature_extract_mci_vs_ad(data_path, model_name, save_path, model_path, step_num):
    model = load_model(model_path + '/' + model_name)

    flatten = model.get_layer('flatten')
    test_model = Model(inputs=model.input, outputs=flatten.output)

    ad_img_dir = data_path + '/' + 'AD'
    mci_img_dir = data_path + '/' + 'MCI'

    mci_img = []
    for img_name in os.listdir(mci_img_dir):
        img_features = []
        img_path = os.path.join(mci_img_dir, img_name)
        img = image.load_img(img_path, target_size=(100, 100))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        mci_img.append(img_tensor)
        img_features.append(test_model.predict(img_tensor)[0])
        img_features = np.append(img_features, np.array(img_name))
        img_features = np.append(img_features, np.array('MCI'))
        img_features_all.append(img_features)

    ad_img = []
    for img_name in os.listdir(ad_img_dir):
        img_features = []
        img_path = os.path.join(ad_img_dir, img_name)
        img = image.load_img(img_path, target_size=(100, 100))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        ad_img.append(img_tensor)
        img_features.append(test_model.predict(img_tensor)[0])
        img_features = np.append(img_features, np.array(img_name))
        img_features = np.append(img_features, np.array('AD'))
        img_features_all.append(img_features)

    #print(img_features_all)

    mci_labels = np.array([1 for i in range(len(mci_img))])
    ad_labels = np.array([0 for i in range(len(ad_img))])

    total_labels = np.concatenate((mci_labels, ad_labels), axis=0)


    # Dataset + Label
    feature_set = []
    for i in range(len(total_labels)):
        temp = np.append(img_features_all[i], total_labels[i])
        feature_set.append(temp)

    # Feature column name
    features_column = []
    for i in range(len(img_features_all[0])):
        features_column.append('img_f' + str(i) + '_' + str(step_num))

    features_column.append('AA')

    df = pd.DataFrame(feature_set, columns=features_column)
    if model_name[-7:-3] == '00.0':
        df.to_csv(save_path + '/img_features_MvsA-' + str(step_num) + '-' + model_name[-8:-3] + '.csv', index=False)
    else:
        df.to_csv(save_path + '/img_features_MvsA-' + str(step_num) + '-' + model_name[-7:-3] + '.csv', index=False)


# for step_no in [3, 4, 5, 6, 8]:

#     for k in ('train', 'validation','test'):
#         img_features_all = []
#         save_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-02-NIA-FeatSel-MvsA/01-FiveTasks/' + str(step_no) + '/' + k
#         model_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-01-TOTAL-MelImg-MvsA-h5/' + str(step_no)
#         data_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-01-TOTAL-MelImg-MvsA/' + str(step_no) + '/' + k

#         model_list = os.listdir(model_path)

#         for model_name in model_list:
#            if (model_name.find('h5') != -1):
#                feature_extract(data_path, model_name, save_path, model_path, step_no)
#                print(str(step_no) + ' is done')
#                K.clear_session()
