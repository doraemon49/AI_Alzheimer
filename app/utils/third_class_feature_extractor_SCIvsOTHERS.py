from keras.models import load_model, Model
from keras.preprocessing import image
from keras import models
import numpy as np
import pandas as pd
import os
from keras import backend as K


def feature_extract_sci_vs_others(data_path, model_name, save_path, model_path, step_num):
    model = load_model(model_path + '/' + model_name)
    # flatten = model.get_layer('flatten')
    # test_model = Model(inputs=model.input, outputs=flatten.output)

    print(model.summary())  # 모델 구조 출력 
    # # Sequential 모델일 경우 model.input이 정의되지 않을 수 있습니다.
    # # 이때, 샘플 데이터를 모델에 한 번 전달하여 입력을 설정하면 해결됩니다
    # # Sequential 모델일 경우, model.input 대신 model.layers[0].input 사용
    # inputs = model.layers[0].input  # 첫 번째 레이어의 input을 가져오기
    # outputs = model.get_layer('flatten').output  # Flatten 레이어의 출력 가져오기

    # # Feature extractor 모델 생성
    # test_model = Model(inputs=inputs, outputs=outputs)




    # sci_img_dir = data_path + '/' + 'SCI'
    # others_img_dir = data_path + '/' + 'OTHERS'

    # kk = 'SCI'
    # sci_img = []
    # for img_name in os.listdir(sci_img_dir):
    #     img_features = []
    #     img_path = os.path.join(sci_img_dir, img_name)
    #     img = image.load_img(img_path, target_size=(100, 100))
    #     img_tensor = image.img_to_array(img)
    #     img_tensor = np.expand_dims(img_tensor, axis=0)
    #     img_tensor /= 255.
    #     sci_img.append(img_tensor)
    #     img_features.append(test_model.predict(img_tensor)[0])
    #     img_features = np.append(img_features, np.array(img_name))
    #     img_features = np.append(img_features, np.array(kk))
    #     img_features_all.append(img_features)

    # kk = 'OTHERS'
    # others_img = []
    # for img_name in os.listdir(others_img_dir):
    #     img_features = []
    #     img_path = os.path.join(others_img_dir, img_name)
    #     img = image.load_img(img_path, target_size=(100, 100))
    #     img_tensor = image.img_to_array(img)
    #     img_tensor = np.expand_dims(img_tensor, axis=0)
    #     img_tensor /= 255.
    #     others_img.append(img_tensor)
    #     img_features.append(test_model.predict(img_tensor)[0])
    #     img_features = np.append(img_features, np.array(img_name))
    #     img_features = np.append(img_features, np.array(kk))
    #     img_features_all.append(img_features)

    # #print(img_features_all)


    # sci_labels = np.array([1 for i in range(len(sci_img))])
    # others_labels = np.array([0 for i in range(len(others_img))])

    # total_labels = np.concatenate((sci_labels, others_labels), axis=0)

    # # Dataset + Label
    # feature_set = []
    # for i in range(len(total_labels)):
    #     temp = np.append(img_features_all[i], total_labels[i])
    #     feature_set.append(temp)

    # # Feature column name
    # features_column = []
    # for i in range(len(img_features_all[0])):
    #     features_column.append('img_f' + str(i) + '_' + str(step_num))

    # features_column.append('AA')



    # """
    # feature_set_sort = []

    # data_pd = pd.read_csv('Files-SvsO-Classify.csv', header=None)

    # cnt = 0
    # for line in range(1, len(data_pd[0][1:]) + 1):
    #     file = data_pd[0][line]
    #     filename = file[:-8]
    #     #print(filename)
    #     for i in range(len(feature_set)):
    #         temp_file = feature_set[i][-3]
    #         #print(temp_file[:-8])
    #         if filename == temp_file[:-8]:
    #             feature_set_sort.append(feature_set[i])
    #             print(temp_file[:-8])
    #             cnt += 1
    # print(cnt)
    # """


    # df = pd.DataFrame(feature_set, columns=features_column)
    # if model_name[-7:-3] == '00.0':
    #     df.to_csv(save_path + '/img_features_SvsO-' + str(step_num) + '-' + model_name[-8:-3] + '.csv', index=False)
    # else:
    #     df.to_csv(save_path + '/img_features_SvsO-' + str(step_num) + '-' + model_name[-7:-3] + '.csv', index=False)
    
    
    # ✅ 이미지 로드 및 전처리
    img = image.load_img(data_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    print("입력 이미지 텐서의 shape:", img_tensor.shape)
    # print("모델이 기대하는 input shape:", test_model.input_shape)

    # ✅ 특징 추출
    # features = test_model.predict(img_tensor)[0]
    features = model.predict(img_tensor)[0]  # 모델을 직접 사용하여 특징 추출

    print("추출된 특징 벡터 크기:", features.shape)

    # ✅ CSV 저장
    feature_set = [features.tolist()]
    features_column = ['img_f' + str(i) + '_' + str(step_num) for i in range(len(features))]
    df = pd.DataFrame(feature_set, columns=features_column)
    df.to_csv(save_path + '/single_img_features.csv', index=False)

    print(f"✅ 특징 벡터 저장 완료: {save_path}/single_img_features.csv")

    return features

# for step_no in [ 1, 3, 4, 8]:

#     for k in ('train' , 'validation' , 'test'):
#         img_features_all = []
#         save_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-02-NIA-FeatSel-SvsO/01-FourTasks/' + str(step_no) + '/' + k
#         model_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-01-TOTAL-MelImg-SvsO-h5/' + str(step_no)
#         data_path = 'E:/01-DATASET/0-DATASET-IB_치매음성-TOTAL/01-EXP-20210215-01-TOTAL-MelImg-SvsO/' + str(step_no) + '/' + k

#         model_list = os.listdir(model_path)

#         for model_name in model_list:
#             if (model_name.find('h5') != -1):
#                 feature_extract(data_path, model_name, save_path, model_path, step_no)
#                 print(str(step_no) + ' , ' + k + ' is done')
#                 K.clear_session()

# # 3️⃣ SCI vs OTHERS 판별 (02Reust 모델을 활용해, 03-class_feature_extractor_SCIvsOTHERS.py 사용)
# mel_image_path = "temp_test.wav.jpg.jpg"
# model_name = "save_model_72.7.h5"  # 실제 모델 파일 이름
# model_path = "app/models/SCIvsOTHERS/1"  # 모델이 저장된 경로
# save_path = "app/feature_data/"  # 특징 저장 경로
# step_num = 1  # 학습 단계 (예: 1)

# sci_features = feature_extract_sci_vs_others(mel_image_path, model_name, save_path, model_path, step_num)    
