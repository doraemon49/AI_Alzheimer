import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def Mel_Spectrogram(wav_path, save_file, sr):
    y, sr = librosa.load(wav_path, sr=sr)

    sec = 30

    index = sr * sec

    if len(y) < index:
        y_segment = y
    else:
        y_segment = y[0:index]

    # Mel-Spectrogram

    S = librosa.feature.melspectrogram(y=y_segment, sr=sr)
    print("Wav length: {}, Mel_S shape:{}".format(len(y_segment) / sr, np.shape(S)))
    plt.figure(figsize=(1, 1))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig(save_file + '.jpg')
    # plt.show()
    plt.close()



"""base_path = 'C:/Users/minwo/바탕 화면/original/'
file_path = base_path + 'sound_data/'
save_path = base_path + 'img_data/mel_spectrogram.ver2/'"""

for num in range(0,11):
#for num in ['CERAD-K']:
    #for k in ('train', 'validation','test'):
        for disease in ( 'SCI', 'MCI', 'AD'):
            base_path = 'D:/01-20200907-DATASET/'
            file_path = base_path + '0-DATASET-아이디어빈스_치매음성-NIA/20210215-NIA-audio-Excluding20201222Data/IB-APPS/' + str(num) + '/' + disease + '/'
            save_path = base_path + '0-DATASET-아이디어빈스_치매음성-TOTAL/01-MELS-IMG-F-NIA20210215-excluding20201222/' + str(num+1)+ '/' + disease + '/'
            file_list = os.listdir(file_path)

            #print(file_path + "  START")
            #print(file_path + "  save_path")

            sr = 48000

            count = 1
            for file_name in file_list:
                if file_name.find('flac') != -1:
                    wav_file = file_path + '/' + file_name
                    save_file = save_path + '/' + file_name[:-5]
                else:
                    wav_file = file_path + '/' + file_name
                    save_file = save_path + '/' + file_name[:-4]
                Mel_Spectrogram(wav_file, save_file, sr)
                # Spectrogram(wav_file, save_file, sr)
                # Spectrum(wav_file, save_file, sr)
                print(file_name + ' has finished ' + str(count))
                count += 1
