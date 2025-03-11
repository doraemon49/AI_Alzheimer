# app/main.py

from fastapi import FastAPI, File, UploadFile
import os
from tensorflow.keras.models import load_model
from app.utils.first_wav_to_mfcc import Mel_Spectrogram
from app.utils.third_class_feature_extractor_SCIvsOTHERS import feature_extract_sci_vs_others
from app.utils.third_class_feature_extractor_MCI_vs_AD import feature_extract_mci_vs_ad

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# 모델 로드
# SCI_MODEL_PATH = "we_dont_have.h5"
# MCI_AD_MODEL_PATH = "we_dont_have.h5"

# sci_model = load_model(SCI_MODEL_PATH)
# mci_ad_model = load_model(MCI_AD_MODEL_PATH)

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    # 1️⃣ 업로드된 음성 파일 저장
    temp_audio_path = f"temp_{file.filename}"
    with open(temp_audio_path, "wb") as buffer:
        buffer.write(file.file.read())

    # 2️⃣ 음성을 멜-스펙트로그램 이미지로 변환 (01-wav_to_mfcc.py 사용)
    mel_image_path = f"{temp_audio_path}.jpg"
    print("음성 to 이미지 변환", mel_image_path)
    Mel_Spectrogram(temp_audio_path, mel_image_path, sr=48000)  # 샘플링 레이트 지정

    # 3️⃣ SCI vs OTHERS 특징 추출 (02Reust 모델을 활용해, 03-class_feature_extractor_SCIvsOTHERS.py 사용)
    model_name = "save_model_72.7.h5"  # 실제 모델 파일 이름
    model_path = "app/models/SCIvsOTHERS/1"  # 모델이 저장된 경로
    save_path = "app/feature_data/"  # 특징 저장 경로
    step_num = 1  # 학습 단계 (예: 1)

    sci_features = feature_extract_sci_vs_others(mel_image_path, model_name, save_path, model_path, step_num)    
    
    # # 4️⃣ SCI vs OTHERS 판별
    # sci_prediction = sci_model.predict(np.array([sci_features]))[0][0]
    # if sci_prediction >= 0.5:
    #     os.remove(temp_audio_path)  # 임시 파일 삭제
    #     os.remove(mel_image_path)
    #     return {"status": "SCI", "message": "정상 상태입니다."}

    # # 5️⃣ MCI vs AD 특징 추출 (02Reust 모델을 활용해, 03-class_feature_extractor_MCI_vs_AD.py 사용)
    # model_name_mci = "save_model_64.2.h5"
    # model_path_mci = "app/models/MCIvsAD/6"
    # save_path_mci = "app/feature_data/"
    # step_num_mci = 1

    # mci_ad_features = feature_extract_mci_vs_ad(mel_image_path, model_name_mci, save_path_mci, model_path_mci, step_num_mci)
    # mci_ad_prediction = mci_ad_model.predict(np.array([mci_ad_features]))[0][0]

    # # 6️⃣ MCI vs AD 판별
    # diagnosis = "MCI" if mci_ad_prediction >= 0.5 else "AD"

    # 임시 파일 삭제
    os.remove(temp_audio_path)
    os.remove(mel_image_path)

    # return {"status": diagnosis, "message": f"{diagnosis} 상태로 판단됩니다."}
