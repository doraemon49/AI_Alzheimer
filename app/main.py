# app/main.py

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
import os, shutil
from tensorflow.keras.models import load_model
from app.utils.first_wav_to_mfcc import Mel_Spectrogram
from app.utils.third_class_feature_extractor_SCIvsOTHERS import feature_extract_sci_vs_others
from app.utils.third_class_feature_extractor_MCI_vs_AD import feature_extract_mci_vs_ad
from typing import List
from pydantic import BaseModel

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app.models.database import SessionLocal, engine
from app.models import models, schemas
from datetime import datetime
# 모든 테이블을 삭제 후 재생성 (데이터는 모두 삭제됩니다!)
# models.Base.metadata.drop_all(bind=engine)
models.Base.metadata.create_all(bind=engine)

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# 허용할 origin 목록
origins = [
    "http://localhost:3000",           # 개발 중 프론트
    "http://15.165.205.236:3000"      # 혹시 EC2 IP에서 프론트도 띄운다면
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # CORS 허용할 Origin
    allow_credentials=True,
    allow_methods=["*"],              # 모든 HTTP 메소드 허용
    allow_headers=["*"],              # 모든 헤더 허용
)

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/signup", response_model=schemas.SignupResponse)
def signup(request: schemas.SignupRequest, db: Session = Depends(get_db)):
    if not request.userInfoAgree:
        return {"status": "fail", "message": "개인정보 수집에 동의하지 않았습니다.", "data": None}

    user = models.User(**request.dict())
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "status": "success",
        "message": "개인정보 수집에 동의하였습니다.",
        "data": {"userId": user.id}
    }

# 모델 로드
# SCI_MODEL_PATH = "we_dont_have.h5"
# MCI_AD_MODEL_PATH = "we_dont_have.h5"

# sci_model = load_model(SCI_MODEL_PATH)
# mci_ad_model = load_model(MCI_AD_MODEL_PATH)

@app.post("/upload", response_model=schemas.DiagnosisResponse)
async def  diagnose(
    userId: int = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    # 사용자 존재 확인
    user = db.query(models.User).filter(models.User.id == userId).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자 정보를 찾을 수 없습니다.")
    
    # 파일 개수 체크
    if len(files) != 11:
        raise HTTPException(status_code=400, detail="11개의 음성 파일이 필요합니다.")
    
    

    # 1️⃣ 업로드된 음성 파일 저장
    voices = []
    file_paths = []

    upload_dir = f"uploads/user_{userId}"
    os.makedirs(upload_dir, exist_ok=True)

    for file in files:
        # 파일 순서대로 바이너리로 변환 (AWS RDS 저장용)
        content = await file.read()
        voices.append(content)

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)  # 로컬에 음성 파일 저장
        file_paths.append(file_path)


    # 2️⃣ 음성을 멜-스펙트로그램 이미지로 변환 (01-wav_to_mfcc.py 사용)
    mel_image_paths = []

    for file_path in file_paths:
        # 이미지 저장 경로: uploads/user_{userId}/xxx.jpg
        file_stem = os.path.splitext(os.path.basename(file_path))[0]
        mel_image_path = os.path.join(upload_dir, f"{file_stem}.jpg")

        Mel_Spectrogram(file_path, mel_image_path, sr=48000)
        mel_image_paths.append(mel_image_path)

    # 3️⃣ SCI vs OTHERS 특징 추출 (02Reust 모델을 활용해, 03-class_feature_extractor_SCIvsOTHERS.py 사용)
    model_name = "save_model_72.7.h5"  # 실제 모델 파일 이름
    model_path = "app/models/SCIvsOTHERS/1"  # 모델이 저장된 경로
    save_path = f"uploads/user_{userId}"  # 특징 저장 경로
    step_num = 1  # 학습 단계 (예: 1)

    sci_features = feature_extract_sci_vs_others(mel_image_paths, model_name, save_path, model_path, step_num)    
    
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
    diagnosis = "Nomal"  # 예시
    confidence = 0.94  # 예시


    # 결과 DB 저장
    result = models.DiagnosisResult(
        user_id=userId,
        voice1=voices[0], voice2=voices[1], voice3=voices[2],
        voice4=voices[3], voice5=voices[4], voice6=voices[5],
        voice7=voices[6], voice8=voices[7], voice9=voices[8],
        voice10=voices[9], voice11=voices[10],        
        diagnosis=diagnosis,
        confidence=confidence,
        created_at=datetime.utcnow()
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    # 진단 끝난 후, 임시 저장용 로컬 파일 및 폴더 정리 (음성 파일과 이미지 파일)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)

    return {
        "status": "success", 
        "data" : {
            "userId": user.id,
            "diagnosis": diagnosis,
            "confidence": confidence        
        },
        "message": "치매 진단 완료하였습니다."
    }
