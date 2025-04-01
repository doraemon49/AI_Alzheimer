# DB 모델 정의 (SQLAlchemy)
from sqlalchemy import Column, Integer, String, Boolean, Date, DateTime, Float, ForeignKey, LargeBinary
from datetime import datetime

from app.models.database import Base
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    userInfoAgree = Column(Boolean, nullable=False)     # 개인정보 동의 (1: 동의, 0: 비동의)
    userName = Column(String(100), nullable=False)
    userDateOfBirth = Column(Date, nullable=False)
    userGender = Column(String(100), nullable=False)        # 성별 ("MALE","FEMALE")
    userEdu = Column(String(100), nullable=False)            # 학력
    userPreResult = Column(String(100), nullable=True)      # 기존 인지 기능 점검 결과(있을 경우 : 사용자가 입력, 없을 경우:null)

    results = relationship("DiagnosisResult", back_populates="user")

class DiagnosisResult(Base):
    __tablename__ = "diagnosis_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    # 11개의 음성 파일 컬럼
    voice1 = Column(LargeBinary, nullable=True)
    voice2 = Column(LargeBinary, nullable=True)
    voice3 = Column(LargeBinary, nullable=True)
    voice4 = Column(LargeBinary, nullable=True)
    voice5 = Column(LargeBinary, nullable=True)
    voice6 = Column(LargeBinary, nullable=True)
    voice7 = Column(LargeBinary, nullable=True)
    voice8 = Column(LargeBinary, nullable=True)
    voice9 = Column(LargeBinary, nullable=True)
    voice10 = Column(LargeBinary, nullable=True)
    voice11 = Column(LargeBinary, nullable=True)    
    
    diagnosis = Column(String(50), nullable=False)                  # 진단 (Normal, MCI, AD)
    confidence = Column(Float, nullable=False)                      # 진단 확률
    created_at = Column(DateTime, default=datetime.utcnow)
    created_at_kst = Column(String(10), default="+09:00")

    user = relationship("User", back_populates="results")