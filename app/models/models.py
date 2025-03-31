# DB 모델 정의 (SQLAlchemy)
from sqlalchemy import Column, Integer, String, Boolean, Date
from app.models.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    userInfoAgree = Column(Boolean, nullable=False)     # 개인정보 동의 (1: 동의, 0: 비동의)
    userName = Column(String(100), nullable=False)
    userDateOfBirth = Column(Date, nullable=False)
    userGender = Column(String(100), nullable=False)        # 성별  (true: 남성, false: 여성)
    userEdu = Column(String(100), nullable=False)            # 학력
    userPreResult = Column(String(100), nullable=False)      # 기존 인지 기능 점검 결과(”없음”, “정상”, “SCI” 등 - 미정)
