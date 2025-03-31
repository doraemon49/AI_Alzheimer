# DB 연결 설정

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 로컬 SQLite DB (나중에 RDS로 변경 가능)
# SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"

from dotenv import load_dotenv
import os
load_dotenv()  # .env 파일 읽기
DATABASE_URL = os.getenv("DATABASE_URL")

# sqlite 사용시
# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
