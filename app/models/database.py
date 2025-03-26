# DB 연결 설정

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 로컬 SQLite DB (나중에 RDS로 변경 가능)
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
# 예: RDS PostgreSQL: "postgresql://username:password@host:port/dbname"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
