# 1. 베이스 이미지 설정 (Python + 필요한 패키지)
FROM python:3.10-slim

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 필요한 파일 복사
COPY . /app

# 4. 필요한 패키지 설치
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    soundfile \
    numpy \
    pandas \
    tensorflow==2.12 \
    keras==2.12 \
    python-multipart \
    librosa \
    matplotlib \
    pydantic \
    sqlalchemy \
    pymysql \
    python-dotenv


# 5. 기본 실행 명령 설정 (웹서버 실행)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# docker build & run # 터미널에 입력

# 1. Docker 이미지 만들기 (이미지 이름은 alz-back이라 하자)
# docker build -t alz-back .

# 1-1 Docker file만 수정했을 시, Docker 이미지 다시 생성
# 변경을 확실히 반영하고 싶기에 --no-cache 사용.
# docker build --no-cache -t alz-back .

# 2. (처음) 컨테이너 실행 (예시, 로컬 디렉토리와 연결해서 데이터 사용)
# docker run -v "${PWD}:/app" -p 8000:8000 alz-back
# docker run --name alz-container -v "${PWD}:/app" -p 8000:8000 alz-back    # 이름 명시

# docker run -v ${PWD}:/app alz-back    # 코드나 데이터셋, 결과를 내 PC 폴더와 공유
# docker run -p 8000:8000 alz-back      # 웹 서버 (FastAPI, Flask 등) 실행 → 외부 접속 가능

# 2-2. (기존) 컨테이너 다시 시작
# docker start elastic_dirac


