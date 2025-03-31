# AI_Alzheimer

음성 파일을 활용하여 Alzheimer 진단하는 AI

<img src="https://github.com/user-attachments/assets/0c20dfeb-042c-4521-9449-f79b5265e0cd" alt="Image" width="300">
<br/>
<br/>

[서비스 링크(배포 예정)](http://localhost:8000/)

[Git Hub 링크](https://github.com/doraemon49/AI_Alzheimer.git)  
<br/>
<br/>

# 0\. Getting Started (시작하기)

## 초기 세팅

```bash
git clone https://github.com/doraemon49/AI_Alzheimer.git

python -m venv venv

.\venv\Scripts\Activate

pip install fastapi

pip install uvicorn

pip install soundfile numpy pandas tensorflow python-multipart

pip install librosa matplotlib tensorflow keras

pip install pydantic sqlalchemy pymysql
pip install python-dotenv

```

## 실행

```bash

.\venv\Scripts\Activate

uvicorn app.main:app --reload
```

## 로컬 웹페이지 접속

http://localhost:8000/

<br/>
<br/>

# 1\. Project Overview (프로젝트 개요)

- 프로젝트 이름: AI 치매 지킴이
- 프로젝트 소개: 음성 파일을 활용하여 Alzheimer 진단하는 AI

- 프로젝트 목적 :
- 프로젝트 설명 : 음성 파일을 이미지로 변환
  <img src="https://github.com/user-attachments/assets/f1ad7969-6552-49c5-810e-c01b762d5b99" alt="Image" width="100">
- 프로젝트 결과 :

<br/>
<br/>

# 2\. Team Members (팀원 및 팀 소개)

|                             김민                              |                             정현석                              |                             이름                              |
| :-----------------------------------------------------------: | :-------------------------------------------------------------: | :-----------------------------------------------------------: |
| ![김민](https://avatars.githubusercontent.com/u/59240554?v=4) | ![정현석](https://avatars.githubusercontent.com/u/87671916?v=4) | ![이름](https://avatars.githubusercontent.com/u/59240554?v=4) |
|                              BE                               |                               BE                                |                              FE                               |
|            [GitHub](https://github.com/doraemon49)            |            [GitHub](https://github.com/whilethis00)             |            [GitHub](https://github.com/doraemon49)            |

<br/>
<br/>

# 3\. Key Features (주요 기능)

- 음성을 통한 치매 진단 AI
- 마이페이지
- 회원가입
  <br/>
  <br/>

# 4\. Tasks & Responsibilities (작업 및 역할 분담)

|        |                                                                                           |                                                             |
| ------ | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 김민   | <img src="https://avatars.githubusercontent.com/u/59240554?v=4" alt="김민" width="100">   | <ul><li>백엔드 개발</li></ul>                               |
| 정현석 | <img src="https://avatars.githubusercontent.com/u/87671916?v=4" alt="정현석" width="100"> | <ul><li>프로젝트 계획 및 관리</li><li>백엔드 개발</li></ul> |
| 이름   | <img src="https://avatars.githubusercontent.com/u/59240554?v=4" alt="이름" width="100">   | <ul><li>프론트엔드 개발</li></ul>                           |

<br/>
<br/>

# 5\. Technology Stack (기술 스택)

## 5.1 Frotend

[![My Skills](https://skillicons.dev/icons?i=figma,flutter&theme=light)](https://skillicons.dev)

## 5.2 Backend

[![My Skills](https://skillicons.dev/icons?i=python,fastapi&theme=light)](https://skillicons.dev)

## 5.3 Cooperation

[![My Skills](https://skillicons.dev/icons?i=git,notion&theme=light)](https://skillicons.dev)

<br/>
<br/>

# 6. Project Structure (프로젝트 구조)

```
<< Back-End >>

📦
├─ .gitignore
├─ README.md
└─ app
   ├─ main.py
   └─ utils
      ├─ 01-wav_to_mfcc.py
      ├─ 03-class_feature_extractor_SCIvsOTHERS.py
      └─ 03_class_feature_extractor_MCI_vs_AD.py


```

```
<< Front-End >>
📦

```

# 7\. Development Workflow (개발 워크플로우)

## 브랜치 전략 (Branch Strategy)

우리의 브랜치 전략은 Git Flow를 기반으로 하며, 다음과 같은 브랜치를 사용합니다.

<< Back-End >>

- Main Branch
  - 배포 가능한 상태의 코드를 유지합니다.
  - 모든 배포는 이 브랜치에서 이루어집니다.
  - 권한이 있는 사용자만 Merge할 수 있습니다. (Branches Rules)-
- dev Branch
  - Main Branch에 업로드되기 전, 개발 단계에서 모든 브랜치의 코드를 통합하는 브랜치입니다.
- feat/AI
  - AI 치매 진단 브랜치입니다.

<< Front-End >>

- Main Branch
  - 배포 가능한 상태의 코드를 유지합니다.
  - 모든 배포는 이 브랜치에서 이루어집니다.
  - 권한이 있는 사용자만 Merge할 수 있습니다. (Branches Rules)
- dev Branch
  - Main Branch에 업로드되기 전, 개발 단계에서 모든 브랜치의 코드를 통합하는 브랜치입니다.
