# 요청/응답 스키마
# Pydantic 스키마 정의

from pydantic import BaseModel
from datetime import date

class SignupRequest(BaseModel):
    userInfoAgree: bool
    userName: str
    userDateOfBirth: date
    userGender: bool
    userEdu: str
    userPreResult: str

class SignupResponseData(BaseModel):
    userId: int

class SignupResponse(BaseModel):
    status: str
    message: str
    data: SignupResponseData | None

