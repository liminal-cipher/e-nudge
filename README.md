🛡️ AI Community Nudge
AI 기반 실시간 유해 콘텐츠 탐지 및 데이터 로깅 시스템

🛠 Tech Stack
Backend: FastAPI

Cloud: Azure (SQL Database, Blob Storage)

AI: Scikit-learn (NLP), Computer Vision API

ORM: SQLAlchemy (NVARCHAR 다국어 처리)

📡 Data Flow
사용자 입력부터 저장까지의 핵심 파이프라인입니다.

Input: 텍스트 + 이미지 수집

Process: asyncio 병렬 처리 (AI 분석 & 스토리지 업로드)

Storage:

Blob Storage: 이미지 원본 저장 → 고유 URL 생성

SQL DB: 분석 스코어(Toxicity Score) + 이미지 URL 매핑

Log: 모델별 추론 시간(inference_time) 개별 적재 (MLOps 기초)

🗄️ Database Schema
데이터 분석 및 관리를 위해 최적화된 구조입니다.

Comments: 유해성 스코어 및 이미지 URL 저장

AdminLog: 모니터링 대상 필터링 로그

MLModel: 모델 버전 관리 및 성능 지표(inference_time) 기록

🌟 Key Point
성능 최적화: 분석과 업로드의 병렬 처리를 통한 응답 속도 향상

데이터 기반: 모든 AI 추론 이력을 로그화하여 모델 성능 분석 가능 설계
