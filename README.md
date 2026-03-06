단순한 차단을 넘어, Text NLP 분석과 **Computer Vision(CV)**을 결합한 이중 검증 파이프라인을 구축했습니다. 데이터의 흐름과 모델 성능 로그를 체계적으로 적재하여 MLOps의 기초를 설계하는 데 집중했습니다.

🛠 Tech Stack
Backend: FastAPI (Python)

Database: Azure SQL Database (SQLAlchemy ORM)

Storage: Azure Blob Storage (Unstructured Data)

AI/ML: Scikit-learn (Text), Computer Vision API (Image)

DevOps: Azure Cloud Services

📊 Data Architecture & Flow
데이터 엔지니어링 관점에서 비정형 데이터(이미지)와 정형 데이터(분석 스코어)를 효율적으로 결합했습니다.

Ingestion: 사용자로부터 텍스트와 이미지 데이터를 수집 (FastAPI).

Parallel Processing: asyncio.gather를 통해 AI 분석과 Cloud 업로드를 병렬 처리하여 지연 시간 단축.

Storage:

Image: Azure Blob Storage에 저장 후 고유 URL 생성.

Metadata: 분석 스코어(Toxicity Score)와 URL을 DB에 매핑.

Monitoring: ml_model 테이블에 모델 버전별 추론 시간을 로깅하여 성능 모니터링 체계 구축.

🔑 Key Features
Multi-Modal Analysis: 텍스트(ComplementNB)와 이미지(CV) 분석 결과를 가중치 기반으로 합산하여 최종 위험도 산출.

Real-time Nudge: 유해 콘텐츠 발견 시 즉시 차단 및 관리자 대시보드 실시간 반영.

Admin Dashboard: 누적된 독성 스코어를 시각화하여 데이터 기반의 커뮤니티 관리 지원.

📂 Database Schema (DBML)
users: 사용자 권한 및 기본 정보.

posts / comments: 커뮤니티 본문 데이터.

admin_log: 필터링된 유해 콘텐츠 감사 로그.

ml_model: 모델별 추론 시간 및 성능 로그 (Data Analysis용).
