# 🛡️ AI Community Nudge
**AI 기반 실시간 유해 콘텐츠 탐지 및 데이터 로깅 시스템**

---

### 🛠 Tech Stack
- **Framework**: `FastAPI`
- **Cloud**: `Azure SQL Database`, `Azure Blob Storage`
- **AI/ML**: `Scikit-learn (NLP)`, `Computer Vision API`
- **Database**: `SQLAlchemy` (NVARCHAR 다국어 처리)

---

### 📡 Data Pipeline
데이터 엔지니어링 관점의 **비정형 데이터 처리 및 로깅** 흐름입니다.

1. **Ingestion**: 사용자로부터 텍스트와 이미지 동시 수집
2. **Parallel Process**: `asyncio.gather`를 통한 AI 분석 및 스토리지 업로드 병렬화 (응답 속도 최적화)
3. **Storage Strategy**:
    - **Blob Storage**: 이미지 원본 저장 후 고유 **URL** 추출
    - **SQL DB**: 분석 스코어(Toxicity Score)와 이미지 URL을 매핑하여 적재
4. **MLOps Logging**: 모델 버전별 추론 시간(`inference_time`)을 개별 테이블에 기록하여 성능 모니터링 기반 마련



---

### 🗄️ Database Schema (ERD)
데이터 분석 효율성을 고려한 관계형 설계입니다.

- **Comments**: AI 분석 라벨 및 독성 스코어 실시간 저장
- **AdminLog**: 유해 콘텐츠 사후 관리를 위한 감사 로그
- **MLModel**: 모델 성능 지표 및 버전 관리 (Data Analysis용)



---

### 🌟 Core Value
- **Scalability**: 이미지 원본은 스토리지에, 주소는 DB에 저장하여 데이터 확장성 확보
- **Traceability**: 모든 AI 추론 이력을 로그화하여 모델 성능 분석 및 데이터 기반 의사결정 가능
- **User Experience**: '넛지(Nudge)' 로직을 통한 클린한 커뮤니티 환경 조성
