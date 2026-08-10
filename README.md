# E-Nudge

> 유해 댓글을 차단하는 대신 작성자의 성찰을 유도하는 넛지 기반 콘텐츠 모더레이션 서비스.
> A nudge-based moderation service that prompts authors to reflect, instead of just blocking toxic comments.

![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ComplementNB-F7931E?logo=scikitlearn&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-SQL%20%7C%20Blob%20%7C%20Custom%20Vision-0078D4)

Microsoft AI School 9기 1차 프로젝트 · 팀 고당스 (6인) · 2026.02.23 ~ 03.10 · **프로젝트 평가 1위**

[발표 자료](docs/presentation.pdf) · [데모 영상](docs/demo.mp4)

## Motivation

차단 중심 모더레이션은 작성자를 바꾸지 못한다. 삭제된 댓글은 우회 표현으로 돌아오고, 반발만 쌓인다. E-Nudge는 게시 전에 "이 댓글이 다른 사람에게 어떻게 읽힐지"를 되묻는 넛지를 끼워 넣어, 차단은 최후 단계로 미룬다.

기술 쪽 출발점은 성능 정체였다. 여러 모델과 파라미터를 튜닝해도 hate F1이 0.5 부근에서 멈췄고, 원인이 모델인지 데이터인지 알 수 없었다. 이 물음을 통제실험으로 규명한 것이 이 프로젝트의 핵심 기여다 ([Results](#results)).

## What It Does

1. **실시간 넛지 모더레이션**: 댓글 등록 시 텍스트(TF-IDF + ComplementNB)와 이미지(Azure Custom Vision)를 병렬 분석해 3단계로 분기한다. 정상 게시 / 넛지 경고(작성자에게 재고 요청) / 차단.
2. **Decision Engine**: `toxicity = hate×1.0 + offensive×0.5`로 결합하고 임계값(0.44 / 0.55)으로 분기한다. 판정은 결정적 Python 로직이며 LLM은 쓰지 않는다.
3. **관리자 대시보드**: 안전 / 주의 무시(경고를 무시하고 게시) / 위험 게시글을 모니터링하고 소프트 삭제한다. 삭제는 감사 로그(AdminLog)에 남는다.

데모 화면은 `templates/demo.html`(사용자)·`templates/admin.html`(관리자)이다.

## Architecture

```
댓글 등록
 ├─ 텍스트: 이모지·URL·특수문자·반복문자 제거 → Kiwi 형태소 분석(오타 교정)
 │          → 품사 필터(KEEP_TAGS) → TF-IDF → ComplementNB 확률
 ├─ 이미지: Blob 업로드 ∥ Custom Vision 분석 (asyncio.gather 병렬)
 └─ Decision Engine: 텍스트·이미지 점수 max 결합 → 3단계 분기
      ↓
 Azure SQL (Comments·AdminLog·MLModel) + Blob(이미지 원본, DB에는 URL만)
```

추론 시간은 모델 버전별로 MLModel 테이블에 기록해 성능 회귀를 추적할 수 있게 했다.

## Tech Decisions

| 영역 | 선택 | 이유 |
| --- | --- | --- |
| 분류 모델 | **ComplementNB + TF-IDF** (KcELECTRA F1 0.739 대신) | 배포 대상이 GPU 없는 Azure B2다. 성능이 더 높아도 서빙할 수 없는 모델은 서비스가 아니다. TF-IDF 가중치로 판정 근거를 설명할 수 있다는 부수 이점도 있다 |
| 판정 방식 | **LLM 미사용** | 댓글마다 호출하면 비용·지연이 쌓이고 판정 근거를 설명하기 어렵다. 넛지는 사용자를 설득해야 하는 기능이라 "왜"를 말할 수 있는 쪽을 택했다 |
| 형태소 분석 | **kiwipiepy** `typos="basic"` | 커뮤니티 댓글은 오타·변형이 많다. 형태소 분석 단계에서 흡수한다 |
| 응답 지연 | **`asyncio.gather` + `run_in_threadpool`** | 텍스트 추론·이미지 분석·Blob 업로드를 병렬화해 응답 지연을 줄인다 |
| 이미지 저장 | **원본은 Blob, DB에는 URL만** | DB 비대화를 막고 저장소와 메타데이터를 따로 확장할 수 있다 |

## Intended Use / Out of Scope

- **용도**: 커뮤니티 댓글의 게시 전 넛지와 사후 관리. 차단은 최후 단계다.
- **하지 않는 것**: 작성자 처벌이나 법적 판단을 하지 않는다. 자동 삭제로 끝내지 않고 소프트 삭제와 감사 로그를 남긴다. 판정 점수를 작성자 프로필에 누적하지 않는다.

## Data

| 항목 | 내용 |
|---|---|
| 텍스트(사람 라벨) | BEEP! 한국어 혐오표현 데이터셋 7,896건, 3클래스. 분포는 none 44.1% · offensive 31.6% · hate 24.2% |
| 텍스트(무라벨) | 커뮤니티 코퍼스 203만 건에 ComplementNB 시드 모델로 pseudo-label 부여 |
| 이미지 | 27,758장. 2단 구조로 Azure Custom Vision 학습 (Base는 NSFW 5클래스, K-Hate는 로고 64 + 밈 627) |
| 라벨 난도 | BEEP! 어노테이터 간 일치도(IAA) 0.496. 사람도 절반은 갈리는 문제다 |

BEEP!은 Moon et al., SocialNLP@ACL 2020이다. 학습 데이터는 repo에 포함하지 않는다. 포함된 것은 학습된 모델(`lr_model.pkl`, `tfidf_vectorizer.pkl`)뿐이다.

## Evaluation

- **지표**: hate F1. 소수 클래스가 관심 대상이라 accuracy는 쓰지 않는다.
- **모델이 아니라는 것부터 확인했다**: ML 알고리즘 9종(LR · Linear SVC · OvO · RF · GBC · MNB · CNB · MLP · Voting)이 전부 51~52%에서 수렴했고, GridSearchCV로 하이퍼파라미터 문제가 아님을 확인했다. 알고리즘을 바꿔도 천장이 같다는 것이 데이터를 의심한 근거다.
- **통제실험 설계**: 모델(KcELECTRA)과 평가셋을 고정하고 **학습 데이터만** 바꿨다. "표현 방식이 어려워서"와 "라벨이 나빠서"라는 두 가설을 한 실험으로 분리하기 위해서다.
- **한계**: 단일 실행이며 시드 반복·분산 측정은 하지 않았다. 통제실험 모델은 KcELECTRA 하나다.

## Results

| 구성 | hate F1 | |
|---|---|---|
| KcELECTRA, 사람 라벨 7,896건만 | **0.739** | 통제실험 A |
| KcELECTRA, + pseudo 6,316건 1:1 혼합 | 0.701 | 통제실험 B (−0.038) |
| ComplementNB (배포 모델) | 0.52 | CPU 서빙 가능. 튜닝 전 0.39 |

**pseudo를 6,316건만, 그것도 1:1로 섞었는데 전 클래스가 내려갔다.** 병목은 모델 용량이 아니라 pseudo 라벨의 품질이었다. ML 점수가 낮을 때는 pseudo가 효과 있는 것처럼 보였지만, 같은 데이터를 DL 기준으로 재니 노이즈였다.

203만 건을 만들어 두고도 소량 혼합에서 이미 하락이 나왔기 때문에 전량 투입은 하지 않았다. "데이터가 많을수록 좋다"가 라벨 노이즈 앞에서 뒤집히는 것을 확인한 지점이고, 사람이 라벨링하는 피드백 루프를 다음 단계로 잡은 근거다.

## Model & Inference

- 배포 모델은 **ComplementNB**다. 파일명 `lr_model.pkl`은 초기 로지스틱 회귀 시절의 잔재이며 내용물은 ComplementNB다.
- CPU 추론(Azure B2), 추론 시간은 요청마다 DB에 로깅.
- Azure ML Designer의 API 이슈로 서빙 경로를 바꿔, 학습된 `.pkl`을 FastAPI에 직접 탑재하는 방식으로 갔다.
- KcELECTRA(0.739)는 GPU 서빙 비용 때문에 배포하지 못했다. 대신 2단계로 나눴다. Phase 1은 ML로 빠르게 배포해 판정 근거를 설명할 수 있게 하고, Phase 2에서 피드백 루프로 사용자 라벨이 쌓이면 KcELECTRA fine-tuning으로 전환한다. 이 격차가 모델 경량화(양자화·distillation)를 공부하게 된 동기다.

## Getting Started

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Azure 리소스(SQL·Blob·Custom Vision) 자격 증명이 필요하다. 교육 과정 구독 종료로 현재는 로컬 텍스트 분석 경로만 재현 가능하다.

## Responsible AI

혐오표현 판정에는 AI가 끝까지 풀 수 없는 부분이 있다. 한국 커뮤니티의 밈과 짤은 같은 이미지라도 맥락에 따라 조롱이 되기도 하고 아니기도 해서, 공통 라벨을 학습시키는 것 자체가 성립하지 않는 경우가 있다. **맥락의 모호성을 없앨 수 있다고 보지 않고 인정한 뒤, 그 자리를 기능으로 메웠다.**

| 원칙 | 적용 |
| --- | --- |
| 투명성 | TF-IDF 계수로 판정 근거 추출 · 관리자에게 toxicity score 수치 공개 · Transparency Note 작성 |
| 책임성 | AI 판정과 사용자 행동 전부 기록 · 오탐 신고를 재학습 데이터로 환류 · 최종 삭제 권한은 관리자 · RAI Impact Assessment로 위험과 대응 문서화 |
| 공정성 | 사람이 라벨링한 BEEP! 데이터 사용 · 텍스트(NLP)와 이미지(CV) 결과 교차 검증 · 신고 데이터로 편향 수정 |
| 신뢰성 | none·offensive·hate 3단계 점진 개입 · "AI 판정 결과입니다" 한계 고지 · 텍스트·이미지 이중 판정 · 신고와 관리자 검토로 오탐·미탐 보완 |
| 개인정보·보안 | toxicity score 중심 저장, 원문 최소 보존 · 첫 접속 시 수집·이용 동의 · 이미지 원본은 Blob 보안 URL · 댓글 데이터와 관리 로그 분리 |
| 포용성 | 장애인 비하·성차별 등 사회적 약자 대상 혐오를 별도로 고려 · 차단이 아닌 넛지로 낙인과 역차별 우려를 완화 · 색상·아이콘·텍스트를 함께 써서 접근성 보완 |

## Team & Contributions

데이터 전처리 파이프라인, 발표 장표 구성 초안, 발표 대본 작성과 리허설은 6인 전원이 함께 했다. 아래는 그 외에 각자 맡은 일이다.

| 이름 | 담당 |
| --- | --- |
| **Youn Jae** (Team Lead · Dev Lead) | 프로젝트 아이템 원안 · 기술 계획서 · 넛지 팝업 UX 설계 · FastAPI 백엔드 · pseudo-labeling 파이프라인 · 오퍼레이션과 협업 조율 |
| Junsang | Custom Vision 데이터 수집·학습 · 비교 모델링 실험 · 프론트엔드 UI · Transparency Note와 RAI · 일정 관리 |
| Kenzie | 모델링(TF-IDF) · pseudo-labeling 파이프라인 · 비교 모델링 실험 · Transparency Note와 RAI · 발표 팩트체크·리서치·영상 · 일정 관리 |
| Yongju | FastAPI 백엔드 · DB 설계 · Azure SQL 연동과 배포(CI/CD) · Transparency Note와 RAI |
| Yuri | 프론트엔드 UI · 발표물 제작 총괄 · 일정 관리 |
| Juhee | 프로젝트 아이템 원안 · 발표 팩트체크·리서치·영상 · 오퍼레이션과 협업 조율 |

## My Role (조윤재)

| 담당 | 산출물 |
| --- | --- |
| 기획 | 프로젝트 아이템 원안 · 기술 계획서 |
| 서비스 설계 | 넛지 팝업 UX 설계 |
| 백엔드 | FastAPI 백엔드 `main.py`, Decision Engine 분기 로직 |
| 데이터 | pseudo-labeling 파이프라인 |
| 팀 운영 | 오퍼레이션과 협업 조율 |

> 역할 분담의 정본은 팀 전원이 합의한 기여도 문서이며, 위 표는 거기서 옮긴 것이다. 일부 항목은 팀원과 함께한 작업이다. 커밋 이력은 팀 계정으로 집중돼 있어 개인별 기여를 반영하지 않는다.

## Retrospective

- **다시 한다면 경량화를 먼저 검토한다.** KcELECTRA 0.739를 두고 0.52를 배포한 것은 인프라 제약 때문이었다. 양자화·distillation으로 그 격차를 좁히는 것이 다음 단계다.
- **pseudo-labeling 전에 라벨 감사부터.** 37만 건을 만들고 나서 품질 문제를 발견했다. 시드 모델의 신뢰도 분포를 먼저 검사했으면 실험 한 사이클을 줄일 수 있었다.

## Status

완료. Microsoft AI School 9기 1차 프로젝트로 2026.02.23 ~ 03.10 진행. Azure 배포는 종료됐고 코드·발표 자료·데모 영상만 남아 있다. 마지막 갱신 2026-08-11.
