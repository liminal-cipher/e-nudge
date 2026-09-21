# E-Nudge

> 유해 댓글을 차단하는 대신 작성자의 성찰을 유도하는 넛지 기반 콘텐츠 모더레이션 서비스.
> A nudge-based moderation service that prompts authors to reflect, instead of just blocking toxic comments.

![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ComplementNB-F7931E?logo=scikitlearn&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-SQL%20%7C%20Blob%20%7C%20Custom%20Vision-0078D4)

Microsoft AI School 9기 1차 프로젝트 · 팀 고당스 (6인) · 2026.02.23 ~ 2026.03.10 · **프로젝트 평가 1위**

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
| 분류 모델 | **ComplementNB + TF-IDF** (KcELECTRA F1 0.739 대신) | 당시 배포 환경은 GPU 없는 Azure B2였고 실시간 댓글 응답과 2.5주 일정까지 고려해야 했다. KcELECTRA가 실시간 서빙 불가능했던 것이 아니라 추가 최적화·운영 부담을 감안해 경량 모델을 택했다 |
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
| 텍스트(사람 라벨) | BEEP! 한국어 혐오표현 데이터셋 7,896건, 3클래스. KcELECTRA 실험에서는 Train 6,316 · Test 1,580으로 80:20 분리하고 Test를 고정했다. 분포는 none 44.1% · offensive 31.6% · hate 24.2% |
| 텍스트(무라벨) | 커뮤니티 코퍼스 약 203만 건을 후보 pool로 두고, 클래스별 ML 라벨러의 confidence·합의 규칙으로 필터링해 371,459건의 pseudo-labeled dataset을 만들었다 |
| 이미지 | 27,758장. 2단 구조로 Azure Custom Vision 학습 (Base는 NSFW 5클래스, K-Hate는 로고 64 + 밈 627) |
| 라벨 난도 | BEEP! 어노테이터 간 일치도(IAA) 0.496. 사람도 절반은 갈리는 문제다 |

BEEP!은 Moon et al., SocialNLP@ACL 2020이다. 학습 데이터는 repo에 포함하지 않는다. 포함된 것은 학습된 모델(`lr_model.pkl`, `tfidf_vectorizer.pkl`)뿐이다.

## Evaluation

- **주요 지표**: hate F1. 소수 클래스 탐지가 관심 대상이라 최종 비교는 hate F1을 중심으로 봤고, 고전 ML 탐색 단계에서는 accuracy도 함께 확인했다.
- **모델 선택·튜닝부터 확인했다**: TF-IDF 기반 ML 알고리즘 9종(LR · Linear SVC · OvO · RF · GBC · MNB · CNB · MLP · Voting)이 비슷한 성능대에 수렴했고, GridSearchCV에서도 큰 돌파가 없었다.
- **데이터 확장 결과가 모델마다 엇갈렸다**: pseudo 데이터를 추가해 ML 모델을 새로 학습했을 때 개선과 악화가 함께 나타나, pseudo-label 자체의 효과와 모델 특성을 분리하기 어려웠다.
- **KcELECTRA로 재진단했다**: 먼저 사람 라벨 Train 6,316건만으로 baseline을 만들고, 같은 KcELECTRA 구성과 고정 Test 1,580에서 pseudo 추가량만 바꿨다. 각 조건은 새 모델로 초기화했다.
- **한계**: 단일 seed 중심의 비교이며 반복 실행·분산 측정을 하지 않았다. 사람 라벨 자체의 양을 늘리는 실험도 하지 않았고, 1:1 pseudo subset의 class composition은 human train과 완전히 같지 않았을 수 있다.

## Results

| 구성 | 학습 데이터 | hate F1 |
|---|---|---:|
| KcELECTRA BEFORE | Human 6,316 | **0.739** |
| KcELECTRA AFTER-Full | Human 6,316 + Pseudo 371,459 | 0.589 |
| KcELECTRA AFTER 1:1 | Human 6,316 + Pseudo 6,316 | 0.701 |
| ComplementNB (배포 모델) | TF-IDF + 고전 ML | 0.52 |

KcELECTRA가 human-only에서 기존 TF-IDF 기반 고전 ML보다 크게 높은 성능을 보여 **기존 표현·모델링 접근의 한계**를 확인했다. 반대로 pseudo를 371,459건 전부 추가하면 0.589, 사람 train과 같은 6,316건만 추가해도 0.701로 human-only baseline보다 낮았다. 당시 만든 **pseudo-label의 품질도 충분하지 않았다**는 결론이다.

프로젝트가 직접 보여준 것은 `좋은 소량 human label > 당시의 대량 pseudo-label`이다. 더 많은 고품질 human label이 더 좋을지는 합리적인 기대지만, 이 프로젝트에서는 직접 실험하지 않았다.

## Model & Inference

- 배포 모델은 **ComplementNB**다. 파일명 `lr_model.pkl`은 초기 로지스틱 회귀 시절의 잔재이며 내용물은 ComplementNB다.
- CPU 추론(Azure B2), 추론 시간은 요청마다 DB에 로깅.
- Azure ML Designer의 API 이슈로 서빙 경로를 바꿔, 학습된 `.pkl`을 FastAPI에 직접 탑재하는 방식으로 갔다.
- KcELECTRA(0.739)는 당시 CPU B2 · 실시간 댓글 응답 · 2.5주 일정에서 추론과 운영 부담이 더 컸다. 실시간 서빙이 불가능해서가 아니라 정확도·지연시간·인프라·일정의 trade-off로 ComplementNB를 배포했다. 향후에는 넛지 후 수정·철회·무시 같은 행동 신호를 검증·정제해 학습 데이터 후보로 쓰는 방향을 제안했지만, 이 피드백 루프는 구현하지 않았다.

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
| 투명성 | 관리자에게 toxicity score 수치 공개 · Transparency Note 작성. 사용자-facing 특징 기여 설명 UI/API는 구현하지 않았다 |
| 책임성 | AI 판정과 사용자 행동 기록 · 최종 삭제 권한은 관리자 · RAI Impact Assessment로 위험과 대응 문서화 · 신고와 행동 신호의 재학습 활용은 향후 검증·정제 과제로 제안 |
| 공정성 | 사람이 라벨링한 BEEP! 데이터 사용 · 텍스트(NLP)와 이미지(CV) 결과 교차 검증 · 신고 데이터는 향후 편향 점검 신호로 활용하는 방향을 제안 |
| 신뢰성 | none·offensive·hate 3단계 점진 개입 · "AI 판정 결과입니다" 한계 고지 · 텍스트·이미지 이중 판정 · 신고와 관리자 검토로 오탐·미탐 보완 |
| 개인정보·보안 | toxicity score 중심 저장, 원문 최소 보존 · 첫 접속 시 수집·이용 동의 · 이미지 원본은 Blob 보안 URL · 댓글 데이터와 관리 로그 분리 |
| 포용성 | 장애인 비하·성차별 등 사회적 약자 대상 혐오를 별도로 고려 · 차단이 아닌 넛지로 낙인과 역차별 우려를 완화 · 색상·아이콘·텍스트를 함께 써서 접근성 보완 |

## Team & Contributions

데이터 전처리 파이프라인, 발표 장표 구성 초안, 발표 대본 작성과 리허설은 6인 전원이 함께 했다. 아래는 그 외에 각자 맡은 일이다.

| 이름 | 역할 | 담당 |
| --- | --- | --- |
| **Youn Jae** | Team Lead / Dev Lead | 프로젝트 아이템 원안 · 기술 계획서 · 시스템 아키텍처 공동 설계 · 넛지 팝업 및 대시보드 UX 공동 설계 · FastAPI 백엔드 · pseudo-labeling 파이프라인 공동 구축 · 데이터 품질 가설 공동 검토와 KcELECTRA 결과 해석 · 최종 발표 모델링 파트 전체 · 오퍼레이션과 협업 조율 |
| Junsang | Custom Vision | Custom Vision 데이터 수집·학습 · 비교 모델링 실험 · 프론트엔드 UI · Transparency Note와 RAI · 일정 관리 |
| Kenzie | Modeling | 모델링(TF-IDF) · pseudo-labeling 파이프라인 · 비교 모델링 실험 · Transparency Note와 RAI · 발표 팩트체크·리서치·영상 · 일정 관리 |
| Yongju | Backend / Database | 시스템 아키텍처 공동 설계 · FastAPI 백엔드 · DB 설계 · Azure SQL 연동과 배포(CI/CD) · Transparency Note와 RAI |
| Yuri | Frontend | 프론트엔드 UI/UX · 발표물 제작 총괄 · PM 및 일정 관리 |
| Juhee | Operations | 프로젝트 아이템 원안 · 발표 팩트체크·리서치·영상 · 오퍼레이션과 협업 조율 |

## My Role (조윤재)

| 담당 | 산출물 |
| --- | --- |
| 기획 | 프로젝트 아이템 원안 · 기술 계획서 |
| 서비스 설계 | 시스템 아키텍처(공동) · 넛지 팝업 및 대시보드 UX(공동) |
| 백엔드 (Dev Lead) | FastAPI 기반 ML 추론 서버, Decision Engine 분기 로직 |
| 데이터·분석 | pseudo-labeling 파이프라인 공동 구축 · 데이터 품질 가설 공동 검토 · KcELECTRA 통제실험 결과 해석 |
| 발표 | 최종 발표에서 모델링 파트 전체 담당 |
| 팀 운영 | 팀 협업 워크플로우 및 오퍼레이션 |

> 역할 분담의 정본은 팀 전원이 합의한 기여도 문서이며, 위 표는 거기서 옮긴 것이다. 일부 항목은 팀원과 함께한 작업이다. 커밋 이력은 팀 계정으로 집중돼 있어 개인별 기여를 반영하지 않는다.

## Retrospective

- **다시 한다면 경량화를 먼저 검토한다.** KcELECTRA 0.739를 두고 0.52를 배포한 것은 인프라 제약 때문이었다. 양자화·distillation으로 그 격차를 좁히는 것이 다음 단계다.
- **pseudo-labeling 전에 라벨 감사를 먼저 했어야 한다.** 약 203만 후보에서 371,459건을 채택한 뒤 품질 문제를 확인했다. 라벨러별 신뢰도와 합의 규칙을 먼저 감사했으면 실험 한 사이클을 줄일 수 있었다.

## Status

완료. Microsoft AI School 9기 1차 프로젝트로 2026.02.23 ~ 2026.03.10 진행. Azure 배포는 종료됐고 코드·발표 자료·데모 영상만 남아 있다. 마지막 갱신 2026-09-21.
