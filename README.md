# AI Community Nudge (E-Nudge)

유해 댓글을 차단하는 대신 작성자의 성찰을 유도하는 넛지 기반 콘텐츠 모더레이션 서비스.
A nudge-based moderation service that prompts authors to reflect, instead of just blocking toxic comments.

![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ComplementNB-F7931E?logo=scikitlearn&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-SQL%20%7C%20Blob%20%7C%20Custom%20Vision-0078D4)

Microsoft AI School 9기 1차 프로젝트 · 팀 고당스(6인) · **1차 프로젝트 평가 1위** · 2026.02.23 ~ 03.10

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

| 선택 | 이유 |
|---|---|
| ComplementNB + TF-IDF (KcELECTRA 0.739 대신) | 배포 대상이 GPU 없는 Azure B2. 성능이 더 높아도 서빙할 수 없는 모델은 서비스가 아니다. TF-IDF 가중치로 판정 근거를 설명할 수 있다는 부수 이점 |
| LLM 미사용 | 댓글마다 호출하면 비용·지연이 쌓이고, 판정 근거를 설명하기 어렵다. 넛지는 사용자를 설득해야 하는 기능이라 "왜"를 말할 수 있는 쪽을 택했다 |
| kiwipiepy `typos="basic"` | 커뮤니티 댓글은 오타·변형이 많다. 형태소 분석 단계에서 흡수한다 |
| `asyncio.gather` + `run_in_threadpool` | 텍스트 추론·이미지 분석·Blob 업로드를 병렬화해 응답 지연을 줄인다 |
| 이미지 원본은 Blob, DB에는 URL만 | DB 비대화 방지, 저장소와 메타데이터의 확장 분리 |

## Intended Use / Out of Scope

- **용도**: 커뮤니티 댓글의 게시 전 넛지와 사후 관리. 차단은 최후 단계다.
- **하지 않는 것**: 작성자 처벌이나 법적 판단을 하지 않는다. 자동 삭제로 끝내지 않고 소프트 삭제와 감사 로그를 남긴다. 판정 점수를 작성자 프로필에 누적하지 않는다.

## Data

| 항목 | 내용 |
|---|---|
| 텍스트(사람 라벨) | BEEP! 한국어 혐오표현 데이터셋 7,896건, 3클래스(hate / offensive / none) |
| 텍스트(무라벨) | 커뮤니티 코퍼스 약 37만 건, ComplementNB 시드 모델로 pseudo-labeling |
| 이미지 | 27,758장으로 Azure Custom Vision 학습 |
| 라벨 난도 | BEEP! 어노테이터 간 일치도(IAA) 0.496. 사람도 절반은 갈리는 문제다 |

학습 데이터는 repo에 포함하지 않는다. 포함된 것은 학습된 모델(`lr_model.pkl`, `tfidf_vectorizer.pkl`)뿐이다.

## Evaluation

- **지표**: hate F1. 소수 클래스가 관심 대상이라 accuracy는 쓰지 않는다.
- **통제실험 설계**: 모델(KcELECTRA)과 평가셋을 고정하고 **학습 데이터만** 바꿨다. "표현 방식이 어려워서"와 "라벨이 나빠서"라는 두 가설을 한 실험으로 분리하기 위해서다.
- **외부 베이스라인**: BEEP! 원저자의 KoBERT hate F1 0.525.
- **한계**: 단일 실행이며 시드 반복·분산 측정은 하지 않았다. 통제실험 모델은 KcELECTRA 하나다.

## Results

| 구성 | hate F1 | |
|---|---|---|
| KcELECTRA, 사람 라벨 7,896건만 | **0.739** | 통제실험 A |
| KcELECTRA, + pseudo 37만 건 혼합 | 0.589 | 통제실험 B (−0.150) |
| ComplementNB (배포 모델) | 0.52 | CPU 서빙 가능 |
| KoBERT (원저자 벤치마크) | 0.525 | 참고 |

**데이터를 37만 건 더했더니 성능이 내려갔다.** 병목은 모델 용량이 아니라 pseudo 라벨의 품질이었다. "데이터가 많을수록 좋다"가 라벨 노이즈 앞에서 뒤집히는 것을 정량으로 확인했다.

이미지 분류(Custom Vision)는 Precision 92.7% / Recall 92.4%.

## Model & Inference

- 배포 모델은 **ComplementNB**다. 파일명 `lr_model.pkl`은 초기 로지스틱 회귀 시절의 잔재이며 내용물은 ComplementNB다.
- CPU 추론(Azure B2), 추론 시간은 요청마다 DB에 로깅.
- KcELECTRA(0.739)는 GPU 서빙 비용 때문에 배포하지 못했다. 이 격차가 모델 경량화(양자화·distillation)를 공부하게 된 동기다.

## Getting Started

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Azure 리소스(SQL·Blob·Custom Vision) 자격 증명이 필요하다. 교육 과정 구독 종료로 현재는 로컬 텍스트 분석 경로만 재현 가능하다.

## Team & My Role

| Member | Role |
| :--- | :--- |
| **Youn Jae** | **Dev Lead / Project Management** |
| Kenzie | ML Modeling / Pseudo-labeling |
| Yongju | Backend API / DB Design |
| Yuri | Frontend Development |
| Junsang | Image Data Collection & Preprocessing |
| Juhee | Text Data Preprocessing |

**조윤재(Dev Lead)의 몫**: 프로젝트 아이템 원안(주희와 공동), 기술 계획서, Decision Engine 분기 로직 설계·구현, 학습된 모델의 FastAPI 서빙 로딩(용주와 백엔드 공동), pseudo-labeling 파이프라인 공동 구축(경아와), 통제실험의 "라벨 품질이 병목" 가설 제기와 결과 해석·발표.

역할 분담의 정본은 팀 전원이 합의한 기여도 문서다. 커밋 이력은 팀 계정으로 집중돼 있어 개인별 기여를 반영하지 않는다.

## Retrospective

- **다시 한다면 경량화를 먼저 검토한다.** KcELECTRA 0.739를 두고 0.52를 배포한 것은 인프라 제약 때문이었다. 양자화·distillation으로 그 격차를 좁히는 것이 다음 단계다.
- **pseudo-labeling 전에 라벨 감사부터.** 37만 건을 만들고 나서 품질 문제를 발견했다. 시드 모델의 신뢰도 분포를 먼저 검사했으면 실험 한 사이클을 줄일 수 있었다.

## Status

완료 (2026-03-11 동결). Microsoft AI School 1차 프로젝트 출품작.
