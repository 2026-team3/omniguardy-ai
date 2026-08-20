# Entrance Behavior Vision Pipeline

현관 카메라 영상에서 정상 활동과 이상행동을 블록 단위로 분류하는 Vision 파이프라인입니다.

## 분류 라벨

| 라벨 | 의미 |
|---|---|
| N1 | 정상 현관 활동: 거주자 출입, 일반 배달, 대기, 이탈 |
| A17 | 초인종 반복 누름 |
| A18 | 문 열기 반복 시도 |
| A19 | 문 반복 발차기 |
| A20 | 문 안쪽을 들여다보기 시도 |
| A21 | 장시간 또는 반복 문 두드림 |

## 데이터 분할

| 구분 | 사용 데이터 |
|---|---|
| Train | AIHub TL/TS, mydata, train 원본에서 파생된 augmentation |
| Validation | AIHub VL/VS 원본 영상만 사용 |
| Test | 별도 전시 카메라 또는 보류 현장 영상으로 추후 구성 |

mydata의 `normal`, `delivery`는 N1으로 사용하고, `lookingInside`는 A20으로 사용합니다.

## 처리 흐름

```text
영상
→ YOLO 사람 탐지 + BoT-SORT tracking
→ MediaPipe Pose 추출
→ annotation 블록별 행동·pose 특징 집계
→ XGBoost 분류
→ N1 / A17~A21 예측 결과
```

- tracking과 pose는 영상당 한 번만 실행합니다.
- 특징과 학습은 annotation의 `start_frame`, `end_frame` 블록 범위를 사용합니다.
- train의 augmentation은 원본 블록 경계를 그대로 사용합니다.

## 실행 순서

가상환경을 활성화한 뒤 프로젝트 루트에서 실행합니다.

```powershell
python .\utils\load_annotation.py
python .\utils\generate_annotations.py
python .\tracking\tracker.py
python .\tracking\extract_behavior_features.py
python .\tracking\pose_extractor.py
python .\tracking\merge_features.py
python .\models\train_XGBoost.py
```

`tracker.py`와 `pose_extractor.py`는 영상별 CSV가 이미 있으면 건너뛰므로, 중단 후 재실행할 수 있습니다.

## 생성 결과

| 경로 | 내용 |
|---|---|
| `splits/annotations/train_annotations.csv` | TL/TS 기반 AIHub train 블록 manifest |
| `splits/annotations/valid_annotations.csv` | VL/VS 기반 validation 블록 manifest |
| `splits/annotations/train_annotations_all.csv` | AIHub + mydata + augmentation train manifest |
| `results/tracking/` | 영상별 사람 tracking 결과 |
| `results/pose/` | 영상별 pose 프레임 특징 |
| `results/features/` | 블록별 behavior·pose 병합 특징 |
| `results/models/xgboost_validation/` | validation report, confusion matrix, 모델 파일 |

## 현재 baseline 결과

AIHub VL/VS hold-out validation 기준:

- Accuracy: **86.36%**
- Macro F1-score: **68.82%**
- Weighted F1-score: **86.20%**

N1 비중이 높으므로 Accuracy만이 아니라 Macro F1-score를 함께 평가합니다. 이 결과는 validation 성능이며, 최종 일반화 성능은 전시 카메라 또는 별도 현장 test 영상으로 추가 검증해야 합니다.

## Git 관리

원본·증강 영상, 실행 결과, 자동 생성 manifest, 모델 가중치는 `.gitignore`로 제외합니다. 코드와 재현에 필요한 설정 파일만 Git으로 관리합니다.
