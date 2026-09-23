# V4 기반 현장 위험음 Fine-tuning

`audio_model_v5_general_*.keras`는 새로운 MIL 파이프라인이 아닙니다. 기존 v4의 3초
슬라이딩 윈도우와 CNN 구조를 유지한 fine-tuned 모델 파일 버전입니다.

## 데이터 분리

현장 데이터는 다음처럼 둡니다.

```text
data/dataset/
├── abnormal/
└── field_splits.csv
```

`event_type`은 선택 항목이라 기존 3열 CSV도 읽을 수 있지만, 유형별 평가와
초인종 제외를 위해 다음 4열 형식을 사용합니다.

```csv
filename,split,group_id,event_type
abnormal/knock_001.wav,train,door_a_day_01,knock
abnormal/handle_pull_001.wav,validation,door_b_night_01,door_handle
abnormal/impact_001.wav,test,door_c_day_01,impact
abnormal/doorbell_001.wav,ignored,doorbell_a_day_01,doorbell
```

field 데이터는 직접 수집한 abnormal WAV만 사용하며 normal WAV는 요구하지
않습니다. 같은 `group_id`는 하나의 split에만 존재해야 합니다. train,
validation, test에는 각각 abnormal 파일이 최소 하나 이상 필요합니다.
`doorbell`은 반드시 `ignored`로 지정하며 학습과 평가에서 제외됩니다.
augmentation은 train split에만 적용되며 validation/test 원본 WAV에는 적용되지
않습니다. 지원 유형은 `knock`, `door_handle`, `forced_door`, `impact`,
`glass_breaking`, `siren`, `footsteps`, `other`, `doorbell`입니다.
같은 event type에 독립된 `group_id`가 3개 이상이면 train/validation/test에 모두
분산해야 하며, 2개이면 서로 다른 두 split에 배치해야 합니다.

## Fine-tuning

```bash
python -m audio_guard.interfaces.cli.finetune_cli \
  --esc50-dir data/ESC-50 \
  --field-data-dir data/dataset \
  --field-splits data/dataset/field_splits.csv \
  --base-model models/audio_model_v4.keras \
  --model-out models/audio_model_v5_general_w1_0.keras \
  --epochs 5 \
  --learning-rate 5e-6 \
  --abnormal-weight-multiplier 1.0
```

fine-tuning은 ESC-50 fold 1~3의 normal/abnormal 데이터와 field abnormal train을
함께 사용합니다. field validation/test/ignored는 학습에 포함하지 않습니다. field
train에는 원본과 약한 background noise, 0.9~1.1배 volume, 최대 50ms time shift를
조합한 변형 하나만 사용합니다. EarlyStopping과 ReduceLROnPlateau는 field가 아닌
ESC-50 fold 4의 `val_pr_auc`를 감시합니다.

`--abnormal-weight-multiplier`는 ESC-50 train 기준 balanced class weight의 abnormal
항목에만 곱합니다. `1.0`, `1.1`, `1.2`를 동일한 `audio_model_v4.keras`에서 각각
독립적으로 실험합니다. 기존 base/output 모델은 덮어쓰지 않습니다.

## 평가와 threshold 선택

```bash
python -m audio_guard.interfaces.cli.evaluate_cli \
  --pipeline v4 \
  --esc50-dir data/ESC-50 \
  --field-data-dir data/dataset \
  --field-splits data/dataset/field_splits.csv \
  --model models/audio_model_v5_general_w1_0.keras \
  --results-dir results/v5_general_w1_0 \
  --thresholds 0.50 0.60 0.70 0.75 0.80 0.85 0.90 \
  --min-field-recall 0.65
```

`threshold_comparison.csv`에는 ESC-50 validation/test의 threshold별 Accuracy,
Precision, Recall, F1, TP, FN, FP, TN이 저장됩니다.
`field_recall_comparison.csv`에는 field abnormal validation/test의 전체 및
event type별 Recall, TP, FN이 별도로 저장됩니다. doorbell은 포함되지 않습니다.
`--min-field-recall`을 지정하면 field validation Recall 조건을 만족하는 후보만
남긴 뒤 ESC-50 validation의 Recall, FN, F1, Precision, FP 순으로 threshold를
선택합니다. `evaluation_summary.json`에는 선택된 threshold의 요약 지표가
저장됩니다. ESC-50 test와 field test는 최종 보고에만 사용합니다.
