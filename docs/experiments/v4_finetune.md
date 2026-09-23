# V4 기반 현장 위험음 Fine-tuning

`audio_model_v5.keras`는 새로운 MIL 파이프라인이 아닙니다. 기존 v4의 3초
슬라이딩 윈도우와 CNN 구조를 유지한 fine-tuned 모델 파일 버전입니다.

## 데이터 분리

현장 데이터는 다음처럼 둡니다.

```text
data/dataset/
├── abnormal/
└── field_splits.csv
```

`field_splits.csv`는 다음 열을 사용합니다.

```csv
filename,split,group_id
abnormal/handle_pull_near_001.wav,train,door_a_day_01
abnormal/handle_pull_far_001.wav,validation,door_b_night_01
abnormal/handle_pull_far_002.wav,test,door_c_day_01
```

field 데이터는 직접 수집한 abnormal WAV만 사용하며 normal WAV는 요구하지
않습니다. 같은 `group_id`는 하나의 split에만 존재해야 합니다. train,
validation, test에는 각각 abnormal 파일이 최소 하나 이상 필요합니다.
augmentation은 train split에만 적용되며 validation/test 원본 WAV에는 적용되지
않습니다.

## Fine-tuning

```bash
python -m audio_guard.interfaces.cli.finetune_cli \
  --esc50-dir data/ESC-50 \
  --field-data-dir data/dataset \
  --field-splits data/dataset/field_splits.csv \
  --base-model models/audio_model_v4.keras \
  --model-out models/audio_model_v5.keras \
  --epochs 10 \
  --learning-rate 1e-5 \
  --abnormal-weight-multiplier 1.2
```

fine-tuning은 ESC-50 fold 1~3의 normal/abnormal 데이터와 field abnormal train을
함께 사용합니다. field validation/test는 학습에 포함하지 않습니다. field train에는
약한 background noise, 0.8배/1.2배 volume 변형을 적용합니다.

`--abnormal-weight-multiplier`는 기존 balanced class weight의 abnormal 항목에만
곱합니다. `1.0`, `1.2`, `1.5`를 각각 실험하고 validation 결과로 모델 하나를
선택합니다. 최종 test는 선택에 사용하지 않습니다.

## 평가와 threshold 선택

```bash
python -m audio_guard.interfaces.cli.evaluate_cli \
  --pipeline v4 \
  --esc50-dir data/ESC-50 \
  --field-data-dir data/dataset \
  --field-splits data/dataset/field_splits.csv \
  --model models/audio_model_v5.keras \
  --results-dir results/v5 \
  --thresholds 0.70 0.75 0.80 0.85 \
  --min-field-recall 0.65
```

`threshold_comparison.csv`에는 ESC-50 validation/test의 threshold별 Accuracy,
Precision, Recall, F1, TP, FN, FP, TN이 저장됩니다.
`field_recall_comparison.csv`에는 field abnormal validation/test의 threshold별
Recall, TP, FN이 별도로 저장됩니다. `--min-field-recall`을 지정하면 field
validation Recall 조건을 만족하는 후보 중 ESC-50 validation F1이 가장 높은
threshold를 선택합니다. ESC-50 test와 field test는 최종 보고에만 사용합니다.
