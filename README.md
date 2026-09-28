# omniguardy-ai audio

실시간 문 소리를 `background`, `knock`, `handle`로 분류하는 TensorFlow/FastAPI
서비스입니다. Spring은 `KNOCK_EVENT` 또는 `HANDLE_EVENT`를 받으면 카메라를
켜고, `NO_EVENT`에는 반응하지 않습니다.

## 런타임 동작

- 입력은 FFmpeg로 22,050 Hz mono PCM WAV로 변환합니다.
- 1초 window와 0.2초 hop으로 Mel spectrogram `(128, 128, 1)`을 만듭니다.
- CNN은 window마다 3-class softmax 확률을 반환합니다.
- Knock/Handle 확률을 class별 threshold와 비교합니다.
- 하나라도 threshold를 넘은 첫 window에서 즉시 이벤트를 반환합니다.
- 이벤트가 발생한 뒤에만 cooldown을 적용해 겹치는 window의 중복 이벤트를 막습니다.
- 누적 risk, voting, 연속 검출 조건은 사용하지 않습니다.

두 event class가 같은 window에서 모두 threshold를 넘으면 확률이 높은 class가
선택됩니다. Background 확률이 가장 높더라도 Knock/Handle이 해당 threshold를
넘으면 이벤트가 발생합니다.

## 설치와 API 실행

Python 3.10 또는 3.11과 FFmpeg가 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
uvicorn main:app --host 0.0.0.0 --port 8000
```

기본 설정은 `configs/audio_config.json`이며 `AUDIO_CONFIG_PATH`로 다른 runtime
config를 지정할 수 있습니다.

### `POST /predict`

`multipart/form-data`의 `file`로 오디오를 전송합니다.

```json
{
  "status": "KNOCK_EVENT",
  "predicted_class": "knock",
  "probabilities": {
    "background": 0.03,
    "knock": 0.94,
    "handle": 0.03
  },
  "window_start_seconds": 0.4,
  "cooldown_suppressed": false
}
```

`status`는 `KNOCK_EVENT`, `HANDLE_EVENT`, `NO_EVENT` 또는 `error_*`입니다.

## 데이터 manifest

Clip random split은 금지됩니다. 먼저 다음 열을 가진 CSV를 준비합니다.

```text
path,class_name,dataset,source_id,session_id,group_id,event_start_seconds,event_end_seconds
```

같은 원본, 녹음 세션 또는 그룹에서 파생된 파일은 동일한 ID를 사용해야 합니다.
70/15/15 split은 이 연결 관계를 하나의 component로 묶은 뒤 생성합니다.

```bash
python -m audio_guard.interfaces.cli.prepare_dataset_cli \
  --input data/metadata.csv \
  --output data/dataset_manifest.csv
```

최종 manifest에는 `split` 열이 추가됩니다. Loader는 `source_id`, `session_id`,
`group_id` 중 하나라도 여러 split에 걸치면 실패합니다. Augmentation은 split 완료
후 train에만 적용됩니다.

직접 수집한 문고리 소리를 Handle의 중심 데이터로 사용합니다. ESC-50의
`door_wood_knock`은 Knock 후보이며, 그 밖의 생활소음은 Background/hard negative로
사용할 수 있습니다. DCASE task별 metadata는 `filename,event_label` 형식으로 먼저
정규화합니다. Door slam/close, 책상·벽 노크, 열쇠, 물체 낙하 등은 Background에
명시적으로 포함해야 합니다.

긴 positive 녹음에는 `event_start_seconds`, `event_end_seconds`를 기록하십시오.
이 구간과 겹치지 않는 window는 Background로 학습됩니다. 시간 annotation이 없는
positive clip은 모든 window가 해당 event class로 처리되므로 event 중심으로 잘라야
합니다.

## 학습, threshold 확정, 최종 평가

```bash
# Adam 1e-3, batch 32, max epoch 50
# EarlyStopping patience 7 + ReduceLROnPlateau
python -m audio_guard.interfaces.cli.train_cli \
  --manifest data/dataset_manifest.csv \
  --model-out models/door_event_model.keras

# validation만 사용해 Knock/Handle threshold를 각각 확정
python -m audio_guard.interfaces.cli.calibrate_cli \
  --manifest data/dataset_manifest.csv \
  --model models/door_event_model.keras \
  --config-out models/door_event_model.config.json

# 모델과 threshold를 고정한 뒤 test를 한 번만 평가
python -m audio_guard.interfaces.cli.evaluate_cli \
  --manifest data/dataset_manifest.csv \
  --model models/door_event_model.keras \
  --config models/door_event_model.config.json
```

최종 평가는 Accuracy, class별 Precision/Recall/F1, Macro F1, Confusion Matrix와
Background→Knock/Handle false positive를 기록합니다. 실제 환경 평가는 timestamp가
있는 연속 녹음으로 Event Precision/Recall, false trigger/hour와 detection latency를
추가 측정합니다.

## 테스트

```bash
pip install pytest
pytest -q
```

원본 데이터, 학습 모델과 평가 결과는 Git에 포함하지 않습니다.
