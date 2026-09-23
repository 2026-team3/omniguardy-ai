# omniguardy-ai audio

FastAPI와 TensorFlow를 사용하는 오디오 이상 탐지 서버입니다. Spring이 전송한
오디오 파일을 요청마다 독립적으로 분석해 `normal` 또는 `abnormal`로 분류하며,
v3·v4 전처리 전략을 선택해 사용할 수 있습니다.

## 설치와 실행

Python 3.10 또는 3.11과 FFmpeg가 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
uvicorn main:app --host 0.0.0.0 --port 8000
```

기본 설정 파일은 `configs/audio_config.json`입니다. 다른 설정은 `AUDIO_CONFIG_PATH` 환경 변수로 지정할 수 있습니다.

```bash
AUDIO_CONFIG_PATH=configs/audio_config.json uvicorn main:app
```

## 구조

```text
src/audio_guard/
├── domain/                  # 값 객체, 위험 판정 정책, 전처리 전략
│   └── pipeline/            # v3/v4 특징 추출
├── application/             # 분석, 학습, fine-tuning, 평가 유스케이스
├── infrastructure/          # Keras, FFmpeg, librosa, 데이터셋 구현
├── interfaces/
│   ├── api/                 # FastAPI 라우터와 스키마
│   └── cli/                 # 예측, 학습, fine-tuning, 평가 CLI
├── config.py
└── labels.py

configs/                     # 런타임 설정
data/                        # 로컬 원본 데이터, Git 추적 제외
results/                     # 평가·실험 산출물, Git 추적 제외
docs/experiments/            # 버전별 실험 기록
tests/                       # pytest 테스트
main.py                      # FastAPI 의존성 조립
```

## API

### `POST /predict`

multipart/form-data의 `file` 필드로 오디오 파일을 전송합니다. 서버는 FFmpeg로 22,050Hz 모노 WAV로 변환한 뒤 설정된 파이프라인과 모델로 분석합니다.

정상 응답:

```json
{
  "status": "normal",
  "probability": 0.12
}
```

`status`는 `normal`, `abnormal` 또는 다음 오류 값입니다.

`abnormal`은 `esc_abnormal_categories`에 속하는 비전 트리거 이벤트가
임계치 이상으로 검출됐다는 뜻입니다. Spring은 `status == "abnormal"`인
경우에만 비전을 실행하고 `normal` 및 모든 오류 값에서는 실행하지 않습니다.

- `error_empty_file`: 빈 파일
- `error_ffmpeg`: 오디오 변환 실패
- `error_short_audio`: 1초 미만 오디오
- `error`: 그 밖의 처리 오류

### `GET /health`

현재 모델 이름, 파이프라인, 임계치, 샘플레이트와 FFmpeg 경로를 반환합니다.

## 설정

`configs/audio_config.json`의 필드는 다음과 같습니다.

| 필드 | 설명 |
| --- | --- |
| `model_path` | 설정 파일을 기준으로 한 Keras 모델 경로 |
| `pipeline` | `v3`, `v4` 중 하나 |
| `sample_rate` | 모델 입력 샘플레이트. 현재 22050 |
| `labels` | `normal`, `abnormal` 라벨 번호 |
| `esc_abnormal_categories` | ESC-50에서 비정상으로 취급할 카테고리 |
| `threshold` | abnormal 판정 임계치 |

## 파이프라인

| 버전 | 입력 단위 | 특징 및 판정 방식 | 기본 모델 출력 |
| --- | --- | --- | --- |
| v3 | 오디오 한 건 | 앞부분을 고정 폭 Mel `(128, 128)`로 변환 | 단일 점수 |
| v4 | 3초 슬라이딩 윈도우 | 끝 구간까지 포함하고 윈도우 최대 점수 사용 | 윈도우별 점수 |

실시간 분석에는 v4를 사용합니다. 파일을 3초 윈도우로 나누어 각 윈도우를
독립적으로 추론하고, 가장 높은 점수를 파일의 최종 점수로 사용합니다. 세부
실험 기록은 [v4](docs/experiments/v4.md),
[v4 기반 fine-tuning](docs/experiments/v4_finetune.md)을 참고합니다.

## 명령행 사용법

기본 학습은 ESC-50을 사용하며, `esc_abnormal_categories`에 포함된 클래스만
`abnormal`로 취급합니다. 현장 데이터는 아래 fine-tuning과 평가 명령에서
선택적으로 추가합니다.

```bash
# 학습
python -m audio_guard.interfaces.cli.train_cli \
  --pipeline v4 \
  --esc50-dir data/ESC-50 \
  --model-out models/audio_model_v4.keras

# 평가
python -m audio_guard.interfaces.cli.evaluate_cli \
  --pipeline v4 \
  --esc50-dir data/ESC-50 \
  --model models/audio_model_v4.keras

# 현장 데이터로 v4 fine-tuning (각 실험은 동일한 v4에서 시작)
python -m audio_guard.interfaces.cli.finetune_cli \
  --esc50-dir data/ESC-50 \
  --field-data-dir data/dataset \
  --field-splits data/dataset/field_splits.csv \
  --base-model models/audio_model_v4.keras \
  --model-out models/audio_model_v5_general_w1_0.keras \
  --epochs 5 \
  --learning-rate 5e-6 \
  --abnormal-weight-multiplier 1.0

# 단일 파일 예측
python -m audio_guard.interfaces.cli.predict_cli sample.wav \
  --config configs/audio_config.json
```

v4는 ESC-50 fold 1~3의 target class를 abnormal, 나머지를 normal로 학습합니다.
현장 fine-tuning은 `dataset/abnormal`의 위험음만 추가하며, 원본 WAV를 녹음 세션
단위로 train/validation/test로 분리한 `field_splits.csv`를 사용합니다. 초인종은
`event_type=doorbell`, `split=ignored`로 지정하여 학습과 평가에서 제외합니다.
전체 분류 성능은 ESC-50 validation/test의 Accuracy, Precision, Recall, F1로
평가하고, 현장 적응 성능은 field abnormal validation/test의 전체 및 event type별
Recall과 FN으로 따로 평가합니다. 평가가 생성한 모델별 `.config.json`을 런타임의
`AUDIO_CONFIG_PATH`로 사용합니다.

## 테스트

```bash
pip install pytest
pytest -q
```

GitHub Actions도 Python 3.11, FFmpeg와 동일한 pytest 명령을 사용합니다.

## 데이터와 산출물 정책

- ESC-50은 `data/ESC-50/`에 둡니다.
- 직접 수집한 현장 위험음은 `data/dataset/abnormal/`에 두고,
  `field_splits.csv`로 train/validation/test/ignored 녹음 세션과 event type을
  관리합니다.
- 직접 수집한 normal 데이터는 필요하지 않으며 ESC-50의 normal 데이터를
  fine-tuning에서도 유지합니다.
- 평가 CSV와 오류 분석 파일은 `results/`에 생성합니다.
- `data/`, `results/`, `models/`는 Git에 커밋하지 않습니다.
- 재현에 필요한 설정, 코드와 실험 설명만 Git으로 관리합니다.
