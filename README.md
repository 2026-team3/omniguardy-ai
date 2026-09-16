# omniguardy-ai (audio)

Raspberry Pi에서 상시 수집되는 오디오를 분석해 **정상(normal) / 비정상(abnormal)** 을 판정하는 FastAPI 기반 오디오 이상탐지 서버입니다. `omniguardy-backend`(Spring Boot)로부터 3초 내외의 오디오 클립을 받아 Mel 스펙트로그램 CNN으로 추론하고, 결과를 반환합니다.

```
Raspberry Pi → (오디오 수집) → Spring Boot → POST /predict → FastAPI
    → 22050Hz/Mono 변환 → Mel Spectrogram → CNN 추론 → normal / abnormal
```

## 프로젝트 구조

```
src/audio_guard/
├── domain/            # 프레임워크 의존 없는 순수 규칙
│   ├── risk_policy.py     # threshold 기반 normal/abnormal 판정
│   └── pipeline/           # 전처리 파이프라인 전략 (v3 / v4 / v5)
├── application/        # 유스케이스 오케스트레이션
│   ├── analyze_clip.py     # 추론 유스케이스
│   ├── train_model.py      # 학습 유스케이스 (pipeline 인자로 v3/v4/v5 선택)
│   ├── evaluate_model.py
│   └── finetune_model.py
├── infrastructure/     # 외부 기술 세부사항
│   ├── ml/                 # Keras 모델 로딩·추론
│   ├── audio/              # ffmpeg 변환, librosa 로딩
│   └── dataset/            # ESC-50 / 현장 데이터셋 로딩
├── interfaces/
│   ├── api/                 # FastAPI 라우터
│   └── cli/                 # predict / train / evaluate CLI
├── config.py
└── labels.py

main.py          # FastAPI 앱 조립
configs/          # audio_config.json
tests/            # pytest
data/             # 원본 오디오 (git 추적 X)
results/          # 실험 산출물 csv (git 추적 X)
docs/experiments/ # 버전별 실험 기록
```

> 위 구조는 리팩터링 진행 중입니다. 현재 브랜치가 아직 이 구조로 완전히 이전되지 않았다면, 루트의 `main.py` / `feature.py` / `train*.py` 등을 참고하세요.

## 요구 사항

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) (시스템 PATH에 설치되어 있어야 함 — 업로드된 오디오를 22050Hz/Mono WAV로 변환하는 데 사용)

```bash
pip install -r requirements.txt
```

## 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API

| Method | Path | 설명 |
|---|---|---|
| `POST` | `/predict` | 오디오 파일(`multipart/form-data`, key: `file`)을 받아 `{"status": "normal" \| "abnormal" \| "error_*", "probability": float}` 반환 |
| `GET` | `/health` | 로드된 모델, 파이프라인 버전, threshold, sample rate, ffmpeg 경로 확인 |

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.wav"
```

### CLI 추론

```bash
python -m audio_guard.interfaces.cli.predict_cli sample.wav --config configs/audio_config.json
```

## 설정 (`configs/audio_config.json`)

```json
{
  "model_path": "models/audio_model_v3_weight_1_1.keras",
  "pipeline": "v3",
  "sample_rate": 22050,
  "labels": {"normal": 0, "abnormal": 1},
  "esc_abnormal_categories": ["door_wood_knock", "door_wood_creaks", "glass_breaking", "siren", "chainsaw", "footsteps"],
  "threshold": 0.7,
  "max_windows": 8
}
```

- `pipeline`: 사용할 전처리·모델 입력 전략 (`v3` / `v4` / `v5`). API·학습·평가 코드가 모두 이 값 하나로 동작을 분기합니다.
- `threshold`: 이 값 이상의 점수를 abnormal로 판정합니다.
- `labels`, `esc_abnormal_categories`, `sample_rate`는 학습 시점과 반드시 일치해야 하며, 불일치 시 기동 시점에 에러가 발생합니다.

## 파이프라인 버전

| 버전 | 특징 | 상세 |
|---|---|---|
| v3 | 클립 전체를 하나의 (128,128) Mel로 변환 | 최초 버전 |
| v4 | 3초 윈도우로 클립을 분할해 각각 추론, 최대 점수로 판정 | 자세한 내용은 `docs/experiments/v4.md` |
| v5 | 윈도우를 최대 `max_windows`개로 제한한 묶음(bag) 단위 MIL 학습 | 자세한 내용은 `docs/experiments/v5.md` |

## 학습 / 평가 / 파인튜닝

```bash
# 학습 (pipeline은 v3 / v4 / v5 중 선택)
python -m audio_guard.interfaces.cli.train_cli --pipeline v5 --field-splits field_splits.csv

# 평가
python -m audio_guard.interfaces.cli.evaluate_cli --pipeline v5 --config configs/audio_config.json
```

- ESC-50 fold 1~3은 학습, fold 4는 threshold 선택(검증), fold 5는 최종 테스트에 사용합니다.
- 현장 녹음(`data/normal/`, `data/abnormal/`)은 파일 단위 누수를 막기 위해 원본 WAV를 먼저 train/validation/test로 분리한 뒤 윈도우를 생성합니다. 같은 녹음 세션에서 파생된 파일은 `field_splits.csv`의 `group_id`로 묶어 같은 분할에 배정하세요.

```csv
filename,split,group_id
normal/example_correct.wav,train,session_01
abnormal/example_wrong.wav,validation,session_02
```

## 테스트

```bash
pytest tests/
```

## 데이터 / 산출물 관리

- `data/`(원본 오디오)와 `results/`(평가 csv 등 실험 산출물)는 git에 커밋하지 않습니다. 대용량 오디오는 git-lfs나 별도 스토리지에 보관하세요.
- 모델 가중치(`models/*.keras`)도 저장소에 직접 커밋하지 않는 것을 권장합니다.

## 관련 저장소

- [`omniguardy-backend`](https://github.com/2026-team3/omniguardy-backend) — Spring Boot 백엔드. Audio/Vision 분석 결과를 통합해 `SecurityEvent`를 관리하고 Agent AI 위험도 판단을 수행합니다.
