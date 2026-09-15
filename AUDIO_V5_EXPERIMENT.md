# Audio V5: 파일 단위 MIL 학습 실험

V5는 새 모델과 특징 추출 버전입니다. 원격 학습, 독립 평가, 설정 변경을 마치기 전에는 API에서 사용되지 않습니다.

## 데이터 분할과 누수 방지

ESC-50 fold 1~3은 학습, fold 4는 검증, fold 5는 최종 테스트에 사용합니다. 현장 `dataset/normal/*.wav`는 라벨 0, `dataset/abnormal/*.wav`는 라벨 1입니다. 현장 원본 WAV를 학습·검증·테스트로 먼저 분리한 뒤 특징 윈도우를 생성하므로, 같은 파일의 윈도우가 서로 다른 분할에 섞이지 않습니다.

기본 파일 단위 분할은 경고를 출력합니다. 별도 WAV가 같은 녹음 세션에서 파생됐는지 확인할 수 없기 때문입니다. 신뢰할 만한 현장 성능 측정을 위해 모든 WAV를 기재한 `field_splits.csv`를 `--field-splits`로 전달하세요.

```csv
filename,split,group_id
normal/example_correct.wav,train,session_01
abnormal/example_wrong.wav,validation,session_02
```

동일한 원본 녹음·세션·기기 조건에서 만든 모든 파일에 같은 `group_id`를 부여합니다. 스크립트는 한 그룹이 여러 분할에 걸쳐 있으면 중단하고, 각 분할에 정상과 비정상 라벨이 모두 있는지도 확인합니다. 현재 라벨당 WAV가 17개뿐이라 독립 현장 테스트셋도 작습니다. 서로 다른 조건의 원본 녹음을 추가 수집해야 합니다.

## 학습과 평가

```bash
python train_v5.py --esc50-dir /path/to/ESC-50 --field-data-dir dataset --field-splits field_splits.csv
python evaluate_v5.py --esc50-dir /path/to/ESC-50 --field-data-dir dataset --field-splits field_splits.csv --min-field-recall 0.80
python -m unittest test_field_dataset.py test_feature_windows.py test_audio_v5.py
```

`--min-field-recall`은 선택 사항입니다. 값 `0.80`은 실행 예시일 뿐, 검증된 기본 기준이 아닙니다. 이 옵션을 사용하지 않으면 통합 검증셋의 F1이 가장 높은 threshold를 선택합니다. ESC-50과 현장 음원의 precision, recall, F1, 혼동행렬 값(TP·FN·FP·TN)을 검증과 테스트에서 각각 출력합니다. 최종 테스트 라벨은 threshold 선택에 사용하지 않습니다.

V5는 녹음 파일마다 라벨 하나를 학습합니다. 모델이 각 3초 윈도우에 점수를 매기고 가장 높은 점수로 파일을 판정합니다. 긴 녹음이 학습에 과도한 영향을 주지 않도록 파일당 균등 간격의 윈도우를 최대 8개 사용합니다. 다만 파일 라벨만으로는 이상 이벤트의 정확한 발생 시각을 알 수 없습니다. 선택하지 않은 윈도우에만 이벤트가 있다면 놓칠 수 있으므로, 발생 시각을 주석으로 기록하거나 이벤트 중심의 3초 클립을 수집하는 것이 좋습니다.

`--field-weight`는 학습 중 현장 데이터의 상대 가중치를 조절합니다. 가중치를 높인다고 성능이 반드시 좋아지는 것은 아니므로 독립 현장 테스트셋에서 비교해야 합니다.

V5 특징 추출은 3초 Mel 시간축 전체를 128열로 보간해 끝부분 프레임을 버리지 않습니다. 짧은 오디오는 특징 추출 전에 무음으로 채웁니다. 이는 기존과 다른 입력 분포이므로 V3/V4 모델 가중치에 V5 전처리를 적용하면 안 됩니다.

## 배포

평가가 선택한 threshold와 모델 경로, 전처리 버전, 샘플레이트, 라벨 매핑을 `models/audio_model_v5.config.json`에 저장합니다. ESC-50과 현장 독립 테스트 결과를 모두 검토한 뒤, API(`main.py`)와 CLI(`predict.py`)에서 이 JSON의 절대 경로를 `AUDIO_CONFIG_PATH`에 지정합니다.

기본 `audio_config.json`은 기존 V3 모델과 threshold 0.7을 유지합니다. `GET /health`에서는 활성 모델, 전처리 버전, threshold를 확인할 수 있습니다. ESC-50 결과만 보고 V5를 배포하지 말고, 독립적으로 수집한 라즈베리파이 3초 현장 녹음도 테스트하세요.
