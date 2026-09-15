# Audio V4 실험 안내

이 실험은 평가 데이터 누수와 학습·평가 전처리 불일치를 해결하기 위해 작성되었습니다. 원격 학습과 독립 테스트를 실행하기 전에는 F1 또는 recall 향상을 주장할 수 없습니다.

1. 원격 학습 환경에 ESC-50을 준비하고 `meta/esc50.csv`와 `audio/` 디렉터리를 함께 둡니다.
2. 현장 WAV 파일을 `dataset/normal`(정상 도어락 소리)과 `dataset/abnormal`(틀린 도어락 소리)에 넣습니다. 다음 명령으로 학습합니다.

   ```bash
   python train_v4.py --esc50-dir /path/to/ESC-50 --field-data-dir dataset
   ```

   ESC-50 fold 1~3과 현장 학습 파일로 학습하고, fold 4와 별도로 분리한 현장 원본 파일로 검증합니다. Fold 5는 학습과 검증에 사용하지 않습니다.
3. 다음 명령으로 평가합니다.

   ```bash
   python evaluate_v4.py --esc50-dir /path/to/ESC-50 --field-data-dir dataset
   python -m unittest test_feature_windows.py test_field_dataset.py
   ```

   Threshold는 fold 4와 현장 검증 파일에서 선택한 뒤 fold 5에 한 번 적용합니다. 배포 전 기존 시스템과 F1, recall, FN, FP를 비교합니다. 단위 테스트는 윈도우 생성과 폴더 라벨 매핑을 확인합니다.
4. V4는 과거의 윈도우 단위 라벨 학습 실험으로 유지합니다. 새 학습과 평가는 V5를 사용합니다. API와 CLI는 기본적으로 `audio_config.json`을 읽어 기존 V3 모델과 threshold 0.7을 유지합니다. 새 모델을 배포하려면 검증된 모델 설정 JSON을 `AUDIO_CONFIG_PATH`에 지정합니다.

현장 검증셋에는 원본 녹음이 몇 개밖에 없어 신뢰할 만한 현장 테스트 결과로 보기 어렵습니다. 같은 녹음 세션에서 나온 여러 파일은 한 데이터 분할에 묶여야 하지만, 현재 기본 분할은 세션이 아닌 파일 기준입니다. ESC-50 fold 5 결과도 현장 오디오가 아닌 ESC-50의 독립 테스트 결과입니다.

별도로 수집한 라즈베리파이 3초 녹음으로 기존 모델과 새 모델을 비교하세요. 특히 사이렌·문 삐걱임·발소리의 미탐(FN), 정상 소음의 오탐(FP)을 확인하세요. 현장 음원의 recall이 안전상 우선이라면 별도의 현장 검증셋에서 최대 오탐 또는 최소 precision 조건을 정해 threshold를 선택하고, 최종 테스트셋에서 threshold를 다시 조정하지 마세요.

`finetune.py`와 `finetune_v3.py`는 과거 실험이며 V4에서는 사용하지 않습니다. 두 모델을 각자의 학습 녹음으로 평가해서는 안 됩니다.
