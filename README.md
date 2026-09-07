# 현관 행동 인식 Vision 파이프라인

현관 카메라 영상에서 정상 활동과 이상 행동을 블록 단위로 분류하는 Vision 파이프라인입니다.

## 분류 라벨

| 라벨 | 의미                                                   |
| ---- | ------------------------------------------------------ |
| N1   | 정상 출입 활동: 거주자 출입, 일반 배달, 대기, 떠남      |
| A17  | 초인종 반복 누르기                                      |
| A18  | 문 열기 반복 시도                                       |
| A19  | 문 반복 발로 차기                                        |
| A20  | 문 안쪽을 엿보려는 시도                                  |
| A21  | 장시간 또는 반복적인 문 두드림(노크)                     |

## 데이터 분할

| 분할       | 사용 데이터                                                           |
| ---------- | ---------------------------------------------------------------------- |
| Train      | AIHub TL/TS, mydata, 그리고 원본 train 데이터에서 파생된 증강 데이터   |
| Validation | AIHub VL/VS 원본 영상만 사용                                           |
| Test       | 별도의 전시장 카메라 영상 또는 추후 구축될 현장 홀드아웃 영상          |

mydata의 `normal`과 `delivery`는 N1로, `lookingInside`는 A20으로 사용됩니다.

## 처리 파이프라인

```text
영상
→ YOLO 사람 탐지 + BoT-SORT 트래킹
→ MediaPipe Pose 추출
→ 어노테이션 블록 단위로 행동/자세 특징 집계
→ XGBoost 분류
→ N1 / A17~A21 예측
```

* 트래킹과 pose 추출은 영상당 한 번만 수행됩니다.
* 특징 추출과 학습은 각 어노테이션 블록의 `start_frame`, `end_frame` 범위를 사용합니다.
* 증강된 학습 데이터는 원본 블록 경계를 그대로 사용합니다.

## 실시간 Vision 및 Agent 연동

`camera/` 패키지는 PC 웹캠, 영상 파일, 또는 라즈베리파이 카메라에서 학습된 Vision 파이프라인을 실행합니다. 분석 윈도우마다 이벤트를 생성하며, 이상 이벤트를 Agent API로 전송할 수 있습니다.

```text
카메라 프레임
→ YOLOv8n 사람 탐지 + BoT-SORT 트래킹
→ MediaPipe Pose 특징 추출
→ XGBoost 행동 분류
→ Vision 이벤트 JSON
→ Agent API
```

| 경로 | 역할 |
| ---- | ---- |
| `camera/realtime_vision.py` | 커맨드라인 진입점 |
| `camera/sources.py` | PC 웹캠, 영상 파일, Pi 카메라 입력 |
| `camera/features.py` | 트래킹 및 pose 특징 추출 |
| `camera/classifier.py` | XGBoost 행동 분류 |
| `camera/events.py` | 이벤트 생성, 로컬 로깅, Agent HTTP 전송 |
| `camera/runner.py` | 실시간 파이프라인 오케스트레이션 |

### PC 웹캠에서 실행

프로젝트 루트에서 가상환경을 활성화한 뒤 다음을 실행합니다:

```powershell
.\.venv\Scripts\python.exe .\camera\realtime_vision.py --source 0
```

미리보기를 중단하려면 `q` 또는 `Esc`를 누릅니다. 이벤트는 `outputs/realtime_events.jsonl`에 기록됩니다. 기존 영상으로 테스트하려면 `0` 대신 파일 경로를 넣습니다.

```powershell
.\.venv\Scripts\python.exe .\camera\realtime_vision.py --source .\sample.mp4 --no-display
```

### Agent API로 이벤트 전송

기본적으로는 이상 행동 라벨만 전송됩니다. `N1` 이벤트까지 전송하려면 `--publish-normal`을 추가합니다.

```powershell
.\.venv\Scripts\python.exe .\camera\realtime_vision.py --source 0 --agent-url http://127.0.0.1:8001/events/vision
```

이벤트 예시:

```json
{
  "source": "vision",
  "event": "vision.suspicious_behavior_detected",
  "prediction": {
    "label": "A18",
    "behavior_confidence": 0.86,
    "risk_level": "HIGH"
  },
  "detection": {
    "person_count": 1,
    "detection_confidence": 0.93
  }
}
```

### 라즈베리파이 전환

* Pi에 연결된 USB 웹캠에는 `--source 0`을 사용합니다.
* Pi CSI 카메라의 경우 Pi에 `picamera2`를 설치한 뒤 `--source picamera --no-display`를 사용합니다.
* 통합 시스템에서는 Spring Boot가 Audio 또는 장치 신호 트리거 이후 `CAMERA_START` MQTT 명령을 전송해야 합니다. Pi는 요청된 클립을 녹화하여 Spring Boot에 업로드하고, Vision은 카메라를 계속 켜두는 대신 업로드된 영상을 분석합니다.


## 현재 베이스라인 결과

AIHub VL/VS 홀드아웃 검증 결과:

* Accuracy: **86.36%**
* Macro F1-score: **68.82%**
* Weighted F1-score: **86.20%**

N1이 데이터의 큰 비중을 차지하기 때문에, Accuracy 단독이 아니라 Macro F1-score를 함께 평가합니다. 이 수치는 validation 성능이며, 최종 일반화 성능은 전시장 카메라 영상 또는 별도의 현장 테스트 영상으로 추가 평가해야 합니다.

## Git 관리

원본 및 증강 영상, 실행 결과, 자동 생성된 매니페스트, 모델 가중치는 `.gitignore`로 Git에서 제외됩니다. 재현에 필요한 코드와 설정 파일만 Git으로 추적됩니다.
