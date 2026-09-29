"""MediaPipe HandLandmarker 결과에서 손가락 펴짐 상태를 계산해
규칙 기반으로 제스처를 분류한다. 학습 없이 landmark 좌표 비교만으로 동작한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# MediaPipe Hand landmark 인덱스 (21개 점)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


@dataclass
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def extended_count(self) -> int:
        return sum([self.thumb, self.index, self.middle, self.ring, self.pinky])


def _is_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
    """손목 대비 tip이 pip보다 더 바깥(위)에 있으면 펴진 것으로 판단한다.
    이미지 좌표계는 y가 아래로 갈수록 커지므로, tip.y < pip.y면 편 상태다."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def _thumb_extended(landmarks, handedness_label: str) -> bool:
    """엄지는 위아래가 아니라 좌우로 펴지므로 x좌표로 판단한다.
    handedness_label은 MediaPipe가 이미지 기준으로 주는 'Left'/'Right'."""
    tip_x = landmarks[THUMB_TIP].x
    ip_x = landmarks[THUMB_IP].x
    if handedness_label == "Right":
        return tip_x < ip_x
    return tip_x > ip_x


def get_finger_state(landmarks, handedness_label: str) -> FingerState:
    return FingerState(
        thumb=_thumb_extended(landmarks, handedness_label),
        index=_is_extended(landmarks, INDEX_TIP, INDEX_PIP),
        middle=_is_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP),
        ring=_is_extended(landmarks, RING_TIP, RING_PIP),
        pinky=_is_extended(landmarks, PINKY_TIP, PINKY_PIP),
    )


def classify_gesture(landmarks, handedness_label: str) -> str:
    """finger state 조합으로 제스처 이름을 반환한다.
    victory: 검지+중지만 펴짐 (비상 신호로 사용)
    fist: 전부 오므림 (엄지 포함)
    open_palm: 검지/중지/약지/새끼 4개가 펴짐 (엄지는 판정에서 제외)
    그 외: unknown

    엄지는 위아래가 아니라 좌우로 펴지는 손가락이라 카메라 각도·손 방향에 따라
    _thumb_extended 판정이 자주 흔들린다. open_palm을 엄지까지 포함해서 판정하면
    나머지 4개가 다 펴져도 엄지 하나 때문에 unknown으로 새는 경우가 많아서,
    open_palm은 엄지를 빼고 나머지 4개만으로 판정한다.
    """
    state = get_finger_state(landmarks, handedness_label)

    if state.index and state.middle and not state.ring and not state.pinky:
        return "victory"
    if state.extended_count() == 0:
        return "fist"
    if state.index and state.middle and state.ring and state.pinky:
        return "open_palm"
    return "unknown"


def frame_gesture_from_hands(gestures_this_frame: Iterable[str]) -> str:
    """한 프레임에 여러 손이 잡힐 수 있으므로, 손 여러 개의 제스처를 하나로 합친다.
    fist가 하나라도 있으면 fist를 우선하고, 그 다음 open_palm, 그 외는 none으로 본다.
    (Silent Signal 트리거는 fist -> open_palm 전환 하나만 보면 되므로 단순화)"""
    gestures = list(gestures_this_frame)
    if "fist" in gestures:
        return "fist"
    if "open_palm" in gestures:
        return "open_palm"
    return "none"


class FistToOpenTracker:
    """주먹(fist)을 잠깐 쥐었다가 손을 펴는(open_palm) '전환 동작'을 감지하는 상태 머신.

    victory 사인처럼 정지 자세 하나로 판정하지 않고, 시간 축에서
    fist가 일정 프레임 이상 유지된 뒤 -> 짧은 시간 안에 open_palm으로 바뀌는지를 본다.
    매 프레임 update()에 그 프레임의 제스처("fist"/"open_palm"/"none"/"unknown")를
    넣어주면 되고, 전환이 확정되는 바로 그 프레임에서만 True를 반환한다.
    """

    def __init__(
        self,
        min_fist_frames: int = 5,
        max_transition_gap_frames: int = 15,
        cooldown_frames: int = 30,
    ) -> None:
        # 주먹으로 인정하기까지 최소 연속 프레임 수 (오탐 방지: 스치듯 지나가는 fist 무시)
        self.min_fist_frames = min_fist_frames
        # fist가 풀린 뒤 open_palm이 이 프레임 수 안에 나와야 같은 동작으로 인정
        self.max_transition_gap_frames = max_transition_gap_frames
        # 한 번 감지된 뒤 다음 감지까지 쉬는 프레임 수 (같은 동작으로 여러 번 트리거되는 것 방지)
        self.cooldown_frames = cooldown_frames

        self._fist_streak = 0
        self._armed = False  # fist를 충분히 오래 쥐어서 open_palm을 기다리는 상태
        self._frames_since_fist = 0
        self._cooldown = 0

    def update(self, gesture: str) -> bool:
        """이번 프레임 제스처를 반영한다. 이 프레임에서 전환이 확정되면 True."""
        if self._cooldown > 0:
            self._cooldown -= 1

        if gesture == "fist":
            self._fist_streak += 1
            self._frames_since_fist = 0
            if self._fist_streak >= self.min_fist_frames:
                self._armed = True
            return False

        if self._armed:
            self._frames_since_fist += 1
            if gesture == "open_palm":
                self._armed = False
                self._fist_streak = 0
                self._frames_since_fist = 0
                if self._cooldown == 0:
                    self._cooldown = self.cooldown_frames
                    return True
                return False
            if self._frames_since_fist > self.max_transition_gap_frames:
                # 너무 오래 지나면 같은 동작으로 보지 않고 취소
                self._armed = False
                self._fist_streak = 0
        else:
            self._fist_streak = 0

        return False

    def reset(self) -> None:
        self._fist_streak = 0
        self._armed = False
        self._frames_since_fist = 0
        self._cooldown = 0


class RepeatedFistToOpenTracker:
    """fist -> open_palm 전환을 required_repeats번 연속으로 반복해야 최종 확정하는 트래커.

    한 번의 전환(FistToOpenTracker)은 우연히도 나올 수 있는 동작이라, "손을 쥐었다 펴는
    동작을 연속 2번" 같은 반복 요구로 오탐을 한 번 더 줄이기 위한 래퍼다.
    내부적으로 FistToOpenTracker를 그대로 재사용하되, 반복 사이 간격을 짧게 둬서
    연속 동작을 놓치지 않고 세도록 하고, 반복 사이 간격이 너무 벌어지면 처음부터 다시 센다.
    """

    def __init__(
        self,
        required_repeats: int = 2,
        min_fist_frames: int = 5,
        max_transition_gap_frames: int = 15,
        min_gap_between_reps_frames: int = 3,
        max_gap_between_reps_frames: int = 45,
    ) -> None:
        self.required_repeats = required_repeats
        # 반복 사이에는 굳이 오래 쉴 필요 없으니 쿨다운을 짧게 둔다
        # (한 반복이 끝나자마자 바로 다음 반복의 fist를 셀 수 있어야 하므로)
        self._single = FistToOpenTracker(
            min_fist_frames=min_fist_frames,
            max_transition_gap_frames=max_transition_gap_frames,
            cooldown_frames=min_gap_between_reps_frames,
        )
        # 한 번 전환된 뒤 다음 전환까지 이 프레임 수 안에 들어와야 "연속 반복"으로 인정
        self.max_gap_between_reps_frames = max_gap_between_reps_frames

        self._rep_count = 0
        self._frames_since_last_rep = 0

    @property
    def rep_count(self) -> int:
        """지금까지 연속으로 인정된 반복 횟수 (required_repeats에 도달하면 0으로 리셋됨)."""
        return self._rep_count

    def update(self, gesture: str) -> bool:
        """이번 프레임 제스처를 반영한다. required_repeats번째 반복이 확정되는
        바로 그 프레임에서만 True를 반환하고, 내부 카운트는 자동으로 초기화된다."""
        single_triggered = self._single.update(gesture)

        if self._rep_count > 0:
            self._frames_since_last_rep += 1
            if self._frames_since_last_rep > self.max_gap_between_reps_frames:
                # 다음 반복이 너무 늦게 오면 연속 동작이 아니라고 보고 처음부터 다시 센다
                self._rep_count = 0
                self._frames_since_last_rep = 0

        if single_triggered:
            self._rep_count += 1
            self._frames_since_last_rep = 0
            if self._rep_count >= self.required_repeats:
                self._rep_count = 0
                return True

        return False

    def reset(self) -> None:
        self._single.reset()
        self._rep_count = 0
        self._frames_since_last_rep = 0