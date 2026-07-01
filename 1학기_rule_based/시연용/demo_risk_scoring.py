import pandas as pd

df = pd.read_csv(
    "./results/시연용/merged_features(test).csv"
)

results = []

for _, row in df.iterrows():

    score = 0
    events = []

    # =========================
    # 일반 이동
    # =========================
    if (row["move_distance"] > 300
        and row["avg_speed"] > 15):
        score += 3
        events.append("일반 이동")

    # =========================
    # 반복 접근 / 배회
    # 현관 앞 반복 이동
    # =========================
    if (
        row["movement_range"] < 250
        and row["trajectory_variance"] > 200
    ):
        score += 15
        events.append("반복 접근")

    # =========================
    # 문 앞 장시간 체류
    # =========================
    if (
        row["frame_count"] > 300
        and row["move_distance"] < 250
    ):
        score += 20
        events.append("문 앞 배회")

    # =========================
    # 손 반복 motion
    # 초인종 / 도어락 반복
    # =========================
    if row["hand_motion"] > 4.0:
        score += 15
        events.append("손 반복 행동")

    # =========================
    # 강한 body motion
    # 위협 / 발로 참
    # =========================
    if row["body_motion"] > 4.0:
        score += 25
        events.append("강한 신체 움직임")

    # =========================
    # 팔 뻗음
    # 도어락 조작 가능성
    # =========================
    if row["arm_extension"] > 0.5:
        score += 10
        events.append("문 조작 행동")

    # =========================
    # 상체 방향 이상
    # 카메라 가림 / 내부 확인
    # =========================
    if abs(row["upper_body_angle"]) > 55:
        score += 15
        events.append("비정상 상체 방향")

    print(row["video"])
    print("person_count:", row["person_count"])
    print("avg_speed:", row["avg_speed"])
    print("frame_count:", row["frame_count"])   
    print()
    # 사용자 뒤 외부인 접근 (현재는 그냥 인원수 다 체크)
    if (
        row["person_count"] >= 3
        and row["avg_speed"] > 10
        and row["avg_speed"] < 25
        and row["frame_count"] > 80
    ):
        score += 75
        events.append("사용자 뒤 접근")

    # 카메라 가림 의심
    if (
        row["arm_extension"] > 0.55
        and abs(row["upper_body_angle"]) > 50
        and row["avg_speed"] < 15
    ):
        score += 25
        events.append("카메라 가림")

    # =========================
    # 조합 bonus
    # =========================
    # 도어락 반복 + 손 반복
    if (
        row["arm_extension"] > 0.5
        and row["hand_motion"] > 5.5
        and row["frame_count"] > 180
    ):

        score += 15
        events.append("반복 도어락 조작")

    # 배회 + 체류
    if (
        row["frame_count"] > 300
        and row["trajectory_variance"] > 200
    ):
        score += 20
        events.append("수상한 체류")

    # =========================
    # 최종 위험도
    # =========================
    if score >= 80:
        level = "HIGH"
    elif score >= 50:
        level = "MIDDLE"
    elif score >= 20:
        level = "LOW"
    else: 
        level = "NORMAL"

    # =========================
    # 저장
    # =========================
    results.append({
    "module": "vision",
    "video": row["video"],
    "events": events,
    "risk_score": score,
    "risk_level": level,
})

result_df = pd.DataFrame(results)
print(result_df.head())

result_df.to_csv(
    "./results/시연용/risk_results(test).csv",
    index=False,
    encoding="utf-8-sig"
)

import json

# =========================
# JSON 변환
# =========================
json_result = result_df.to_dict(
    orient="records"
)

# =========================
# JSON 저장
# =========================
with open(
    "./results/시연용/risk_results(test).json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_result,
        f,
        ensure_ascii=False,
        indent=4
    )

print("risk_results(test).json 저장 완료")

print()
print("risk_results(test).csv 저장 완료")