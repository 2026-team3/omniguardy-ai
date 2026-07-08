import pandas as pd

df = pd.read_csv(
    "./results/merged_features(train).csv"
)

results = []

for _, row in df.iterrows():

    score = 0
    events = []

    # =========================
    # 일반 이동
    # =========================
    if row["move_distance"] > 300:
        score += 5
        events.append("일반 이동")

    # =========================
    # 반복 접근 / 배회
    # 현관 앞 반복 이동
    # =========================
    if (
        row["movement_range"] < 250
        and row["trajectory_variance"] > 200
    ):
        score += 30
        events.append("반복 접근")

    # =========================
    # 문 앞 장시간 체류
    # =========================
    if (
        row["frame_count"] > 300
        and row["move_distance"] < 250
    ):
        score += 40
        events.append("문 앞 배회")

    # =========================
    # 손 반복 motion
    # 초인종 / 도어락 반복
    # =========================
    if row["hand_motion"] > 2.0:
        score += 35
        events.append("손 반복 행동")

    # =========================
    # 강한 body motion
    # 위협 / 발로 참
    # =========================
    if row["body_motion"] > 1.0:
        score += 50
        events.append("강한 신체 움직임")

    # =========================
    # 팔 뻗음
    # 도어락 조작 가능성
    # =========================
    if row["arm_extension"] > 0.18:
        score += 25
        events.append("문 조작 행동")

    # =========================
    # 상체 방향 이상
    # 카메라 가림 / 내부 확인
    # =========================
    if abs(row["upper_body_angle"]) > 45:
        score += 20
        events.append("비정상 상체 방향")

    # =========================
    # 조합 bonus
    # =========================
    # 도어락 반복 + 손 반복
    if (
        row["arm_extension"] > 0.18
        and row["hand_motion"] > 2.0
    ):

        score += 35
        events.append("반복 도어락 조작")

    # 배회 + 체류
    if (
        row["frame_count"] > 300
        and row["trajectory_variance"] > 200
    ):
        score += 40
        events.append("수상한 체류")

    # =========================
    # 최종 위험도
    # =========================
    if score >= 90:
        level = "HIGH"
    elif score >= 35:
        level = "MEDIUM"
    else:
        level = "LOW"

    # =========================
    # 저장
    # =========================
    results.append({
        "video": row["video"],
        "events": ", ".join(events),
        "risk_score": score,
        "risk_level": level
    })

result_df = pd.DataFrame(results)
print(result_df.head())

result_df.to_csv(
    "./results/risk_results(train).csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("risk_results(train).csv 저장 완료")