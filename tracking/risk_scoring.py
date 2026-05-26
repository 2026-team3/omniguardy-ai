import pandas as pd

df = pd.read_csv(
    "./results/behavior_features.csv"
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
    # 장시간 체류
    # =========================
    if row["frame_count"] > 400:

        score += 20
        events.append("장시간 체류")

    # =========================
    # 문 앞 배회
    # =========================
    if (
        row["frame_count"] > 300
        and row["move_distance"] < 200
    ):

        score += 30
        events.append("문 앞 배회")

    # =========================
    # 위험 단계
    # =========================
    if score >= 60:
        level = "HIGH"

    elif score >= 30:
        level = "MEDIUM"

    else:
        level = "LOW"

    results.append({

        "video": row["video"],

        "track_id": row["track_id"],

        "events": ", ".join(events),

        "risk_score": score,

        "risk_level": level
    })

result_df = pd.DataFrame(results)

print(result_df.head())

result_df.to_csv(
    "./results/risk_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print("risk_results.csv 저장 완료")