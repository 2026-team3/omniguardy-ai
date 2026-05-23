import os
import json

label_dir = "./VL"  # 다운받은 라벨 폴더 경로

actions = set()

for file in os.listdir(label_dir):
    if file.endswith(".json"):
        with open(os.path.join(label_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        action = data.get("action", None)
        if action:
            actions.add(action)

print("데이터셋에 있는 행동 종류:", actions)