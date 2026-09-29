#원본 recording 관계를 보존하며 70/15/15 split을 생성

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from audio_guard.labels import LABELS

RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
INPUT_COLUMNS = {
    "recording_id",
    "path",
    "class_name",
    "dataset",
    "source_id",
    "session_id",
    "group_id",
}


def _connected_components(rows):
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen = {}
    for index, row in enumerate(rows):
        for identity in ("source_id", "session_id", "group_id"):
            key = (row["dataset"], identity, row[identity])
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    components = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)
    return list(components.values())


def assign_group_splits(rows, seed=42):
    if len(_connected_components(rows)) < 3:
        raise ValueError("At least three independent recording groups are required")
    totals = Counter(row["class_name"] for row in rows)
    targets = {
        split: {name: totals[name] * ratio for name in LABELS}
        for split, ratio in RATIOS.items()
    }
    counts = {split: Counter() for split in RATIOS}
    assignments = [None] * len(rows)
    components = _connected_components(rows)
    rng = np.random.default_rng(seed)
    rng.shuffle(components)
    components.sort(key=len, reverse=True)

    for component in components:
        component_counts = Counter(rows[index]["class_name"] for index in component)

        def cost(candidate):
            total_cost = 0.0
            for split in RATIOS:
                added = component_counts if split == candidate else Counter()
                total_cost += sum(
                    (counts[split][name] + added[name] - targets[split][name]) ** 2
                    / max(1.0, targets[split][name])
                    for name in LABELS
                )
                total_target = len(rows) * RATIOS[split]
                total_after = sum(counts[split].values()) + (
                    len(component) if split == candidate else 0
                )
                total_cost += (
                    (total_after - total_target) ** 2 / max(1.0, total_target)
                )
            return total_cost

        selected = min(RATIOS, key=cost)
        counts[selected].update(component_counts)
        for index in component:
            assignments[index] = selected
    if set(assignments) != set(RATIOS):
        raise ValueError("Not enough independent groups to populate every split")
    return assignments


def prepare_manifest(input_path: Path, output_path: Path, seed=42):
    with Path(input_path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not INPUT_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(INPUT_COLUMNS - set(reader.fieldnames or ()))
            raise ValueError(f"Input metadata is missing columns: {missing}")
        rows = list(reader)
        fieldnames = [name for name in reader.fieldnames if name != "split"] + ["split"]
    if not rows:
        raise ValueError("Input metadata is empty")
    if any(not (row.get(column) or "").strip() for row in rows for column in INPUT_COLUMNS):
        raise ValueError("Input metadata contains an empty required value")
    if any(row["class_name"] not in LABELS for row in rows):
        raise ValueError("Input metadata contains an unknown class")
    assignments = assign_group_splits(rows, seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row, split in zip(rows, assignments):
            writer.writerow({**row, "split": split})
    return output_path
