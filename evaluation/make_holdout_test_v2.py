"""Create a leakage-safe internal holdout without changing existing manifests.

The generated files live below ``results/evaluation/lstm_v2/splits``.  An
augmented video and its source video always share ``group_id``; this is also
the only group identifier used by train_lstm_v2.py.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {"A18": "A18_20", "A20": "A18_20"}
_AUGMENT = re.compile(r"_aug\d+$", re.IGNORECASE)


def canonical_video(value: object) -> str:
    """Return the source-video name for both originals and ``*_augN`` files."""
    name = Path(str(value)).stem
    return _AUGMENT.sub("", name)


def add_group_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    # Do not rely on annotation_video: it is empty for some mydata rows.  The
    # source prefix avoids accidental grouping of equal basenames from sources.
    frame["group_id"] = frame.apply(
        lambda row: f"{row.get('source', 'unknown')}:{canonical_video(row['video'])}", axis=1
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="splits/annotations/train_annotations_all.csv")
    parser.add_argument("--output-dir", default="results/evaluation/lstm_v2/splits")
    parser.add_argument("--frac", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.frac < 0.5:
        raise SystemExit("--frac must be between 0 and 0.5")

    frame = add_group_id(pd.read_csv(ROOT / args.annotations))
    frame["merged_label"] = frame["label"].replace(LABEL_MAP)
    # A group may contain N1 blocks as well as its action. Select its non-N1
    # action for stratification where available; otherwise it is N1.
    def group_label(labels: pd.Series) -> str:
        actions = labels[labels != "N1"]
        return actions.mode().iat[0] if not actions.empty else "N1"
    groups = frame.groupby("group_id")["merged_label"].agg(group_label)

    selected: set[str] = set()
    for label, ids in groups.groupby(groups):
        ids = ids.sort_index()
        count = max(1, round(len(ids) * args.frac))
        selected.update(ids.sample(n=min(count, len(ids)), random_state=args.seed).index)

    holdout = frame[frame.group_id.isin(selected)].drop(columns="merged_label")
    train = frame[~frame.group_id.isin(selected)].drop(columns="merged_label")
    overlap = set(train.group_id) & set(holdout.group_id)
    assert not overlap, f"group leakage into holdout: {sorted(overlap)[:5]}"

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train_annotations_holdout_excluded.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(output / "holdout_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"group_id": groups.index, "stratify_label": groups.values,
                  "split": ["holdout" if x in selected else "train" for x in groups.index]}).to_csv(
        output / "group_split.csv", index=False, encoding="utf-8-sig")
    print(f"train rows={len(train)}, groups={train.group_id.nunique()}")
    print(f"holdout rows={len(holdout)}, groups={holdout.group_id.nunique()}")
    print(holdout.label.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
