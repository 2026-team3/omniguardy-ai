# Audio V5: file-label MIL experiment

V5 is a new model and feature version. It is **not** active in the API until
remote training, independent evaluation, and an explicit config switch.

## Data and leakage

ESC-50 folds 1-3 train, fold 4 validates, and fold 5 is untouched test data.
Field `dataset/normal/*.wav` is label 0 and `dataset/abnormal/*.wav` is label 1.
Field original WAVs are split into train/validation/test before feature windows
are generated. A file-level fallback split emits a warning; it cannot prove
that separate WAVs from one recording session stay together. For reliable
field metrics, provide `--field-splits field_splits.csv` with every WAV listed:

```csv
filename,split,group_id
normal/example_correct.wav,train,session_01
abnormal/example_wrong.wav,validation,session_02
```

Use one `group_id` for every take or derived clip from the same original
recording/session/device condition. The script rejects a group crossing splits
and requires both labels in each split. With only 17 WAVs per label, the field
test is still small and should be expanded with independent recordings.

## Training and evaluation

```bash
python train_v5.py --esc50-dir /path/to/ESC-50 --field-data-dir dataset --field-splits field_splits.csv
python evaluate_v5.py --esc50-dir /path/to/ESC-50 --field-data-dir dataset --field-splits field_splits.csv --min-field-recall 0.80
python -m unittest test_field_dataset.py test_feature_windows.py test_audio_v5.py
```

`--min-field-recall` is optional and is a product policy choice, not a proven
default. Without it, combined validation F1 selects the threshold. Both ESC
and field validation/test precision, recall, F1 and confusion counts are
reported separately. Test labels are never used to select the threshold.

V5 trains one label per recording. The model scores each 3-second window and
uses the highest score for the file. It caps each recording at eight
evenly-spaced windows so long recordings do not automatically get more weight.
With only file labels, the exact abnormal event timestamp is unknown; an
event in an unsampled window can still be missed. Time annotations or
purposefully event-centered 3-second clips are needed to remove that limit.
`--field-weight` controls the relative field-sample training weight and must
be compared on independent field test recordings rather than assumed helpful.

The V5 feature interpolates the full 3-second Mel time axis to 128 columns
instead of discarding the last frames. Short waveform windows are silence
padded before feature extraction. This is a new input distribution and must
not be used with V3/V4 model weights.

## Deployment

Evaluation writes `models/audio_model_v5.config.json` with the selected
threshold, model path, feature version, sample rate and label mapping. Review
both held-out test results, then set `AUDIO_CONFIG_PATH` to its absolute path
for both `main.py` and `predict.py`. The default `audio_config.json` keeps
V3 and threshold 0.7. `GET /health` reports the active model, pipeline and
threshold. Do not deploy V5 based only on the ESC-50 result; test independent
3-second Raspberry Pi recordings as well.
