# Audio v4 experiment

This experiment fixes evaluation leakage and train/evaluation preprocessing
inconsistency. It does not claim that F1 or recall improves without remote
training and held-out testing.

1. Download ESC-50 on the remote training machine. Keep its `meta/esc50.csv`
   and `audio/` directories together.
2. Place field WAV files under `dataset/normal` (correct door-lock sounds) and
   `dataset/abnormal` (incorrect door-lock sounds). Run
   `python train_v4.py --esc50-dir /path/to/ESC-50 --field-data-dir dataset`.
   Folds 1-3 and field training files train the model. Fold 4 and separate
   original field files monitor training. Fold 5 remains untouched.
3. Run `python evaluate_v4.py --esc50-dir /path/to/ESC-50 --field-data-dir dataset`.
   The threshold is selected on fold 4 plus field validation files and reported
   once on fold 5. Record F1, recall, FN, and
   FP alongside the old system's results before deployment.
   Run `python -m unittest test_feature_windows.py test_field_dataset.py` for
   the windowing and directory-label checks.
4. V4 is retained as a historical window-label experiment. Use V5 for new
   training and evaluation. The API and CLI now read `audio_config.json` by
   default, preserving the existing V3 model and 0.7 threshold. Set
   `AUDIO_CONFIG_PATH` to a validated model manifest to deploy a new model.

The field validation has only a few original recordings and must not be treated
as a reliable field test. Verify that different takes from the same recording
session are kept together; the current split is by file, not session. The
fold-5 result is an ESC-50 held-out test, not a field-audio test. Collect
independent 3-second Raspberry Pi recordings and compare both models on those
clips too. In particular, inspect siren, door creak, and footsteps false
negatives and normal-noise false positives. If field-audio recall is the safety
priority, choose a threshold on a separate field-audio validation set under an
explicit maximum false-positive or minimum-precision constraint; do not tune
it on the final test set.

`finetune.py` and `finetune_v3.py` are historical experiments and are not used
by v4. They should not be evaluated on their own training recordings.
