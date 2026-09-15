# Audio v4 experiment

This experiment fixes evaluation leakage and train/evaluation preprocessing
inconsistency. It does not claim that F1 or recall improves without remote
training and held-out testing.

1. Download ESC-50 on the remote training machine. Keep its `meta/esc50.csv`
   and `audio/` directories together.
2. Run `python train_v4.py --esc50-dir /path/to/ESC-50`. Folds 1-3 train the
   model, fold 4 monitors training, and fold 5 remains untouched.
3. Run `python evaluate_v4.py --esc50-dir /path/to/ESC-50`. The threshold is
   selected on fold 4 and reported once on fold 5. Record F1, recall, FN, and
   FP alongside the old system's results before deployment.
   Run `python -m unittest test_feature_windows.py` for the windowing checks.
4. Deploy the new model only after validation: set `AUDIO_MODEL_PATH` to the
   v4 model, `AUDIO_PIPELINE_VERSION=v4`, and `AUDIO_THRESHOLD` to the selected
   threshold. The API defaults to the existing v3 model, v3 preprocessing, and
   0.7 threshold until explicitly changed.

The fold-5 result is an ESC-50 held-out test, not a field-audio test. Collect
independent 3-second Raspberry Pi recordings and compare both models on those
clips too. In particular, inspect siren, door creak, and footsteps false
negatives and normal-noise false positives. If field-audio recall is the safety
priority, choose a threshold on a separate field-audio validation set under an
explicit maximum false-positive or minimum-precision constraint; do not tune
it on the final test set.

`finetune.py` and `finetune_v3.py` are historical experiments and are not used
by v4. They should not be evaluated on their own training recordings.
