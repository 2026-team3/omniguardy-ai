from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from sklearn.model_selection import train_test_split

from feature import audio_to_mel


# =====================================
# 경로 설정
# =====================================

DATASET_DIR = Path("dataset")

# 항상 최초 ESC-50 모델에서 시작
OLD_MODEL = "models/audio_model.keras"

# 이번 실험 결과
NEW_MODEL = "models/audio_model_v3_weight_1_1.keras"


# =====================================
# 설정
# =====================================

RANDOM_STATE = 42

TEST_SIZE = 0.2

BATCH_SIZE = 8

EPOCHS = 30


# =====================================
# Class Weight
# =====================================

# Normal : Abnormal = 1 : 1

CLASS_WEIGHTS = {
    0: 1.0,
    1: 1.0
}


# =====================================
# Random Seed
# =====================================

np.random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)


# =====================================
# 데이터 증강
# =====================================

def augment_audio(audio, sr):

    augmented = []

    # -------------------------
    # 1. 원본
    # -------------------------

    augmented.append(
        audio
    )


    # -------------------------
    # 2. Noise
    # -------------------------

    noise = np.random.normal(
        0,
        0.003,
        len(audio)
    )

    augmented.append(
        audio + noise
    )


    # -------------------------
    # 3. Volume 0.8
    # -------------------------

    augmented.append(
        audio * 0.8
    )


    # -------------------------
    # 4. Volume 1.2
    # -------------------------

    augmented.append(
        audio * 1.2
    )


    # -------------------------
    # 5. Time Shift
    # -------------------------

    shift = np.random.randint(
        -sr // 2,
        sr // 2
    )

    augmented.append(
        np.roll(
            audio,
            shift
        )
    )


    return augmented


# =====================================
# WAV 파일 목록 가져오기
# =====================================

def get_file_list():

    files = []
    labels = []

    classes = {
        "normal": 0,
        "abnormal": 1
    }


    print("==============================")
    print("Original Dataset")
    print("==============================")


    for folder, label in classes.items():

        folder_path = (
            DATASET_DIR / folder
        )

        wav_files = sorted(
            folder_path.glob("*.wav")
        )


        print(
            f"{folder}: {len(wav_files)} files"
        )


        for file in wav_files:

            files.append(
                str(file)
            )

            labels.append(
                label
            )


    return (
        np.array(files),
        np.array(
            labels,
            dtype=np.int32
        )
    )


# =====================================
# Audio Load
# =====================================

def load_audio(file_path):

    audio, sr = sf.read(
        file_path
    )


    # Stereo -> Mono

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )


    # float32 통일

    audio = audio.astype(
        np.float32
    )


    return audio, sr


# =====================================
# Training Dataset 생성
#
# Train에만 Augmentation 적용
# =====================================

def build_train_dataset(
    files,
    labels
):

    X = []
    y = []


    print("\n")
    print("==============================")
    print("Building Train Dataset")
    print("==============================")


    for file_path, label in zip(
        files,
        labels
    ):

        audio, sr = load_audio(
            file_path
        )


        augmented_audio_list = (
            augment_audio(
                audio,
                sr
            )
        )


        for aug_audio in (
            augmented_audio_list
        ):

            mel = audio_to_mel(
                aug_audio,
                sr
            )


            X.append(
                mel
            )

            y.append(
                label
            )


    X = np.array(
        X,
        dtype=np.float32
    )


    X = X[
        ...,
        np.newaxis
    ]


    y = np.array(
        y,
        dtype=np.int32
    )


    return X, y


# =====================================
# Validation Dataset 생성
#
# Validation에는 Augmentation X
# 원본만 사용
# =====================================

def build_validation_dataset(
    files,
    labels
):

    X = []
    y = []


    print("\n")
    print("==============================")
    print("Building Validation Dataset")
    print("==============================")


    for file_path, label in zip(
        files,
        labels
    ):

        audio, sr = load_audio(
            file_path
        )


        mel = audio_to_mel(
            audio,
            sr
        )


        X.append(
            mel
        )

        y.append(
            label
        )


    X = np.array(
        X,
        dtype=np.float32
    )


    X = X[
        ...,
        np.newaxis
    ]


    y = np.array(
        y,
        dtype=np.int32
    )


    return X, y


# =====================================
# Main
# =====================================

print("==============================")
print("Fine-tuning V3")
print("Class Weight = 1 : 1")
print("==============================")


# =====================================
# 1. 원본 파일 목록
# =====================================

files, labels = (
    get_file_list()
)


print(
    "\nTotal original files:",
    len(files)
)

print(
    "Normal original:",
    np.sum(
        labels == 0
    )
)

print(
    "Abnormal original:",
    np.sum(
        labels == 1
    )
)


# =====================================
# 2. 원본 파일 기준
# Train / Validation Split
# =====================================

train_files, val_files, train_labels, val_labels = (
    train_test_split(
        files,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
)


print("\n")
print("==============================")
print("Original File Split")
print("==============================")


print(
    "Train files:",
    len(train_files)
)

print(
    "Validation files:",
    len(val_files)
)


print(
    "Train Normal:",
    np.sum(
        train_labels == 0
    )
)

print(
    "Train Abnormal:",
    np.sum(
        train_labels == 1
    )
)


print(
    "Validation Normal:",
    np.sum(
        val_labels == 0
    )
)

print(
    "Validation Abnormal:",
    np.sum(
        val_labels == 1
    )
)


# =====================================
# 3. Dataset 생성
# =====================================

X_train, y_train = (
    build_train_dataset(
        train_files,
        train_labels
    )
)


X_val, y_val = (
    build_validation_dataset(
        val_files,
        val_labels
    )
)


print("\n")
print("==============================")
print("Final Dataset")
print("==============================")


print(
    "X_train:",
    X_train.shape
)

print(
    "Train Normal:",
    np.sum(
        y_train == 0
    )
)

print(
    "Train Abnormal:",
    np.sum(
        y_train == 1
    )
)


print(
    "\nX_val:",
    X_val.shape
)

print(
    "Validation Normal:",
    np.sum(
        y_val == 0
    )
)

print(
    "Validation Abnormal:",
    np.sum(
        y_val == 1
    )
)


# =====================================
# 4. 기존 모델 Load
# =====================================

print("\n")
print("==============================")
print("Loading Base Model")
print("==============================")


model = tf.keras.models.load_model(
    OLD_MODEL,
    compile=False
)


print(
    "Loaded:",
    OLD_MODEL
)


model.summary()


# =====================================
# 5. Compile
# =====================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )

    ]
)


# =====================================
# 6. Callback
# =====================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1

    ),


    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        verbose=1

    )

]


# =====================================
# 7. Training
# =====================================

print("\n")
print("==============================")
print("Training")
print("==============================")


print(
    "Class weights:",
    CLASS_WEIGHTS
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=CLASS_WEIGHTS,

    callbacks=callbacks,

    verbose=1

)


# =====================================
# 8. Save
# =====================================

model.save(
    NEW_MODEL
)


print("\n")
print("==============================")
print("Fine-tuning Finished")
print("==============================")


print(
    "Saved:",
    NEW_MODEL
)