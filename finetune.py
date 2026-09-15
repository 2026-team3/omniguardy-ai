import os
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from feature import audio_to_mel


# =====================================
# 경로 설정
# =====================================

DATASET_DIR = Path("dataset")

OLD_MODEL = "models/audio_model.keras"

NEW_MODEL = "models/audio_model_v2.keras"


# =====================================
# 데이터 증강
# =====================================

def augment_audio(audio, sr):

    augmented = []

    # 원본
    augmented.append(audio)


    # -------------------------
    # 잡음 추가
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
    # 음량 변화
    # -------------------------

    augmented.append(
        audio * 0.8
    )

    augmented.append(
        audio * 1.2
    )


    # -------------------------
    # 시간 이동
    # -------------------------

    shift = np.random.randint(
        -sr // 2,
        sr // 2
    )

    augmented.append(
        np.roll(audio, shift)
    )


    return augmented



# =====================================
# 데이터셋 로드
# =====================================

def load_dataset():

    X = []
    y = []


    classes = {
        "normal": 0,
        "abnormal": 1
    }


    for folder, label in classes.items():

        folder_path = DATASET_DIR / folder


        files = list(
            folder_path.glob("*.wav")
        )


        print(
            folder,
            "files:",
            len(files)
        )


        for file in files:


            audio, sr = sf.read(
                file
            )


            # 스테레오를 모노로 변환

            if audio.ndim > 1:

                audio = np.mean(
                    audio,
                    axis=1
                )


            # 증강 적용

            for aug_audio in augment_audio(
                audio,
                sr
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


    # CNN 입력 형태

    X = X[..., np.newaxis]


    y = np.array(
        y,
        dtype=np.int32
    )


    return X, y



# =====================================
# 실행부
# =====================================


print("==============================")
print("Loading dataset")
print("==============================")


X, y = load_dataset()


print(
    "X shape:",
    X.shape
)

print(
    "normal:",
    np.sum(y == 0)
)

print(
    "abnormal:",
    np.sum(y == 1)
)



# =====================================
# 학습·검증 데이터 분할
# =====================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# =====================================
# 클래스 가중치
# =====================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)


class_weights = {
    i: w
    for i, w in enumerate(weights)
}


print(
    "class weights:",
    class_weights
)



# =====================================
# 기존 모델 로드
# =====================================

print(
    "Loading model..."
)


model = tf.keras.models.load_model(
    OLD_MODEL,
    compile=False
)


model.summary()



# =====================================
# 추가 학습용 모델 컴파일
# =====================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)



# =====================================
# 콜백
# =====================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True

    ),


    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3

    )

]



# =====================================
# 학습
# =====================================

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_val, y_val),

    epochs=30,

    batch_size=8,

    class_weight=class_weights,

    callbacks=callbacks

)



# =====================================
# 저장
# =====================================

model.save(
    NEW_MODEL
)


print("==============================")
print("Fine tuning finished")
print("Saved:", NEW_MODEL)
print("==============================")
