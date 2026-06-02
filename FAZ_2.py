import os
import re
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

# =========================
# AYARLAR
# =========================
DATA_PATH = "C:/Users/yagiz/Desktop/Site/Dataset"
SAMPLE_RATE = 22050
DURATION = 3


# =========================
# LABEL CLEANING
# =========================
def normalize_label(raw_label):

    raw_label = raw_label.lower()
    raw_label = re.sub(r'[^a-z]', '', raw_label)

    # NÖTR
    if raw_label in ["notr", "neutral", "ntr", "n"]:
        return "notr"

    # MUTLU
    if raw_label in ["mutlu", "happy"]:
        return "mutlu"

    # ÖFKELİ
    if raw_label in ["ofkeli", "angry", "furious", "fkeli", "akn"]:
        return "ofkeli"

    # ÜZGÜN
    if raw_label in ["uzgun", "sad", "mutsuz", "zgn"]:
        return "uzgun"

    # ŞAŞKIN
    if raw_label in ["saskin", "shocked", "surprised", "saskn"]:
        return "saskin"

    return raw_label


# =========================
# FEATURE EXTRACTION
# =========================
def stat_features(feature):

    return np.hstack([
        np.mean(feature, axis=1),
        np.std(feature, axis=1),
        np.min(feature, axis=1),
        np.max(feature, axis=1),
        np.median(feature, axis=1),
        np.percentile(feature, 25, axis=1),
        np.percentile(feature, 75, axis=1)
    ])


def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    # Sabit uzunluk
    target_length = SAMPLE_RATE * DURATION

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_stats = stat_features(mfcc)

    # Delta
    delta = librosa.feature.delta(
        mfcc,
        mode="nearest"
    )

    delta_stats = stat_features(delta)

    # Delta-Delta
    delta2 = librosa.feature.delta(
        mfcc,
        order=2,
        mode="nearest"
    )

    delta2_stats = stat_features(delta2)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=40
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_stats = stat_features(mel_db)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr
    )

    contrast_stats = stat_features(contrast)

    # RMS
    rms = librosa.feature.rms(y=y)

    rms_stats = np.array([
        np.mean(rms),
        np.std(rms),
        np.min(rms),
        np.max(rms),
        np.median(rms),
        np.percentile(rms, 25),
        np.percentile(rms, 75)
    ])

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)

    zcr_stats = np.array([
        np.mean(zcr),
        np.std(zcr)
    ])

    # Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y)

    flatness_stats = np.array([
        np.mean(flatness),
        np.std(flatness),
        np.min(flatness),
        np.max(flatness),
        np.median(flatness),
        np.percentile(flatness, 25),
        np.percentile(flatness, 75)
    ])

    # Spectral Features
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    spectral_stats = np.array([
        np.mean(centroid),
        np.std(centroid),

        np.mean(bandwidth),
        np.std(bandwidth),

        np.mean(rolloff),
        np.std(rolloff)
    ])

    return np.hstack([
        mfcc_stats,
        delta_stats,
        delta2_stats,
        mel_stats,
        contrast_stats,
        flatness_stats,
        rms_stats,
        zcr_stats,
        spectral_stats
    ])


# =========================
# DATASET LOAD
# =========================
X = []
y = []
groups = []

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("DATA_PATH bulunamadı!")

for root, _, files in os.walk(DATA_PATH):

    for file in files:

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(root, file)

        try:

            file_name = os.path.basename(file)

            parts = file_name.split("_")

            # SPEAKER ID
            group_id = parts[0] + "_" + parts[1]

            # LABEL
            label = normalize_label(parts[4])

            # FEATURE
            features = extract_features(file_path)

            X.append(features)
            y.append(label)
            groups.append(group_id)

            print(
                "Eklendi:",
                file_name,
                "->",
                label,
                "|",
                group_id
            )

        except Exception as e:

            print("Hata:", file_path)
            print("Sebep:", repr(e))


X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print("\nToplam veri:", len(X))
print("Toplam kişi:", len(set(groups)))


# =========================
# GROUP K-FOLD
# =========================
gkf = GroupKFold(n_splits=5)

accuracies = []
macro_f1_scores = []

all_true = []
all_pred = []

fold = 1

for train_idx, test_idx in gkf.split(X, y, groups):

    print(f"\n--- Fold {fold} ---")

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # SCALE
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # MODEL
    model = SVC(
        kernel="rbf",
        C=5,
        gamma=0.0007,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # PREDICT
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # METRICS
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro"
    )

    accuracies.append(test_acc)
    macro_f1_scores.append(macro_f1)

    all_true.extend(y_test)
    all_pred.extend(test_pred)

    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")

    fold += 1


# =========================
# FINAL RESULTS
# =========================
print("\n========================")
print(f"ORTALAMA ACCURACY: {np.mean(accuracies) * 100:.2f}%")
print(f"ORTALAMA MACRO F1: {np.mean(macro_f1_scores):.4f}")
print("========================")


# =========================
# CLASSIFICATION REPORT
# =========================
print("\nCLASSIFICATION REPORT")
print("========================")

report = classification_report(
    all_true,
    all_pred
)

print(report)


# =========================
# CONFUSION MATRIX
# =========================
labels = sorted(list(set(y)))

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=labels
)

print("\nCONFUSION MATRIX")
print("========================")
print(cm)


# =========================
# CONFUSION MATRIX VISUAL
# =========================
plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation='nearest')

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(labels))

plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

for i in range(len(labels)):
    for j in range(len(labels)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.ylabel("Gerçek Etiket")
plt.xlabel("Tahmin Edilen Etiket")

plt.tight_layout()

plt.show()


# =========================
# BASIC VISUALS
# =========================

# Ortalama MFCC
mfcc_all = np.array([f[:13] for f in X])

mfcc_mean = np.mean(mfcc_all, axis=0)

plt.figure()

plt.plot(mfcc_mean)

plt.title("Ortalama MFCC")

plt.xlabel("MFCC Index")
plt.ylabel("Ortalama Değer")

plt.show()


# ZCR Histogram
zcr_values = X[:, -1]

plt.figure()

plt.hist(zcr_values, bins=20)

plt.title("ZCR Dağılımı")

plt.xlabel("ZCR")
plt.ylabel("Frekans")

plt.show()


# Feature Distribution
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(12, 5))

plt.boxplot(
    X_scaled,
    showfliers=False
)

plt.title("Ölçeklenmiş Feature Dağılımı")

plt.xlabel("Feature Index")
plt.ylabel("Standardized Value")

plt.show()

final_scaler = StandardScaler()
X_final_scaled = final_scaler.fit_transform(X)

final_model = SVC(
    kernel="rbf",
    C=5,
    gamma=0.0007,
    class_weight="balanced"
)

final_model.fit(X_final_scaled, y)

joblib.dump(final_model, "emotion_model.pkl")
joblib.dump(final_scaler, "scaler.pkl")

print("Final model kaydedildi: emotion_model.pkl")
print("Scaler kaydedildi: scaler.pkl")
