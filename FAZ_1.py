import os
import re
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# =========================
# AYARLAR
# =========================
DATA_PATH = "/Users/macbook/Downloads/Midterm_Dataset_2026"
SAMPLE_RATE = 22050
DURATION = 3


# =========================
# LABEL CLEANING
# =========================
def normalize_label(raw_label):
    raw_label = raw_label.lower()
    raw_label = re.sub(r'[^a-z]', '', raw_label)
    return raw_label


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta.T, axis=0)

    delta2 = librosa.feature.delta(mfcc, order=2)
    delta2_mean = np.mean(delta2.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    return np.hstack([
        mfcc_mean,
        delta_mean,
        delta2_mean,
        chroma_mean,
        np.mean(centroid),
        np.mean(bandwidth),
        np.mean(rolloff),
        rms_mean,
        zcr_mean
    ])


# =========================
# DATASET LOAD
# =========================
X, y, groups = [], [], []

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

            # 🔥 GERÇEK KİŞİ TANIMI
            group_id = parts[0] + "_" + parts[1]   # G03_D02

            # 🔥 LABEL
            label = normalize_label(parts[4])       # Uzgun vs

            features = extract_features(file_path)

            X.append(features)
            y.append(label)
            groups.append(group_id)

            print("Eklendi:", file_name, "->", label, "|", group_id)

        except Exception as e:
            print("Hata:", file_path, e)


X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print("\nToplam veri:", len(X))
print("Toplam kişi (group):", len(set(groups)))


# =========================
# GROUP K-FOLD
# =========================
gkf = GroupKFold(n_splits=5)

accuracies = []

fold = 1

for train_idx, test_idx in gkf.split(X, y, groups):

    print(f"\n--- Fold {fold} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # SCALE
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # MODEL (SVM)
    model = SVC(kernel="rbf", C=10)
    model.fit(X_train, y_train)

    # PREDICT
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    accuracies.append(test_acc)

    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    fold += 1


print("\n========================")
print(f"ORTALAMA ACCURACY: {np.mean(accuracies) * 100:.2f}%")
print("========================")


# =========================
# BASIC VISUALS
# =========================
mfcc_all = np.array([f[:13] for f in X])
mfcc_mean = np.mean(mfcc_all, axis=0)

plt.figure()
plt.plot(mfcc_mean)
plt.title("Ortalama MFCC")
plt.show()

zcr_values = X[:, -1]

plt.figure()
plt.hist(zcr_values, bins=20)
plt.title("ZCR Dağılımı")
plt.show()

plt.figure()
plt.boxplot(X)
plt.title("Feature Dağılımı")
plt.show()
