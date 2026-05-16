import os
import re
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score




# ayarlar

DATA_PATH = "C:/Users/yagiz/Desktop/Site/Dataset"
SAMPLE_RATE = 22050
DURATION = 3



# Normalize

def normalize_label(raw_label):
    raw_label = raw_label.lower()
    raw_label = re.sub(r'[^a-z]', '', raw_label)

    if raw_label in ["notr", "neutral", "ntr", "n"]:
        return "notr"

    if raw_label in ["mutlu", "happy"]:
        return "mutlu"

  
    if raw_label in ["ofkeli", "angry", "furious", "fkeli", "akn"]:
        return "ofkeli"

 
    if raw_label in ["uzgun", "sad", "mutsuz", "zgn"]:
        return "uzgun"

   
    if raw_label in ["saskin", "shocked", "surprised", "saskn"]:
        return "saskin"

    return raw_label


# Feature

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
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    target_length = SAMPLE_RATE * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_stats = stat_features(mfcc)

    delta = librosa.feature.delta(mfcc, mode="nearest")
    delta_stats = stat_features(delta)

    delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
    delta2_stats = stat_features(delta2)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_stats = stat_features(mel_db)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_stats = stat_features(contrast)

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

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_stats = np.array([np.mean(zcr), np.std(zcr)])

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

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    spectral_stats = np.array([
        np.mean(centroid), np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff), np.std(rolloff)
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
        spectral_stats,

    ])



# dataset yükleme


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

        
            group_id = parts[0] + "_" + parts[1]   # G03_D02

        
            label = normalize_label(parts[4])       # Uzgun vs

            features = extract_features(file_path)

            X.append(features)
            y.append(label)
            groups.append(group_id)

            print("Eklendi:", file_name, "->", label, "|", group_id)


        except Exception as e:

            print("Hata:", file_path)

            print("Sebep:", repr(e))


X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print("\nToplam veri:", len(X))
print("Toplam kişi (group):", len(set(groups)))



# Gruplama

gkf = GroupKFold(n_splits=5)

accuracies = []

fold = 1

for train_idx, test_idx in gkf.split(X, y, groups):

    print(f"\n--- Fold {fold} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

  
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    model = SVC(kernel="rbf", C=5,gamma=0.0007, class_weight="balanced")
    model.fit(X_train, y_train)

    
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



# Basit arayüz

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(12, 5))
plt.boxplot(X_scaled, showfliers=False)
plt.title("Ölçeklenmiş Feature Dağılımı")
plt.xlabel("Feature Index")
plt.ylabel("Standardized Value")
plt.show()
