import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# AYARLAR

DATA_PATH = "/Users/macbook/Downloads/dataset"
SAMPLE_RATE = 22050
DURATION = 3


# FEATURE EXTRACTION

def extract_features_from_signal(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    return np.hstack([mfcc_mean, rms_mean, zcr_mean])


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=DURATION)
    return extract_features_from_signal(y, sr)


# DATASET HAZIRLAMA

X, y = [], []

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("DATA_PATH bulunamadı!")

for file in os.listdir(DATA_PATH):

    if not file.lower().endswith(".wav"):
        continue

    file_path = os.path.join(DATA_PATH, file)

    try:
        # label dosya isminden
        label = file.split("_")[0]

        features = extract_features(file_path)

        X.append(features)
        y.append(label)

    except Exception as e:
        print(f"Hata: {file} -> {e}")

print("Toplam veri:", len(X))

if len(X) == 0:
    raise ValueError("Hiç veri yüklenmedi!")

X = np.array(X)
y = np.array(y)


# FEATURE SCALING

scaler = StandardScaler()
X = scaler.fit_transform(X)



# MODEL

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))


# GRAFİKLER

mfcc_all = np.array([f[:13] for f in X])
mfcc_mean_global = np.mean(mfcc_all, axis=0)

plt.figure()
plt.plot(mfcc_mean_global)
plt.title("Ortalama MFCC")
plt.savefig("mfcc_mean.png")
plt.show()

zcr_values = X[:, -1]

plt.figure()
plt.hist(zcr_values, bins=20)
plt.title("ZCR Dağılımı")
plt.savefig("zcr_hist.png")
plt.show()

plt.figure()
plt.boxplot(X)
plt.title("Feature Boxplot")
plt.savefig("boxplot.png")
plt.show()