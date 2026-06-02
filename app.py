import streamlit as st
import joblib
import numpy as np
import librosa
import tempfile
import matplotlib.pyplot as plt
import librosa.display
from streamlit_mic_recorder import mic_recorder

# =========================
# SAYFA AYARLARI
# =========================
st.set_page_config(
    page_title="İşaretler ve Sistemler - Duygu Analizi",
    page_icon="🎧",
    layout="wide"
)

# =========================
# TASARIM
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 24px;
    color: #cbd5e1;
    margin-bottom: 35px;
}

.card {
    background-color: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0px 4px 25px rgba(0,0,0,0.35);
    margin-top: 25px;
}

.result-card h1 {
    font-size: 56px;
    color: white;
}

.result-card h2 {
    color: #e0f2fe;
}

.info-text {
    color: #e2e8f0;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# AYARLAR
# =========================
SAMPLE_RATE = 22050
DURATION = 3

emotion_emojis = {
    "mutlu": "😊",
    "uzgun": "😢",
    "ofkeli": "😠",
    "notr": "😐",
    "saskin": "😲"
}

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
# GRAFİK FONKSİYONLARI
# =========================
def plot_waveform(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Ses Dalga Formu")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Genlik")
    return fig


def plot_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )
    ax.set_title("Mel-Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig


def plot_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(
        mfcc,
        sr=sr,
        x_axis="time",
        ax=ax
    )
    ax.set_title("MFCC")
    fig.colorbar(img, ax=ax)
    return fig


def get_audio_summary(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    return {
        "RMS Enerji": np.mean(rms),
        "ZCR": np.mean(zcr),
        "Spectral Centroid": np.mean(centroid)
    }


# =========================
# MODEL YÜKLEME
# =========================
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# BAŞLIK
# =========================
st.markdown("""
<div class="main-title">🎧 İşaretler ve Sistemler</div>
<div class="sub-title">Ses Tabanlı Duygu Analizi Sistemi | Emo-Challenge 2026</div>
""", unsafe_allow_html=True)

# =========================
# ANA SAYFA
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    <div class="card">
        <h3>📌 Proje Bilgisi</h3>
        <p class="info-text">
        Bu sistem, yüklenen ses dosyasından zaman ve frekans düzlemi öznitelikleri çıkararak konuşmacının duygusunu tahmin eder.
        </p>
        <hr>
        <p><b>Model:</b> SVM</p>
        <p><b>Kernel:</b> RBF</p>
        <p><b>C:</b> 5</p>
        <p><b>Gamma:</b> 0.0007</p>
        <p><b>Başarı:</b> %84–85</p>
        <p><b>Ses Süresi:</b> 3 saniye</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>📁 Ses Dosyası Yükleme</h3>
        <p class="info-text">Lütfen WAV formatında bir ses dosyası yükleyin.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Bir WAV ses dosyası seçin",
        type=["wav"]
    )
    st.markdown("### 🎙️ Mikrofon ile Kayıt")

    audio = mic_recorder(
        start_prompt="Kaydı Başlat",
        stop_prompt="Kaydı Durdur",
        just_once=False,
        use_container_width=True
    )

audio_source = None
source_type = None

if uploaded_file is not None:
    audio_source = uploaded_file
    source_type = "upload"

elif audio is not None:
    audio_source = audio["bytes"]
    source_type = "mic"

if audio_source is not None:
    st.markdown("## 🔊 Yüklenen Ses")
    st.markdown("## 🔊 Alınan Ses")

    if source_type == "upload":
        st.audio(audio_source, format="audio/wav")
        audio_bytes = audio_source.read()
    else:
        st.audio(audio_source, format="audio/wav")
        audio_bytes = audio_source

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    with st.spinner("Ses dosyası analiz ediliyor..."):
        features = extract_features(tmp_path)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

    emoji = emotion_emojis.get(prediction, "🎭")

    st.markdown(f"""
    <div class="result-card">
        <h2>Tahmin Edilen Duygu</h2>
        <h1>{emoji} {prediction.upper()}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📊 Ses Özeti")

    audio_summary = get_audio_summary(tmp_path)

    m1, m2, m3 = st.columns(3)

    m1.metric("RMS Enerji", f"{audio_summary['RMS Enerji']:.4f}")
    m2.metric("ZCR", f"{audio_summary['ZCR']:.4f}")
    m3.metric("Spectral Centroid", f"{audio_summary['Spectral Centroid']:.2f}")

    st.markdown("## 📈 Ses Analiz Grafikleri")

    tab1, tab2, tab3 = st.tabs([
        "Dalga Formu",
        "Mel-Spectrogram",
        "MFCC"
    ])

    with tab1:
        st.pyplot(plot_waveform(tmp_path))

    with tab2:
        st.pyplot(plot_mel_spectrogram(tmp_path))

    with tab3:
        st.pyplot(plot_mfcc(tmp_path))

else:
    st.markdown("""
    <div class="card">
        <h3>🚀 Kullanım</h3>
        <p class="info-text">
        Bir WAV dosyası yüklediğinizde sistem otomatik olarak ses özniteliklerini çıkarır,
        kaydedilmiş SVM modeli ile duygu tahmini yapar ve analiz grafiklerini gösterir.
        </p>
    </div>
    """, unsafe_allow_html=True)