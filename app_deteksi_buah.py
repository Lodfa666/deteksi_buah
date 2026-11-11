import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import io

# ==================================================
# KONFIGURASI APLIKASI
# ==================================================

st.set_page_config(
    page_title="FruitScan - Deteksi Buah",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "best.pt"

KELAS_BUAH = {
    "good apple": ("Apel", "Segar"),
    "bad apple": ("Apel", "Busuk"),
    "good banana": ("Pisang", "Segar"),
    "bad banana": ("Pisang", "Busuk"),
    "good orange": ("Jeruk", "Segar"),
    "bad orange": ("Jeruk", "Busuk"),
    "good pomegranate": ("Delima", "Segar"),
    "bad pomegranate": ("Delima", "Busuk")
}

# ==================================================
# CSS / TAMPILAN
# ==================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F8F9F6;
    }

    [data-testid="stSidebar"] {
        background-color: #263D32;
    }

    [data-testid="stSidebar"] * {
        color: #F5F7F3 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 8px 10px;
        border-radius: 7px;
    }

    h1, h2, h3 {
        color: #29332D;
    }

    .page-subtitle {
        color: #78827B;
        font-size: 14px;
        margin-top: -12px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E8ECE6;
        border-radius: 10px;
        padding: 20px;
        min-height: 115px;
    }

    .metric-label {
        color: #78827B;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #263D32;
        font-size: 28px;
        font-weight: 700;
    }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #29332D;
        margin-bottom: 12px;
    }

    .info-box {
        background: #FFFFFF;
        border: 1px solid #E8ECE6;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }

    div.stButton > button {
        background-color: #527D62;
        color: white;
        border: none;
        border-radius: 7px;
        padding: 9px 20px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #3D654D;
        color: white;
        border: none;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #B8C8B9;
        border-radius: 10px;
        padding: 12px;
    }

    hr {
        border-color: #E5EAE3;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# MODEL YOLO
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return YOLO(str(MODEL_PATH))


# ==================================================
# SESSION STATE
# ==================================================

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

if "hasil_deteksi" not in st.session_state:
    st.session_state.hasil_deteksi = None

if "nama_file" not in st.session_state:
    st.session_state.nama_file = None


# ==================================================
# FUNGSI GAMBAR
# ==================================================

def buka_gambar(file):
    gambar = Image.open(file)
    gambar = ImageOps.exif_transpose(gambar)

    if getattr(gambar, "is_animated", False):
        gambar.seek(0)

    return gambar.convert("RGB")


def gambar_font():
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


# ==================================================
# FUNGSI DETEKSI
# ==================================================

def deteksi_buah(gambar, confidence):
    model = load_model()

    if model is None:
        raise FileNotFoundError(
            "File best.pt tidak ditemukan. Letakkan best.pt satu folder dengan app_deteksi_buah.py."
        )

    hasil = model.predict(
        source=gambar,
        conf=confidence,
        iou=0.4,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    result = hasil[0]
    gambar_hasil = gambar.copy()
    draw = ImageDraw.Draw(gambar_hasil)
    font = gambar_font()

    data_deteksi = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            nilai_conf = float(box.conf[0].item())

            nama_kelas = result.names[class_id]

            if nama_kelas not in KELAS_BUAH:
                continue

            nama_buah, kondisi = KELAS_BUAH[nama_kelas]

            koordinat = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, koordinat)

            warna = "#3D9B62" if kondisi == "Segar" else "#D65B50"

            teks = f"{nama_buah} - {kondisi} ({nilai_conf * 100:.1f}%)"

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=warna,
                width=4
            )

            bbox_teks = draw.textbbox((x1, max(0, y1 - 25)), teks, font=font)

            draw.rectangle(
                bbox_teks,
                fill=warna
            )

            draw.text(
                (x1, max(0, y1 - 25)),
                teks,
                fill="white",
                font=font
            )

            data_deteksi.append({
                "Nama Buah": nama_buah,
                "Kondisi": kondisi,
                "Confidence": round(nilai_conf * 100, 2)
            })

    return gambar_hasil, data_deteksi


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown(
        "<h2 style='color:#FFFFFF;margin-bottom:0;'>🍃 FruitScan</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#C6D5C9;font-size:12px;'>Sistem Deteksi Kesegaran Buah</p>",
        unsafe_allow_html=True
    )

    st.divider()

    halaman = st.radio(
        "MENU UTAMA",
        [
            "Dashboard",
            "Deteksi Gambar",
            "Riwayat Deteksi",
            "Tentang Aplikasi"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.caption("FruitScan • YOLOv8n")


# ==================================================
# DASHBOARD
# ==================================================

if halaman == "Dashboard":

    st.title("Dashboard")
    st.markdown(
        '<p class="page-subtitle">Ringkasan aktivitas deteksi kesegaran buah.</p>',
        unsafe_allow_html=True
    )

    riwayat = st.session_state.riwayat

    total = len(riwayat)
    segar = sum(1 for x in riwayat if x["Kondisi"] == "Segar")
    busuk = sum(1 for x in riwayat if x["Kondisi"] == "Busuk")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Deteksi</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Buah Segar</div>
            <div class="metric-value">{segar}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Buah Busuk</div>
            <div class="metric-value">{busuk}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="section-title">Informasi Aplikasi</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>FruitScan</b><br><br>
        Aplikasi berbasis web untuk mendeteksi jenis dan kondisi kesegaran buah
        menggunakan teknologi computer vision dengan model YOLOv8n.<br><br>
        <b>Jenis buah yang didukung:</b><br>
        Apel, Pisang, Jeruk, dan Delima.
    </div>
    """, unsafe_allow_html=True)

    if riwayat:
        st.markdown('<div class="section-title">Deteksi Terakhir</div>',
                    unsafe_allow_html=True)

        df = pd.DataFrame(riwayat)
        st.dataframe(
            df.tail(5).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# DETEKSI GAMBAR
# ==================================================

elif halaman == "Deteksi Gambar":

    st.title("Deteksi Gambar")
    st.markdown(
        '<p class="page-subtitle">Unggah gambar buah untuk mengetahui kondisi kesegarannya.</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        file_gambar = st.file_uploader(
            "Pilih gambar buah",
            type=[
                "jpg", "jpeg", "jfif", "png",
                "webp", "bmp", "gif", "tif", "tiff"
            ],
            help="Format yang didukung: JPG, JPEG, JFIF, PNG, WEBP, BMP, GIF, TIF, TIFF."
        )

    with col2:
        confidence = st.slider(
            "Batas Confidence",
            min_value=0.10,
            max_value=1.00,
            value=0.50,
            step=0.05
        )

    if file_gambar is not None:

        try:
            gambar = buka_gambar(file_gambar)

            st.markdown("#### Pratinjau Gambar")

            _, preview_col, _ = st.columns([1, 2, 1])

            with preview_col:
                st.image(
                    gambar,
                    caption=file_gambar.name,
                    width=450
                )
            if st.button("Mulai Deteksi", use_container_width=True):

                with st.spinner("Sedang menganalisis gambar..."):
                    gambar_hasil, data = deteksi_buah(
                        gambar,
                        confidence
                    )

                st.session_state.hasil_deteksi = {
                    "gambar": gambar_hasil,
                    "data": data
                }

                st.session_state.nama_file = file_gambar.name

                waktu = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                for item in data:
                    st.session_state.riwayat.append({
                        "Waktu": waktu,
                        "Nama File": file_gambar.name,
                        "Nama Buah": item["Nama Buah"],
                        "Kondisi": item["Kondisi"],
                        "Confidence (%)": item["Confidence"]
                    })

            hasil = st.session_state.hasil_deteksi

            if hasil is not None and st.session_state.nama_file == file_gambar.name:

                st.divider()
                st.markdown("### Hasil Deteksi")

                col_asli, col_hasil = st.columns(2)

                with col_asli:
                    st.markdown("**Gambar Asli**")
                    st.image(gambar, use_container_width=True)

                with col_hasil:
                    st.markdown("**Hasil Analisis**")
                    st.image(hasil["gambar"], use_container_width=True)

                if hasil["data"]:

                    st.success(
                        f"Berhasil mendeteksi {len(hasil['data'])} objek buah."
                    )

                    df_hasil = pd.DataFrame(hasil["data"])

                    st.markdown("#### Detail Hasil")

                    st.dataframe(
                        df_hasil,
                        use_container_width=True,
                        hide_index=True
                    )

                    jumlah_segar = sum(
                        1 for x in hasil["data"]
                        if x["Kondisi"] == "Segar"
                    )

                    jumlah_busuk = sum(
                        1 for x in hasil["data"]
                        if x["Kondisi"] == "Busuk"
                    )

                    c1, c2 = st.columns(2)

                    c1.metric("Buah Segar", jumlah_segar)
                    c2.metric("Buah Busuk", jumlah_busuk)

                else:
                    st.warning(
                        "Tidak ditemukan buah yang sesuai dengan kelas model."
                    )

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")


# ==================================================
# RIWAYAT DETEKSI
# ==================================================

elif halaman == "Riwayat Deteksi":

    st.title("Riwayat Deteksi")
    st.markdown(
        '<p class="page-subtitle">Daftar hasil deteksi yang dilakukan selama sesi aplikasi.</p>',
        unsafe_allow_html=True
    )

    riwayat = st.session_state.riwayat

    if riwayat:

        df = pd.DataFrame(riwayat)

        st.dataframe(
            df.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Download Riwayat CSV",
            data=csv,
            file_name="riwayat_deteksi_buah.csv",
            mime="text/csv"
        )

        if st.button("Hapus Riwayat"):
            st.session_state.riwayat = []
            st.session_state.hasil_deteksi = None
            st.session_state.nama_file = None
            st.rerun()

    else:
        st.info("Belum ada riwayat deteksi.")


# ==================================================
# TENTANG APLIKASI
# ==================================================

elif halaman == "Tentang Aplikasi":

    st.title("Tentang Aplikasi")
    st.markdown(
        '<p class="page-subtitle">Informasi mengenai sistem FruitScan.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
        <h3>🍃 FruitScan</h3>
        FruitScan merupakan aplikasi deteksi kesegaran buah berbasis web
        yang menggunakan metode computer vision dan model YOLOv8n.
        Sistem ini dirancang untuk mengenali buah dan mengklasifikasikan
        kondisinya menjadi segar atau busuk.
        <hr>
        <b>Teknologi yang digunakan</b><br>
        • Python<br>
        • Streamlit<br>
        • YOLOv8n<br>
        • Pillow<br>
        • Pandas
        <hr>
        <b>Kelas deteksi</b><br>
        • Apel Segar dan Busuk<br>
        • Pisang Segar dan Busuk<br>
        • Jeruk Segar dan Busuk<br>
        • Delima Segar dan Busuk
    </div>
    """, unsafe_allow_html=True)