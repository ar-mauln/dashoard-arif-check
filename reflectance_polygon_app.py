# reflectance_polygon_app.py
# Aplikasi sederhana: upload CSV reflectance, input 4 koordinat polygon,
# filter titik di dalam polygon, plot, lalu download CSV hasil.
# Jalankan: streamlit run reflectance_polygon_app.py

import io
from typing import List, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.path import Path


st.set_page_config(page_title="Reflectance Polygon Acquire", layout="wide")
st.title("Reflectance Acquire by Polygon")
st.caption("Upload CSV reflectance → input 4 koordinat polygon → proses → download CSV hasil.")


def detect_column(columns: List[str], candidates: List[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_number(value):
    """Terima input angka Indonesia/English: -6,123 atau -6.123."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "")
    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    return float(text)


def point_in_polygon(df: pd.DataFrame, lon_col: str, lat_col: str, polygon: List[Tuple[float, float]]) -> pd.Series:
    # Path memakai urutan (longitude, latitude)
    path = Path(polygon)
    points = df[[lon_col, lat_col]].astype(float).to_numpy()
    return path.contains_points(points, radius=1e-12)


def csv_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


with st.sidebar:
    st.header("1. Input CSV")
    uploaded_file = st.file_uploader("Upload file CSV reflectance", type=["csv"])
    st.info("Format yang didukung minimal punya kolom latitude dan longitude. Kolom reflectance seperti B2_blue, B3_green, B4_red, B8_nir akan ikut terbawa ke output.")

if uploaded_file is None:
    st.warning("Upload CSV dulu untuk mulai proses.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"CSV gagal dibaca: {exc}")
    st.stop()

if df.empty:
    st.error("CSV kosong.")
    st.stop()

lat_guess = detect_column(df.columns.tolist(), ["latitude", "lat", "y"])
lon_guess = detect_column(df.columns.tolist(), ["longitude", "lon", "lng", "long", "x"])

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("2. Pilih kolom koordinat")
    lat_col = st.selectbox("Kolom latitude", df.columns.tolist(), index=df.columns.tolist().index(lat_guess) if lat_guess else 0)
    lon_col = st.selectbox("Kolom longitude", df.columns.tolist(), index=df.columns.tolist().index(lon_guess) if lon_guess else 0)

with col_b:
    st.subheader("Preview data")
    st.write(f"Total baris: **{len(df):,}**")
    st.dataframe(df.head(10), use_container_width=True)

# Bersihkan baris koordinat invalid
work_df = df.copy()
work_df[lat_col] = pd.to_numeric(work_df[lat_col], errors="coerce")
work_df[lon_col] = pd.to_numeric(work_df[lon_col], errors="coerce")
work_df = work_df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

if work_df.empty:
    st.error("Tidak ada baris dengan latitude/longitude valid.")
    st.stop()

min_lat, max_lat = float(work_df[lat_col].min()), float(work_df[lat_col].max())
min_lon, max_lon = float(work_df[lon_col].min()), float(work_df[lon_col].max())

st.subheader("3. Masukkan koordinat polygon 4 titik")
st.caption("Isi dengan urutan mengelilingi area: titik 1 → 2 → 3 → 4. Longitude dulu atau latitude dulu? Di form ini dipisah jelas.")

# Default polygon mengikuti bounding box data agar langsung bisa dites
pad_lat = (max_lat - min_lat) * 0.05 if max_lat != min_lat else 0.001
pad_lon = (max_lon - min_lon) * 0.05 if max_lon != min_lon else 0.001
DEFAULTS = [
    (min_lat - pad_lat, min_lon - pad_lon),
    (min_lat - pad_lat, max_lon + pad_lon),
    (max_lat + pad_lat, max_lon + pad_lon),
    (max_lat + pad_lat, min_lon - pad_lon),
]

coord_cols = st.columns(4)
polygon_lat_lon = []
for i in range(4):
    with coord_cols[i]:
        st.markdown(f"**Titik {i+1}**")
        lat_val = st.text_input(f"Latitude {i+1}", value=f"{DEFAULTS[i][0]:.9f}", key=f"lat_{i}")
        lon_val = st.text_input(f"Longitude {i+1}", value=f"{DEFAULTS[i][1]:.9f}", key=f"lon_{i}")
        polygon_lat_lon.append((lat_val, lon_val))

run = st.button("Proses polygon", type="primary", use_container_width=True)

if not run:
    st.stop()

try:
    # Simpan sebagai (lon, lat) untuk hitungan geometry
    polygon_lon_lat = [(normalize_number(lon), normalize_number(lat)) for lat, lon in polygon_lat_lon]
except Exception as exc:
    st.error(f"Koordinat polygon belum valid: {exc}")
    st.stop()

if len(set(polygon_lon_lat)) < 3:
    st.error("Polygon minimal butuh 3 titik unik. Cek lagi koordinatnya.")
    st.stop()

mask = point_in_polygon(work_df, lon_col, lat_col, polygon_lon_lat)
result_df = work_df.loc[mask].copy()

st.subheader("4. Hasil proses")
metric_cols = st.columns(3)
metric_cols[0].metric("Total titik input valid", f"{len(work_df):,}")
metric_cols[1].metric("Titik dalam polygon", f"{len(result_df):,}")
metric_cols[2].metric("Titik di luar polygon", f"{len(work_df) - len(result_df):,}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(work_df[lon_col], work_df[lat_col], s=18, alpha=0.35, label="Semua titik")
if not result_df.empty:
    ax.scatter(result_df[lon_col], result_df[lat_col], s=28, alpha=0.9, label="Dalam polygon")

closed_polygon = polygon_lon_lat + [polygon_lon_lat[0]]
poly_lon = [p[0] for p in closed_polygon]
poly_lat = [p[1] for p in closed_polygon]
ax.plot(poly_lon, poly_lat, linewidth=2, label="Polygon")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Plot titik reflectance dan polygon")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig, use_container_width=True)

st.dataframe(result_df, use_container_width=True)

output_name = "reflectance_polygon_output.csv"
st.download_button(
    label="Download CSV hasil polygon",
    data=csv_download_bytes(result_df),
    file_name=output_name,
    mime="text/csv",
    type="primary",
    use_container_width=True,
)

if result_df.empty:
    st.warning("Tidak ada titik masuk polygon. Coba cek urutan atau koordinat polygon.")
else:
    st.success(f"CSV siap: {output_name}")
