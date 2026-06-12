# reflectance_polygon_app.py
# Aplikasi sederhana: upload CSV reflectance, input 4 koordinat polygon,
# filter titik di dalam polygon, plot, lalu download CSV hasil.
# Jalankan: streamlit run reflectance_polygon_app.py

from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Reflectance Polygon Acquire", layout="wide")
st.title("Reflectance Acquire by Polygon")
st.caption("Upload CSV reflectance → input 4 koordinat polygon → proses → download CSV hasil.")


def detect_column(columns: List[str], candidates: List[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_number(value) -> float:
    """Terima input angka Indonesia/English: -6,123 atau -6.123."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "")
    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    return float(text)


def point_in_polygon(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    polygon: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Ray-casting algorithm (pure numpy).
    polygon : list of (lon, lat) tuples — urutan apapun (CW/CCW) sama saja.
    Returns boolean numpy array, True = titik di dalam polygon.
    """
    pts_x = df[lon_col].to_numpy(dtype=float)
    pts_y = df[lat_col].to_numpy(dtype=float)

    poly_arr = np.array(polygon, dtype=float)
    n = len(poly_arr)
    inside = np.zeros(len(pts_x), dtype=bool)

    j = n - 1
    for i in range(n):
        xi, yi = poly_arr[i]
        xj, yj = poly_arr[j]

        # kondisi perpotongan sisi polygon dengan sinar horizontal ke kanan
        cond = ((yi > pts_y) != (yj > pts_y)) & (
            pts_x < (xj - xi) * (pts_y - yi) / (yj - yi + 1e-15) + xi
        )
        inside ^= cond
        j = i

    return inside


def csv_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Input CSV")
    uploaded_file = st.file_uploader("Upload file CSV reflectance", type=["csv"])
    st.info(
        "Format yang didukung minimal punya kolom latitude dan longitude. "
        "Kolom reflectance seperti B2_blue, B3_green, B4_red, B8_nir akan "
        "ikut terbawa ke output."
    )

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
    lat_col = st.selectbox(
        "Kolom latitude",
        df.columns.tolist(),
        index=df.columns.tolist().index(lat_guess) if lat_guess else 0,
    )
    lon_col = st.selectbox(
        "Kolom longitude",
        df.columns.tolist(),
        index=df.columns.tolist().index(lon_guess) if lon_guess else 0,
    )

with col_b:
    st.subheader("Preview data")
    st.write(f"Total baris: **{len(df):,}**")
    st.dataframe(df.head(10), use_container_width=True)

# ── Bersihkan baris koordinat invalid ────────────────────────────────────────
work_df = df.copy()
work_df[lat_col] = pd.to_numeric(work_df[lat_col], errors="coerce")
work_df[lon_col] = pd.to_numeric(work_df[lon_col], errors="coerce")
work_df = work_df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

if work_df.empty:
    st.error("Tidak ada baris dengan latitude/longitude valid.")
    st.stop()

min_lat, max_lat = float(work_df[lat_col].min()), float(work_df[lat_col].max())
min_lon, max_lon = float(work_df[lon_col].min()), float(work_df[lon_col].max())

# ── Input polygon ─────────────────────────────────────────────────────────────
st.subheader("3. Masukkan koordinat polygon 4 titik")
st.caption(
    "Isi dengan urutan mengelilingi area: titik 1 → 2 → 3 → 4. "
    "Longitude dulu atau latitude dulu? Di form ini dipisah jelas."
)

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
        lat_val = st.text_input(
            f"Latitude {i+1}", value=f"{DEFAULTS[i][0]:.9f}", key=f"lat_{i}"
        )
        lon_val = st.text_input(
            f"Longitude {i+1}", value=f"{DEFAULTS[i][1]:.9f}", key=f"lon_{i}"
        )
        polygon_lat_lon.append((lat_val, lon_val))

run = st.button("Proses polygon", type="primary", use_container_width=True)

if not run:
    st.stop()

# ── Validasi & proses ─────────────────────────────────────────────────────────
try:
    # Simpan sebagai (lon, lat) untuk hitungan geometry
    polygon_lon_lat = [
        (normalize_number(lon), normalize_number(lat)) for lat, lon in polygon_lat_lon
    ]
except Exception as exc:
    st.error(f"Koordinat polygon belum valid: {exc}")
    st.stop()

if len(set(polygon_lon_lat)) < 3:
    st.error("Polygon minimal butuh 3 titik unik. Cek lagi koordinatnya.")
    st.stop()

mask = point_in_polygon(work_df, lon_col, lat_col, polygon_lon_lat)
result_df = work_df.loc[mask].copy()

# ── Metrik ────────────────────────────────────────────────────────────────────
st.subheader("4. Hasil proses")
metric_cols = st.columns(3)
metric_cols[0].metric("Total titik input valid", f"{len(work_df):,}")
metric_cols[1].metric("Titik dalam polygon", f"{len(result_df):,}")
metric_cols[2].metric("Titik di luar polygon", f"{len(work_df) - len(result_df):,}")

# ── Plot Plotly (menggantikan matplotlib) ─────────────────────────────────────
closed_polygon = polygon_lon_lat + [polygon_lon_lat[0]]
poly_lon = [p[0] for p in closed_polygon]
poly_lat = [p[1] for p in closed_polygon]

fig = go.Figure()

# Semua titik
fig.add_trace(
    go.Scatter(
        x=work_df[lon_col],
        y=work_df[lat_col],
        mode="markers",
        marker=dict(size=5, color="steelblue", opacity=0.35),
        name="Semua titik",
    )
)

# Titik dalam polygon
if not result_df.empty:
    fig.add_trace(
        go.Scatter(
            x=result_df[lon_col],
            y=result_df[lat_col],
            mode="markers",
            marker=dict(size=7, color="orangered", opacity=0.9),
            name="Dalam polygon",
        )
    )

# Garis polygon
fig.add_trace(
    go.Scatter(
        x=poly_lon,
        y=poly_lat,
        mode="lines",
        line=dict(color="green", width=2),
        name="Polygon",
    )
)

fig.update_layout(
    title="Plot titik reflectance dan polygon",
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

# ── Tabel & download ──────────────────────────────────────────────────────────
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
