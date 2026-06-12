# reflectance_polygon_app.py
# Klik 4 titik di peta → otomatis jadi koordinat polygon → filter CSV reflectance
# Jalankan: streamlit run reflectance_polygon_app.py

import sys
import subprocess

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("folium")
_ensure("streamlit_folium", "streamlit_folium")
_ensure("pandas")
_ensure("numpy")

from typing import List, Tuple
import numpy as np
import pandas as pd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import streamlit as st

st.set_page_config(page_title="Reflectance Polygon Acquire", layout="wide")
st.title("🛰️ Reflectance Acquire by Polygon")
st.caption("Upload CSV reflectance → gambar polygon di peta → proses → download CSV hasil.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def point_in_polygon(df, lon_col, lat_col, polygon):
    pts_x = df[lon_col].to_numpy(dtype=float)
    pts_y = df[lat_col].to_numpy(dtype=float)
    poly_arr = np.array(polygon, dtype=float)
    n = len(poly_arr)
    inside = np.zeros(len(pts_x), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly_arr[i]
        xj, yj = poly_arr[j]
        cond = ((yi > pts_y) != (yj > pts_y)) & (
            pts_x < (xj - xi) * (pts_y - yi) / (yj - yi + 1e-15) + xi
        )
        inside ^= cond
        j = i
    return inside

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ── Sidebar: upload CSV ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload CSV")
    uploaded_file = st.file_uploader("File CSV reflectance", type=["csv"])
    st.info("Minimal ada kolom latitude & longitude.")

if uploaded_file is None:
    st.warning("Upload CSV dulu.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"CSV gagal dibaca: {e}")
    st.stop()

if df.empty:
    st.error("CSV kosong.")
    st.stop()

lat_guess = detect_column(df.columns.tolist(), ["latitude", "lat", "y"])
lon_guess = detect_column(df.columns.tolist(), ["longitude", "lon", "lng", "long", "x"])

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("2. Kolom koordinat")
    lat_col = st.selectbox("Kolom latitude", df.columns.tolist(),
        index=df.columns.tolist().index(lat_guess) if lat_guess else 0)
    lon_col = st.selectbox("Kolom longitude", df.columns.tolist(),
        index=df.columns.tolist().index(lon_guess) if lon_guess else 0)
with col_b:
    st.subheader("Preview data")
    st.write(f"Total baris: **{len(df):,}**")
    st.dataframe(df.head(8), use_container_width=True)

work_df = df.copy()
work_df[lat_col] = pd.to_numeric(work_df[lat_col], errors="coerce")
work_df[lon_col] = pd.to_numeric(work_df[lon_col], errors="coerce")
work_df = work_df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

if work_df.empty:
    st.error("Tidak ada baris dengan koordinat valid.")
    st.stop()

center_lat = float(work_df[lat_col].mean())
center_lon = float(work_df[lon_col].mean())


# ── Peta interaktif ───────────────────────────────────────────────────────────
st.subheader("3. Gambar polygon di peta")
st.info(
    "**Cara pakai:**  \n"
    "1. Klik ikon **polygon** (segi lima) di toolbar kiri peta  \n"
    "2. Klik titik-titik area yang ingin diseleksi  \n"
    "3. Klik ganda / klik titik pertama untuk menutup polygon  \n"
    "4. Tekan **Proses polygon** di bawah"
)

# Tile layers yang tersedia (gratis tanpa API key)
tile_options = {
    "OpenStreetMap": "OpenStreetMap",
    "Satelit (Esri)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Terrain (Stamen)": "https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
}
tile_choice = st.selectbox("Pilih basemap", list(tile_options.keys()))

m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Tambah tile layer sesuai pilihan
if tile_choice == "OpenStreetMap":
    pass  # default sudah OSM
elif tile_choice == "Satelit (Esri)":
    folium.TileLayer(
        tiles=tile_options[tile_choice],
        attr="Esri World Imagery",
        name="Satelit",
        overlay=False,
        control=True,
    ).add_to(m)
else:
    folium.TileLayer(
        tiles=tile_options[tile_choice],
        attr="Stamen Terrain",
        name="Terrain",
        overlay=False,
        control=True,
    ).add_to(m)

# Plot semua titik CSV di peta sebagai layer tipis
sample = work_df if len(work_df) <= 2000 else work_df.sample(2000, random_state=42)
for _, row in sample.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=3,
        color="#3b82f6",
        fill=True,
        fill_opacity=0.5,
        weight=0,
        tooltip=f"lat={row[lat_col]:.6f}, lon={row[lon_col]:.6f}",
    ).add_to(m)

# Draw toolbar: hanya polygon
draw = Draw(
    draw_options={
        "polygon": True,
        "polyline": False,
        "rectangle": True,   # rectangle juga berguna
        "circle": False,
        "marker": False,
        "circlemarker": False,
    },
    edit_options={"edit": True, "remove": True},
)
draw.add_to(m)

map_data = st_folium(m, height=520, width=None, returned_objects=["all_drawings"])

# ── Ekstrak koordinat dari gambar user ───────────────────────────────────────
polygon_lon_lat = []

if map_data and map_data.get("all_drawings"):
    drawings = map_data["all_drawings"]
    if drawings:
        last = drawings[-1]
        geo_type = last.get("geometry", {}).get("type", "")
        coords = last.get("geometry", {}).get("coordinates", [])

        if geo_type == "Polygon" and coords:
            # GeoJSON Polygon: coords[0] = list of [lon, lat], terakhir = titik pertama (closed)
            ring = coords[0]
            polygon_lon_lat = [(c[0], c[1]) for c in ring[:-1]]  # hapus duplikat penutup
        elif geo_type == "LineString" and coords:
            polygon_lon_lat = [(c[0], c[1]) for c in coords]

# Tampilkan koordinat yang terdeteksi
if polygon_lon_lat:
    st.success(f"✅ Polygon terdeteksi: **{len(polygon_lon_lat)} titik**")
    coord_df = pd.DataFrame(polygon_lon_lat, columns=["Longitude", "Latitude"])
    coord_df.index += 1
    st.dataframe(coord_df, use_container_width=True)
else:
    # Fallback: input manual
    st.warning("Belum ada polygon dari peta. Atau isi koordinat manual di bawah.")
    st.subheader("Input koordinat manual (fallback)")
    min_lat = float(work_df[lat_col].min())
    max_lat = float(work_df[lat_col].max())
    min_lon = float(work_df[lon_col].min())
    max_lon = float(work_df[lon_col].max())
    pad_lat = (max_lat - min_lat) * 0.05 or 0.001
    pad_lon = (max_lon - min_lon) * 0.05 or 0.001
    DEFAULTS = [
        (min_lat - pad_lat, min_lon - pad_lon),
        (min_lat - pad_lat, max_lon + pad_lon),
        (max_lat + pad_lat, max_lon + pad_lon),
        (max_lat + pad_lat, min_lon - pad_lon),
    ]
    ccols = st.columns(4)
    manual_pts = []
    for i in range(4):
        with ccols[i]:
            st.markdown(f"**Titik {i+1}**")
            la = st.text_input(f"Latitude {i+1}", value=f"{DEFAULTS[i][0]:.9f}", key=f"lat_{i}")
            lo = st.text_input(f"Longitude {i+1}", value=f"{DEFAULTS[i][1]:.9f}", key=f"lon_{i}")
            try:
                manual_pts.append((float(lo.replace(",", ".")), float(la.replace(",", "."))))
            except Exception:
                pass
    if len(manual_pts) == 4:
        polygon_lon_lat = manual_pts

# ── Tombol proses ─────────────────────────────────────────────────────────────
run = st.button("🔍 Proses polygon", type="primary", use_container_width=True)
if not run:
    st.stop()

if len(polygon_lon_lat) < 3:
    st.error("Polygon minimal butuh 3 titik. Gambar polygon di peta dulu.")
    st.stop()

# ── Filter & hasil ────────────────────────────────────────────────────────────
mask = point_in_polygon(work_df, lon_col, lat_col, polygon_lon_lat)
result_df = work_df.loc[mask].copy()

st.subheader("4. Hasil proses")
mc = st.columns(3)
mc[0].metric("Total titik valid", f"{len(work_df):,}")
mc[1].metric("Dalam polygon",     f"{len(result_df):,}")
mc[2].metric("Di luar polygon",   f"{len(work_df) - len(result_df):,}")

# ── Plot hasil di peta kedua ──────────────────────────────────────────────────
st.subheader("5. Visualisasi hasil")
m2 = folium.Map(location=[center_lat, center_lon], zoom_start=13)

if tile_choice != "OpenStreetMap":
    folium.TileLayer(
        tiles=tile_options[tile_choice],
        attr="Basemap",
        name=tile_choice,
    ).add_to(m2)

# Semua titik — biru tipis
for _, row in sample.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=3, color="#94a3b8", fill=True, fill_opacity=0.4, weight=0,
    ).add_to(m2)

# Titik dalam polygon — merah
for _, row in result_df.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=5, color="#ef4444", fill=True, fill_opacity=0.85, weight=0,
        tooltip=f"lat={row[lat_col]:.6f}, lon={row[lon_col]:.6f}",
    ).add_to(m2)

# Garis polygon
if polygon_lon_lat:
    poly_latlon = [(p[1], p[0]) for p in polygon_lon_lat]
    poly_latlon.append(poly_latlon[0])
    folium.PolyLine(poly_latlon, color="#16a34a", weight=2.5).add_to(m2)

st_folium(m2, height=450, width=None, returned_objects=[])

# ── Tabel & download ──────────────────────────────────────────────────────────
st.dataframe(result_df, use_container_width=True)

output_name = "reflectance_polygon_output.csv"
st.download_button(
    label="⬇️ Download CSV hasil polygon",
    data=csv_bytes(result_df),
    file_name=output_name,
    mime="text/csv",
    type="primary",
    use_container_width=True,
)

if result_df.empty:
    st.warning("Tidak ada titik dalam polygon. Coba gambar ulang area yang lebih luas.")
else:
    st.success(f"✅ {len(result_df):,} titik berhasil diekstrak → {output_name}")
