# reflectance_polygon_app.py
# Peta interaktif Folium + Plotly Mapbox
# Download peta sebagai HTML (berisi tombol Save PNG via html2canvas)
# Jalankan: streamlit run reflectance_polygon_app.py

import sys, subprocess

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("plotly")
_ensure("pandas")
_ensure("numpy")
_ensure("streamlit_folium", "streamlit_folium")
_ensure("folium")

import numpy as np
import pandas as pd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Reflectance Polygon Acquire", layout="wide")

# ── Navigasi Menu ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ Reflectance Tools")
    st.markdown("---")
    menu = st.radio(
        "Pilih Menu",
        options=["📂 Polygon dari CSV", "🌍 Ambil Data Sentinel"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("v1.2.0")

# ════════════════════════════════════════════════════════════════════
# MENU: Sentinel — Coming Soon
# ════════════════════════════════════════════════════════════════════
if menu == "🌍 Ambil Data Sentinel":
    st.title("🌍 Ambil Data Reflectance — Sentinel")
    st.markdown("---")

    col_mid = st.columns([1, 2, 1])[1]
    with col_mid:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
                border: 1px solid #334155;
                border-radius: 16px;
                padding: 48px 32px;
                text-align: center;
                margin-top: 32px;
            ">
                <div style="font-size: 64px; margin-bottom: 16px;">🛰️</div>
                <h2 style="color: #f1f5f9; margin-bottom: 8px;">Coming Soon</h2>
                <p style="color: #94a3b8; font-size: 16px; margin-bottom: 24px;">
                    Fitur ini sedang dalam pengembangan.<br>
                    Kami sedang membangun integrasi langsung ke <strong style="color:#38bdf8;">Sentinel Hub API</strong>
                    untuk mengambil data reflectance otomatis tanpa perlu upload CSV manual.
                </p>
                <div style="
                    display: inline-block;
                    background: #1e40af22;
                    border: 1px solid #3b82f6;
                    border-radius: 8px;
                    padding: 10px 20px;
                    color: #60a5fa;
                    font-size: 13px;
                    margin-bottom: 28px;
                ">⚙️ &nbsp; In Progress
                </div>
                <hr style="border-color: #334155; margin: 24px 0;">
                <p style="color: #64748b; font-size: 13px; margin: 0;">
                    Fitur yang direncanakan:
                </p>
                <ul style="color: #94a3b8; font-size: 13px; text-align: left; margin-top: 12px; line-height: 2;">
                    <li>🗓️ Pilih rentang tanggal akuisisi</li>
                    <li>🗺️ Gambar area of interest di peta</li>
                    <li>☁️ Filter cloud cover otomatis</li>
                    <li>📡 Download band B2, B3, B4, B8 langsung</li>
                    <li>📊 Preview citra dan ekspor CSV reflectance</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💬 Pantau update atau hubungi developer untuk info lebih lanjut.")
    st.stop()

# ════════════════════════════════════════════════════════════════════
# MENU: Polygon dari CSV (menu utama)
# ════════════════════════════════════════════════════════════════════
st.title("📂 Reflectance Acquire by Polygon")
st.caption("Upload CSV → gambar polygon di peta → proses → download hasil.")


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
    poly = np.array(polygon, dtype=float)
    n = len(poly)
    inside = np.zeros(len(pts_x), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > pts_y) != (yj > pts_y)) & (
            pts_x < (xj - xi) * (pts_y - yi) / (yj - yi + 1e-15) + xi)
        inside ^= cond
        j = i
    return inside

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")

# Tombol Save PNG yang di-inject ke dalam HTML file (via html2canvas CDN)
PNG_INJECT = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<style>
  #save-png-btn {
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: #16a34a; color: #fff; border: none; border-radius: 8px;
    padding: 12px 22px; font-size: 15px; font-weight: 600;
    cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    transition: background 0.2s;
  }
  #save-png-btn:hover { background: #15803d; }
  #save-png-msg {
    position: fixed; bottom: 74px; right: 24px; z-index: 9999;
    background: #1e293b; color: #fff; padding: 8px 16px;
    border-radius: 6px; font-size: 13px; display: none;
  }
</style>
<button id="save-png-btn">🖼️ Save as PNG</button>
<div id="save-png-msg">Sedang memproses...</div>
<script>
document.getElementById("save-png-btn").addEventListener("click", function() {
  var msg = document.getElementById("save-png-msg");
  msg.style.display = "block";
  msg.innerText = "Sedang memproses...";
  setTimeout(function() {
    html2canvas(document.body, {
      useCORS: true, allowTaint: true,
      width: window.innerWidth, height: window.innerHeight,
      scale: 2
    }).then(function(canvas) {
      var link = document.createElement("a");
      link.download = "peta_reflectance_polygon.png";
      link.href = canvas.toDataURL("image/png");
      link.click();
      msg.innerText = "Tersimpan!";
      setTimeout(function() { msg.style.display = "none"; }, 2500);
    }).catch(function(err) {
      msg.innerText = "Gagal: " + err;
      setTimeout(function() { msg.style.display = "none"; }, 3000);
    });
  }, 300);
});
</script>
"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    st.error(f"CSV gagal dibaca: {e}"); st.stop()

if df.empty:
    st.error("CSV kosong."); st.stop()

lat_guess = detect_column(df.columns.tolist(), ["latitude","lat","y"])
lon_guess = detect_column(df.columns.tolist(), ["longitude","lon","lng","long","x"])

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("2. Kolom koordinat")
    lat_col = st.selectbox("Kolom latitude",  df.columns.tolist(),
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
    st.error("Tidak ada baris dengan koordinat valid."); st.stop()

center_lat = float(work_df[lat_col].mean())
center_lon = float(work_df[lon_col].mean())


# ════════════════════════════════════════════════════════════════════
# STEP 3 — Peta Folium untuk GAMBAR POLYGON
# ════════════════════════════════════════════════════════════════════
st.subheader("3. Gambar polygon di peta")

tile_choice = st.selectbox("Basemap", ["OpenStreetMap", "Satelit (Esri)", "Satelit (Google)"])
tile_map = {
    "OpenStreetMap": ("OpenStreetMap", "© OpenStreetMap contributors"),
    "Satelit (Esri)": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Tiles © Esri"),
    "Satelit (Google)": (
        "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "© Google"),
}
tile_url, tile_attr = tile_map[tile_choice]

m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
               tiles=tile_url, attr=tile_attr)

sample = work_df if len(work_df) <= 3000 else work_df.sample(3000, random_state=42)
fg = folium.FeatureGroup(name="Titik CSV")
for _, row in sample.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=4, color="#3b82f6", fill=True, fill_opacity=0.7, weight=0,
        tooltip=f"{row[lat_col]:.6f}, {row[lon_col]:.6f}",
    ).add_to(fg)
fg.add_to(m)

Draw(
    draw_options={
        "polygon":   {"shapeOptions": {"color": "#16a34a"}},
        "rectangle": {"shapeOptions": {"color": "#f59e0b"}},
        "polyline": False, "circle": False,
        "marker": False, "circlemarker": False,
    },
    edit_options={"edit": True, "remove": True},
).add_to(m)
folium.LayerControl().add_to(m)

st.info("**Cara pakai:** Klik ikon ⬡ polygon di toolbar kiri → klik titik area → tutup polygon → **Proses**")
map_data = st_folium(m, height=500, use_container_width=True, returned_objects=["all_drawings"])

# ── Ekstrak koordinat polygon ─────────────────────────────────────────────────
polygon_lon_lat = []
if map_data and map_data.get("all_drawings"):
    last = map_data["all_drawings"][-1]
    geo_type = last.get("geometry", {}).get("type", "")
    coords   = last.get("geometry", {}).get("coordinates", [])
    if geo_type == "Polygon" and coords:
        polygon_lon_lat = [(c[0], c[1]) for c in coords[0][:-1]]
    elif geo_type in ("LineString", "MultiPoint") and coords:
        polygon_lon_lat = [(c[0], c[1]) for c in coords]

if polygon_lon_lat:
    st.success(f"✅ Polygon terdeteksi — **{len(polygon_lon_lat)} titik**")
    cdf = pd.DataFrame(polygon_lon_lat, columns=["Longitude","Latitude"])
    cdf.index += 1
    st.dataframe(cdf, use_container_width=True)
else:
    st.warning("Belum ada polygon. Gambar dulu di peta, atau isi manual di bawah.")
    st.subheader("Input koordinat manual (opsional)")
    min_lat, max_lat = float(work_df[lat_col].min()), float(work_df[lat_col].max())
    min_lon, max_lon = float(work_df[lon_col].min()), float(work_df[lon_col].max())
    pad_lat = (max_lat - min_lat) * 0.05 or 0.001
    pad_lon = (max_lon - min_lon) * 0.05 or 0.001
    DEFS = [(min_lat-pad_lat, min_lon-pad_lon),(min_lat-pad_lat, max_lon+pad_lon),
            (max_lat+pad_lat, max_lon+pad_lon),(max_lat+pad_lat, min_lon-pad_lon)]
    ccols = st.columns(4)
    manual = []
    for i in range(4):
        with ccols[i]:
            st.markdown(f"**Titik {i+1}**")
            la = st.text_input(f"Lat {i+1}", value=f"{DEFS[i][0]:.9f}", key=f"lat_{i}")
            lo = st.text_input(f"Lon {i+1}", value=f"{DEFS[i][1]:.9f}", key=f"lon_{i}")
            try: manual.append((float(lo.replace(",",".")), float(la.replace(",","."))))
            except: pass
    if len(manual) == 4:
        polygon_lon_lat = manual

# ── Proses ────────────────────────────────────────────────────────────────────
run = st.button("🔍 Proses polygon", type="primary", use_container_width=True)
if not run:
    st.stop()

if len(polygon_lon_lat) < 3:
    st.error("Polygon minimal 3 titik. Gambar dulu di peta."); st.stop()

mask = point_in_polygon(work_df, lon_col, lat_col, polygon_lon_lat)
result_df = work_df.loc[mask].copy()

# ── Metrik ────────────────────────────────────────────────────────────────────
st.subheader("4. Hasil")
mc = st.columns(3)
mc[0].metric("Total titik valid",  f"{len(work_df):,}")
mc[1].metric("Dalam polygon",      f"{len(result_df):,}")
mc[2].metric("Di luar polygon",    f"{len(work_df)-len(result_df):,}")

# ════════════════════════════════════════════════════════════════════
# STEP 5 — Peta HASIL (Plotly Mapbox, tampil di app)
# ════════════════════════════════════════════════════════════════════
st.subheader("5. Visualisasi hasil di peta")

mapbox_style = "open-street-map" if tile_choice == "OpenStreetMap" else "carto-positron"
closed = polygon_lon_lat + [polygon_lon_lat[0]]
poly_lon_list = [p[0] for p in closed]
poly_lat_list = [p[1] for p in closed]

fig = go.Figure()
fig.add_trace(go.Scattermapbox(
    lat=work_df[lat_col], lon=work_df[lon_col],
    mode="markers", marker=dict(size=6, color="#3b82f6", opacity=0.4),
    name="Semua titik",
    hovertemplate="lat: %{lat:.6f}<br>lon: %{lon:.6f}<extra></extra>",
))
if not result_df.empty:
    fig.add_trace(go.Scattermapbox(
        lat=result_df[lat_col], lon=result_df[lon_col],
        mode="markers", marker=dict(size=9, color="#ef4444", opacity=0.9),
        name="Dalam polygon",
        hovertemplate="lat: %{lat:.6f}<br>lon: %{lon:.6f}<extra></extra>",
    ))
fig.add_trace(go.Scattermapbox(
    lat=poly_lat_list, lon=poly_lon_list,
    mode="lines", line=dict(color="#16a34a", width=3),
    name="Polygon",
))
fig.update_layout(
    mapbox=dict(style=mapbox_style, center=dict(lat=center_lat, lon=center_lon), zoom=13),
    margin=dict(l=0, r=0, t=0, b=0), height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# STEP 6 — Download: HTML (+ tombol Save PNG di dalamnya) + CSV
# ════════════════════════════════════════════════════════════════════
st.subheader("6. Download peta & data")

# Buat peta Folium hasil lengkap
m_result = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                      tiles=tile_url, attr=tile_attr)

fg_all = folium.FeatureGroup(name="Semua titik")
for _, row in work_df.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=4, color="#3b82f6", fill=True, fill_opacity=0.5, weight=0,
        tooltip=f"lat={row[lat_col]:.6f}, lon={row[lon_col]:.6f}",
    ).add_to(fg_all)
fg_all.add_to(m_result)

if not result_df.empty:
    fg_in = folium.FeatureGroup(name="Dalam polygon")
    for _, row in result_df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6, color="#ef4444", fill=True, fill_opacity=0.9, weight=0,
            tooltip=f"lat={row[lat_col]:.6f}, lon={row[lon_col]:.6f}",
        ).add_to(fg_in)
    fg_in.add_to(m_result)

closed_latlon = [(p[1], p[0]) for p in closed]
folium.PolyLine(closed_latlon, color="#16a34a", weight=3).add_to(m_result)
folium.Polygon(locations=closed_latlon, color="#16a34a", weight=2,
               fill=True, fill_color="#16a34a", fill_opacity=0.08).add_to(m_result)
folium.LayerControl().add_to(m_result)

# Inject tombol Save PNG ke HTML sebelum </body>
raw_html = m_result.get_root().render()
html_with_btn = raw_html.replace("</body>", PNG_INJECT + "\n</body>")
html_bytes_encoded = html_with_btn.encode("utf-8")

# Tombol download
dl_cols = st.columns(2)
with dl_cols[0]:
    st.download_button(
        label="🗺️ Download Peta (HTML)",
        data=html_bytes_encoded,
        file_name="peta_reflectance_polygon.html",
        mime="text/html",
        use_container_width=True,
        help="Buka di browser → klik tombol hijau 'Save as PNG' di pojok kanan bawah",
    )
with dl_cols[1]:
    st.download_button(
        label="⬇️ Download CSV hasil polygon",
        data=csv_bytes(result_df),
        file_name="reflectance_polygon_output.csv",
        mime="text/csv",
        use_container_width=True,
        help=f"{len(result_df):,} baris titik dalam polygon",
    )

st.info(
    "💡 **Cara save PNG:** Download HTML → buka di Chrome/Firefox → "
    "klik tombol hijau **🖼️ Save as PNG** di pojok kanan bawah peta."
)

# ── Tabel ─────────────────────────────────────────────────────────────────────
st.dataframe(result_df, use_container_width=True)

if result_df.empty:
    st.warning("Tidak ada titik dalam polygon. Coba perluas area gambar.")
else:
    st.success(f"✅ {len(result_df):,} titik berhasil diekstrak.")
