import pandas as pd
import numpy as np
import folium
from gpsspline.gps import PyGPSSmoothingSpline

# Đọc dữ liệu
df = pd.read_csv("data/OSMC_30day_982f_f952_0246.csv", skiprows=[1], dtype={
    'latitude': 'float64',
    'longitude': 'float64'
})
df = df[['time', 'latitude', 'longitude']].dropna().head(50)

# Thời gian tính bằng giây
df['t_sec'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%SZ')
df['t_sec'] = df['t_sec'].astype('int64') // 1_000_000_000
df['t_sec'] -= df['t_sec'].min()

t = df['t_sec'].to_numpy()
lat = df['latitude'].to_numpy()
lon = df['longitude'].to_numpy()

# Làm mượt
model = PyGPSSmoothingSpline(
    lat=lat,
    lon=lon,
    t=t,
    k=4,
    lam=None,
    sigma=8.5,
    T=2,
    remove_velocity=True
)

lat_smooth, lon_smooth = model.evaluate(t)

# Tạo bản đồ centered tại điểm trung bình
center = [np.mean(lat), np.mean(lon)]
m = folium.Map(location=center, zoom_start=6, tiles='OpenStreetMap')

# Thêm đường gốc (đỏ) và đường làm mượt (xanh)
original_coords = list(zip(lat, lon))
smoothed_coords = list(zip(lat_smooth, lon_smooth))

folium.PolyLine(original_coords, color='red', weight=2.5, opacity=0.5, tooltip='Original').add_to(m)
folium.PolyLine(smoothed_coords, color='blue', weight=3, tooltip='Smoothed').add_to(m)

# Thêm điểm bắt đầu/kết thúc
folium.Marker(original_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(original_coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)

# Lưu ra file HTML
m.save("gps_path_map.html")
print("✅ Đã lưu bản đồ vào gps_path_map.html")
