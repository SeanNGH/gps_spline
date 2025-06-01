import numpy as np
import matplotlib.pyplot as plt
from gpsspline.gps import PyGPSSmoothingSpline

# ==== Bước 1: Tạo dữ liệu GPS giả định ====
np.random.seed(0)
n_points = 100
t = np.linspace(0, 10, n_points)

# Tọa độ lat/lon ban đầu (theo một vòng tròn nhỏ quanh Hà Nội)
center_lat = 21.03
center_lon = 105.85
radius_deg = 0.005
lat = center_lat + radius_deg * np.sin(2 * np.pi * t / 10)
lon = center_lon + radius_deg * np.cos(2 * np.pi * t / 10)

# Thêm nhiễu Student-t
noise_lat = 0.0005 * np.random.standard_t(df=4.5, size=n_points)
noise_lon = 0.0005 * np.random.standard_t(df=4.5, size=n_points)
lat_noisy = lat + noise_lat
lon_noisy = lon + noise_lon

# ==== Bước 2: Làm mượt với PyGPSSmoothingSpline ====
model = PyGPSSmoothingSpline(
    lat=lat_noisy,
    lon=lon_noisy,
    t=t,
    k=4,
    lam=None,       # Tự động tính λ
    sigma=8.5,      # Đơn vị: mét
    T=2,
    remove_velocity=True
)

# ==== Bước 3: Đánh giá tại các điểm gốc ====
lat_smooth, lon_smooth = model.evaluate(t)

# ==== Bước 4: Hiển thị ====
plt.figure(figsize=(8, 6))
plt.plot(lon_noisy, lat_noisy, 'ro-', alpha=0.5, label="Noisy GPS")
plt.plot(lon, lat, 'k--', label="True Path")
plt.plot(lon_smooth, lat_smooth, 'b-', label="Smoothed Path")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Smoothing GPS Path using PyGPSSmoothingSpline")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
