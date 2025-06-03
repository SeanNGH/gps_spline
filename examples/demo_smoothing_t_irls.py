import numpy as np
import matplotlib.pyplot as plt
from gpsspline.gps import PyGPSSmoothingSpline

# ==== Bước 1: Tạo dữ liệu GPS giả định ====
np.random.seed(0)
n_points = 100
t = np.linspace(0, 10, n_points)

# Đường chuẩn hình sin
center_lat = 21.03
center_lon = 105.85
length_deg = 0.02

lon = center_lon + np.linspace(-length_deg/2, length_deg/2, n_points)
lat = center_lat + 0.003 * np.sin(3 * np.pi * lon / length_deg)

# ==== Nhiễu Gaussian nhẹ + chèn outlier ====
noise_scale = 0.0001
noise_lat = noise_scale * np.random.normal(0, 1, size=n_points)
noise_lon = noise_scale * np.random.normal(0, 1, size=n_points)

# Chèn outlier tại 4 vị trí
outlier_indices = [20,33, 40, 50, 60, 75]
noise_lat[outlier_indices] += 0.0015
noise_lon[outlier_indices] -= 0.0015

# Tổng hợp dữ liệu nhiễu
lat_noisy = lat + noise_lat
lon_noisy = lon + noise_lon


# ==== Bước 2: Làm mượt với PyGPSSmoothingSpline ====
model = PyGPSSmoothingSpline(
    lat=lat_noisy,
    lon=lon_noisy,
    t=t,
    k=4,
    lam=None,       # Tự động tính λ
    sigma=15,      # Đơn vị: mét
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
