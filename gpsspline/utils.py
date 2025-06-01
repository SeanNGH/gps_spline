import numpy as np
from scipy.signal import welch, detrend
from scipy.constants import pi
from scipy.stats import t as student_t

def estimate_rms_derivative(x, dt, order=1):
    x_detrended = detrend(x)
    f, Pxx = welch(x_detrended, fs=1/dt)
    Pxx_derivative = (2*pi*f)**(2*order) * Pxx
    return np.sqrt(np.sum(Pxx_derivative) * (f[1] - f[0]))

def robust_weights_t(residuals, df, sigma):
    return (1 + (residuals / sigma)**2 / df)**(- (df + 1) / 2)

def detect_outliers_student_t(residuals, nu=4.5, sigma_s=8.5, beta=1/100):
    """
    Trả về chỉ mục các điểm KHÔNG phải ngoại lai (giữ lại để tính MSE).
    """
    cdf = student_t.cdf(residuals / sigma_s, df=nu)
    mask = (cdf >= beta / 2) & (cdf <= 1 - beta / 2)
    return mask
def estimate_effective_sample_size(gamma, spectrum_type='omega^-2'):
    """
    Ước lượng kích thước mẫu hiệu dụng neff từ gamma = σ / (u_rms * Δt)
    dựa trên các hệ số thực nghiệm (Hình 5, bài báo Early & Sykulski 2020).
    """
    spectrum_map = {
        'omega^-2': (10.0, 0.69),
        'omega^-3': (3.0, 0.82),
        'omega^-4': (1.0, 1.5)
    }
    if spectrum_type not in spectrum_map:
        raise ValueError(f"Spectrum type '{spectrum_type}' không được hỗ trợ. Dùng một trong {list(spectrum_map.keys())}")

    C, m = spectrum_map[spectrum_type]
    neff = C * (gamma ** m)
    return neff
