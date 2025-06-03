import numpy as np
from .smooth import PySmoothingSpline

class PyBivariateSmoothingSpline:
    def __init__(self, t, x, y, k=4, lam=None, sigma=8.5, T=2, remove_velocity=True):
        self.t = np.asarray(t)
        self.k = k
        self.T = T
        self.sigma = sigma
        self.lam = lam
        self.remove_velocity = remove_velocity

        # 1. Ước lượng thành phần vận tốc trung bình nếu cần
        if self.remove_velocity:
            self.poly_x = np.polyfit(t, x, T + 1)
            self.poly_y = np.polyfit(t, y, T + 1)
            residual_x = x - np.polyval(self.poly_x, t)
            residual_y = y - np.polyval(self.poly_y, t)
        else:
            self.poly_x = self.poly_y = None
            residual_x = x
            residual_y = y

        # 2. Khởi tạo spline smoothing với auto_lambda
        self.spline_x = PySmoothingSpline(
            t, residual_x, k, sigma=sigma, T=T, auto_lambda_ranged=True
        )
        self.spline_y = PySmoothingSpline(
            t, residual_y, k, sigma=sigma, T=T, auto_lambda_ranged=True
        )

        # 3. Bổ sung robust fit nếu dữ liệu nhiễu
        self.spline_x.fit_irls_t_distribution()
        self.spline_y.fit_irls_t_distribution()

        print("Poly coef X:", self.poly_x)
        print("Poly coef Y:", self.poly_y)

    def evaluate(self, t_eval):
        x_smooth = self.spline_x.evaluate(t_eval)
        y_smooth = self.spline_y.evaluate(t_eval)

        if self.remove_velocity:
            x_trend = np.polyval(self.poly_x, t_eval)
            y_trend = np.polyval(self.poly_y, t_eval)
            return x_smooth + x_trend, y_smooth + y_trend
        else:
            return x_smooth, y_smooth
