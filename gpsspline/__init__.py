
from .bspline import PyBSpline
from .interpolate import PyInterpolatingSpline
from .smooth import PySmoothingSpline
from .bivariate import PyBivariateSmoothingSpline
from .gps import PyGPSSmoothingSpline

class PySmoothingSpline(PyBSpline):
    def __init__(self, t, x, k, lam=None, sigma=None, T=2, gamma_type='omega^-2'):
        self.t = t
        self.x = x
        self.k = k
        self.T = T
        self.sigma = sigma
        self.knots = not_a_knot_vector(t, k)  # thay thế nút chuẩn

        super().__init__(t, k)  # gọi khởi tạo từ PyBSpline, tạo self.basis

        # Nếu không truyền lam nhưng có sigma thì tự động tính
        if lam is None:
            if sigma is None:
                raise ValueError("Must provide either lam or sigma for lambda estimation.")
            self.lam = self.calculate_optimal_lambda(sigma, gamma_type)
        else:
            self.lam = lam

        self.coeffs = self._solve()
