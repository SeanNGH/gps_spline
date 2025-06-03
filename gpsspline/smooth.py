import numpy as np
from scipy.linalg import solve
from scipy.interpolate import BSpline
from .bspline import PyBSpline
from .utils import estimate_rms_derivative, estimate_effective_sample_size, detect_outliers_student_t

class PySmoothingSpline(PyBSpline):
    def __init__(self, t, x, k=4, sigma=15, lam=None, T=2, auto_lambda=False, auto_lambda_ranged=False):
        super().__init__(t, k)
        self.t = np.asarray(t)
        self.x = np.asarray(x)
        self.k = k
        self.T = T
        self.sigma = sigma

        if auto_lambda:
            self.lam = self.calculate_optimal_lambda()
        elif auto_lambda_ranged:
            self.lam = 1.0  # khởi tạo
        else:
            self.lam = lam if lam is not None else 1.0

        self.coeffs = self._solve()

        if auto_lambda_ranged:
            self.optimize_lambda_ranged(sigma_s=self.sigma)

        print(f"lambda: {self.lam}")
        print(f"num knots: {len(self.knots)}, num basis: {len(self.basis)}")

    def _solve(self):
        X = self.design_matrix(self.t)
        V = self._derivative_matrix(self.T)
        epsilon = 1e-6  # tránh ma trận gần suy biến
        A = X.T @ X + self.lam * (V.T @ V) + epsilon * np.eye(X.shape[1])
        b = X.T @ self.x
        return solve(A, b)

    def _derivative_matrix(self, order):
        return np.array([[b.derivative(order)(t) for b in self.basis] for t in self.t])

    def evaluate(self, t):
        spline = BSpline(self.knots, self.coeffs, self.k - 1)
        return spline(t)

    def fit_irls_t_distribution(self, nu=4.5, sigma_s=15, max_iter=20, tol=1e-4):
        X = self.design_matrix(self.t)
        x = self.x
        V = self._derivative_matrix(self.T)
        coeffs = np.copy(self.coeffs)

        for _ in range(max_iter):
            residuals = x - X @ coeffs
            weights_diag = 1.0 / (1 + (residuals**2) * (nu / sigma_s**2))
            W = np.diag(weights_diag)

            A = X.T @ W @ X + self.lam * (V.T @ V)
            b = X.T @ W @ x
            coeffs_new = solve(A, b)

            if np.linalg.norm(coeffs_new - coeffs) < tol:
                break
            coeffs = coeffs_new

        self.coeffs = coeffs
        return coeffs

    def calculate_optimal_lambda(self, spectrum='omega^-2'):
        x_rms_T = estimate_rms_derivative(self.t, self.x, self.T)
        delta_t = np.mean(np.diff(self.t))
        urms = estimate_rms_derivative(self.t, self.x, 1)
        Gamma = self.sigma / (urms * delta_t)

        n_eff = estimate_effective_sample_size(Gamma, spectrum_type=spectrum)
        lam_opt = (1 - 1 / n_eff) / (x_rms_T ** 2)
        return lam_opt

    def optimize_lambda_ranged(self, lam_range=np.logspace(-6, 3, 30), nu=4.5, sigma_s=15, beta=1 / 100):
        best_mse = np.inf
        best_lam = self.lam
        best_coeffs = self.coeffs

        for lam_try in lam_range:
            if lam_try < 1e-5:
                continue  # bỏ qua lambda quá nhỏ gây bất ổn

            self.lam = lam_try
            coeffs_try = self._solve()
            self.coeffs = coeffs_try
            mse_try = self.calculate_true_mse(sigma_s)

            if mse_try < best_mse:
                best_mse = mse_try
                best_lam = lam_try
                best_coeffs = coeffs_try

        self.lam = best_lam
        self.coeffs = best_coeffs
        return best_lam

    def calculate_ranged_mse(self, coeffs, nu=4.5, sigma_s=15, beta=1 / 100):
        X = self.design_matrix(self.t)
        residuals = self.x - X @ coeffs
        mask = ~detect_outliers_student_t(residuals, nu, sigma_s, beta)
        return np.mean(residuals[mask] ** 2)

    def smoothing_matrix(self):
        X = self.design_matrix(self.t)
        V = self._derivative_matrix(self.T)
        A = X.T @ X + self.lam * (V.T @ V) + 1e-6 * np.eye(X.shape[1])
        A_inv = np.linalg.inv(A)
        return X @ A_inv @ X.T

    def calculate_true_mse(self, sigma_noise):
        N = len(self.x)
        S = self.smoothing_matrix()
        residual_term = np.linalg.norm((S - np.eye(N)) @ self.x)**2 / N
        trace_term = sigma_noise**2 * np.trace(S) / N
        return residual_term + trace_term - sigma_noise**2
