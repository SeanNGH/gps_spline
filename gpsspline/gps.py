import numpy as np
from pyproj import Transformer, CRS
from .bivariate import PyBivariateSmoothingSpline

class PyGPSSmoothingSpline:
    def __init__(self, lat, lon, t, k, lam, sigma, T=2, remove_velocity=True):
        self.lat = lat
        self.lon = lon
        self.t = t
        self.k = k
        self.lam = lam
        self.sigma = sigma
        self.T = T
        self.remove_velocity = remove_velocity

        # 1. Chiếu sang tọa độ phẳng (x, y)
        self.transformer = self._create_transformer()
        self.xy = self._project_coords()
        self.x = self.xy[:, 0]
        self.y = self.xy[:, 1]

        # 2. Loại bỏ thành phần vận tốc trung bình nếu yêu cầu
        if self.remove_velocity:
            self.poly_x = np.polyfit(t, self.x, deg=T+1)
            self.poly_y = np.polyfit(t, self.y, deg=T+1)
            residual_x = self.x - np.polyval(self.poly_x, t)
            residual_y = self.y - np.polyval(self.poly_y, t)
        else:
            self.poly_x = np.zeros(T+2)
            self.poly_y = np.zeros(T+2)
            residual_x = self.x
            residual_y = self.y

        # 3. Làm mượt phần dư
        self.spline = PyBivariateSmoothingSpline(
            t, residual_x, residual_y, k, lam, sigma, T
        )

    def _create_transformer(self):
        lon0 = np.mean(self.lon)
        crs_geo = CRS.from_epsg(4326)
        crs_proj = CRS.from_proj4(
            f"+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84"
        )
        return Transformer.from_crs(crs_geo, crs_proj, always_xy=True)

    def _project_coords(self):
        x, y = self.transformer.transform(self.lon, self.lat)
        return np.stack((x, y), axis=-1)

    def evaluate(self, t_eval):
        """
        Trả về lat/lon được làm mượt tại t_eval.
        Gồm cả thành phần spline và thành phần vận tốc trung bình đã tách.
        """
        spline_x, spline_y = self.spline.evaluate(t_eval)
        trend_x = np.polyval(self.poly_x, t_eval)
        trend_y = np.polyval(self.poly_y, t_eval)
        x_total = spline_x + trend_x
        y_total = spline_y + trend_y

        lon_eval, lat_eval = self.transformer.transform(x_total, y_total, direction='INVERSE')
        return lat_eval, lon_eval
