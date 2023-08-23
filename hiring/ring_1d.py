import numpy as np
import scipy

__all__ = ['Ring1D']


class Ring1D:

    def __init__(self, x, camera_width=72., m_to_muas=3.8, target_flux=0.66):
        if isinstance(x, str):
            x = [x]
        if hasattr(x, '__iter__') and isinstance(x[0], str):
            try:
                x = np.asarray([np.load(_) for _ in x])
                assert x.ndim == 2
            except Exception:
                raise ValueError
        else:
            try:
                x = np.atleast_2d(x)
                assert x.ndim == 2
            except Exception:
                raise ValueError
        self.x = x
        self.camera_width = float(camera_width)
        self.m_to_muas = float(m_to_muas)
        self.target_flux = float(target_flux)
        self.flux = np.mean(self.x, axis=1)
        self.flux *= (self.target_flux / np.mean(self.flux))
        self.fft_x = None
        self.fftfreq_x = None

    def fft(self, n_pad=200000, workers=-1):
        self.fftfreq()
        x = np.zeros((self.x.shape[0], self.x.shape[1] + 2 * n_pad), dtype=self.x.dtype)
        x[:, n_pad:(self.x.shape[1] + n_pad)] = self.x
        self.fft_x = scipy.fft.fft(x, norm='backward', overwrite_x=True, workers=workers)
        self.fft_x *= (self.flux[:, np.newaxis] / self.fft_x[:, :1])
        return self.fft_x

    def fftfreq(self, n_pad=200000):
        self.fftfreq_x = scipy.fft.fftfreq(self.x.shape[1] + 2 * n_pad,
                                           self.camera_width * self.m_to_muas / 206265. / 1e6 /
                                           self.x.shape[1]) / 1e9
        return self.fftfreq_x

    def r_flux(self, indices):
        return np.mean(self.flux) / np.mean(self.flux[indices])
