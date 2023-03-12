import numpy as np
from scipy.interpolate import interp1d
from collections import OrderedDict
from .lensing_bands import *
from copy import deepcopy
# from time import time #####

__all__ = ['BandAllocation']


class BandAllocation:
    def __init__(self, base_resolution=512, camera_width=24., resolution_boost=[3, 3], spin=0.9375,
                 theta_o=163, theta_d=90, n_phi=1000, max_pixels_per_run=600000, band_config=None):
        # t0 = time() #####
        if band_config is not None:
            self._band_config = OrderedDict(np.load(band_config))
        else:
            self._band_config = OrderedDict()
            self.band_config['base_resolution'] = int(base_resolution)
            assert self.band_config['base_resolution'] > 0
            assert not self.band_config['base_resolution'] % 2 # should be even for now
            self.band_config['camera_width'] = float(camera_width)
            assert self.band_config['camera_width'] > 0
            self.band_config['resolution_boost'] = np.array(resolution_boost).astype(int).flatten()
            assert np.all(self.band_config['resolution_boost'] > 1)
            assert self.band_config['resolution_boost'].size >= 1
            self.band_config['spin'] = float(spin)
            assert 0. <= self.band_config['spin'] <= 1.
            self.band_config['theta_o'] = float(theta_o)
            assert 0. <= self.band_config['theta_o'] <= 180.
            self.band_config['theta_d'] = float(theta_d)
            assert 0. <= self.band_config['theta_d'] <= 180.
            self.band_config['n_phi'] = int(n_phi)
            assert self.band_config['n_phi'] > 0
            self.band_config['max_pixels_per_run'] = int(max_pixels_per_run)

        # print(1, time() - t0) #####
        self.n_layer = self.band_config['resolution_boost'].size + 1
        self.layer_resolution = self.band_config['base_resolution'] * np.cumprod(
            np.insert(self.band_config['resolution_boost'], 0, 1))
        self.image_resolution = self.layer_resolution[-1]
        self.layer_x = [self.get_x(self.band_config['camera_width'], _) for _ in
                        self.layer_resolution]
        self.layer_xy = [np.meshgrid(_, _) for _ in self.layer_x]
        self.phi_all = np.linspace(0.001, 359.999, self.band_config['n_phi'])

        # print(2, time() - t0) #####
        if band_config is None:
            self.band_config['r_band'] = np.array(
                [[r_b(j, self.band_config['spin'], self.band_config['theta_o'],
                self.band_config['theta_d'], i) for j in self.phi_all] for i in
                range(1, self.n_layer)])
        # print(21, time() - t0) #####
        self.r_band_interp = [
            [interp1d(self.phi_all, _r, fill_value='extrapolate', assume_sorted=True) for _r in
            _r_band] for _r_band in np.moveaxis(self.band_config['r_band'], 1, 2)]
        self.layer_i = [None]
        # print(22, time() - t0) #####
        for i in range(1, self.n_layer):
            # print(221, time() - t0) #####
            layer_r = np.sqrt(self.layer_xy[i][0]**2 + self.layer_xy[i][1]**2)
            layer_phi = np.arctan2(self.layer_xy[i][1], self.layer_xy[i][0]) % (2 * np.pi)
            layer_phi = layer_phi * 180 / np.pi
            # print(222, time() - t0) #####
            self.layer_i.append(
                np.where(np.logical_and(layer_r >= self.r_band_interp[i - 1][0](layer_phi),
                                        layer_r <= self.r_band_interp[i - 1][1](layer_phi))))
        # print(3, time() - t0) #####
        if band_config is None:
            self.band_config['x_all'] = self.layer_xy[0][0].flatten()
            self.band_config['y_all'] = self.layer_xy[0][1].flatten()
            for i in range(1, self.n_layer):
                self.band_config['x_all'] = np.concatenate((self.band_config['x_all'],
                                                            self.layer_xy[i][0][self.layer_i[i]]))
                self.band_config['y_all'] = np.concatenate((self.band_config['y_all'],
                                                            self.layer_xy[i][1][self.layer_i[i]]))
        self.layer_n = np.array([self.band_config['base_resolution']**2] +
                                [_l[0].size for _l in self.layer_i[1:]])
        self.layer_cumn = np.cumsum(np.insert(self.layer_n, 0, 0))
        if self.band_config['max_pixels_per_run'] > 0:
            self.band_config['n_run'] = int(np.ceil(self.band_config['x_all'].size /
                                                    self.band_config['max_pixels_per_run']))
        else:
            self.band_config['n_run'] = 1
        # print(4, time() - t0) #####

    @property
    def band_config(self):
        return self._band_config

    @staticmethod
    def get_x(camera_width, resolution):
        dx = camera_width / resolution
        return np.arange(-camera_width / 2 + dx / 2, camera_width / 2 + dx / 2, dx)

    def save_config(self, path):
        np.savez(path, **self.band_config)
        if self.band_config['n_run'] > 1:
            for i in range(self.band_config['n_run']):
                _band_config = deepcopy(self.band_config)
                _band_config['x_all'] = _band_config['x_all'][
                    (i * _band_config['max_pixels_per_run']):
                    ((i + 1) * _band_config['max_pixels_per_run'])]
                _band_config['y_all'] = _band_config['y_all'][
                    (i * _band_config['max_pixels_per_run']):
                    ((i + 1) * _band_config['max_pixels_per_run'])]
                with open(path + f'.{i}', 'wb') as _f:
                    np.savez(_f, **_band_config) # avoid duplicate npz extensions

    def read_image(self, path, target='I_nu'):
        self.layer_image_raw = [np.full((_r, _r), -1.) for _r in self.layer_resolution]
        if self.band_config['n_run'] > 1:
            result = np.concatenate([np.nan_to_num(np.load(path + f'.{i}')[target]) for i in
                                     range(self.band_config['n_run'])])
        else:
            result = np.nan_to_num(np.load(path)[target])
        self.layer_image_raw[0] = result[self.layer_cumn[0]:self.layer_cumn[1]].reshape(
            (self.layer_resolution[0], self.layer_resolution[0]))
        for i in range(1, self.n_layer):
            self.layer_image_raw[i][self.layer_i[i]] = result[
                self.layer_cumn[i]:self.layer_cumn[i + 1]]
        self.layer_image = [np.kron(self.layer_image_raw[i],
                                    np.ones((self.image_resolution // self.layer_resolution[i],
                                             self.image_resolution // self.layer_resolution[i])))
                            for i in range(self.n_layer)]
        self.total_image = self.layer_image[-1]
        for img in self.layer_image[::-1][1:]:
            self.total_image = np.where(self.total_image >= 0., self.total_image, img)
        assert np.all(self.total_image >= 0.)
        return self.total_image
