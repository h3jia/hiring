import numpy as np
import jax.numpy as jnp
from collections import OrderedDict
from .lensing_bands import *
from copy import deepcopy
#from time import time #####

__all__ = ['BandAllocation']


class BandAllocation:
    def __init__(self, base_resolution=512, camera_width=24., resolution_boost=[3, 3], spin=0.9375,
                 theta_o=163, theta_d=90, n_phi=1000, max_pixels_per_run=600000, band_config=None):
        #t0 = time() #####
        if band_config is not None:
            self._band_config = OrderedDict(np.load(band_config))
        else:
            self._band_config = OrderedDict()
            self.band_config['base_resolution'] = int(base_resolution)
            assert self.band_config['base_resolution'] > 0
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

        #print(1, time() - t0) #####
        self.layer_x = [self.get_x(self.band_config['camera_width'], _) for _ in
                        self.layer_resolution]
        self.layer_xy = [np.meshgrid(_, _) for _ in self.layer_x]
        self.phi_all = np.linspace(0.001, 359.999, self.band_config['n_phi'])

        #print(2, time() - t0) #####
        if band_config is None:
            self.band_config['r_band'] = np.moveaxis(
                [[r_b(j, self.band_config['spin'], self.band_config['theta_o'],
                self.band_config['theta_d'], i) for j in self.phi_all] for i in
                range(1, self.n_layer)], 1, 2) # n_band, 2, n_phi
            for i in range(1, self.n_layer):
                #print(21, time() - t0) #####
                layer_r = jnp.sqrt(self.layer_xy[i][0]**2 + self.layer_xy[i][1]**2)
                layer_phi = jnp.arctan2(self.layer_xy[i][1], self.layer_xy[i][0]) % (2 * jnp.pi)
                layer_phi *= 180 / jnp.pi
                #print(22, time() - t0) #####
                self.band_config[f'layer_{i}'] = np.array(np.where(jnp.logical_and(
                    layer_r >= jnp.interp(layer_phi, self.phi_all,
                                          self.band_config['r_band'][i - 1, 0], period=360.),
                    layer_r <= jnp.interp(layer_phi, self.phi_all,
                                          self.band_config['r_band'][i - 1, 1], period=360.)
                )))
                #print(23, time() - t0) #####
                del layer_r, layer_phi
            self.band_config['x_all'] = self.layer_xy[0][0].flatten()
            self.band_config['y_all'] = self.layer_xy[0][1].flatten()
            for i in range(1, self.n_layer):
                self.band_config['x_all'] = np.concatenate((
                    self.band_config['x_all'],
                    self.layer_xy[i][0][tuple(self.band_config[f'layer_{i}'])]
                ))
                self.band_config['y_all'] = np.concatenate((
                    self.band_config['y_all'],
                    self.layer_xy[i][1][tuple(self.band_config[f'layer_{i}'])]
                ))
        #print(3, time() - t0) #####
        self.layer_n = np.array([self.band_config['base_resolution']**2] +
                                [self.band_config[f'layer_{i}'].shape[-1] for i in
                                 range(1, self.n_layer)])
        self.layer_cumn = np.cumsum(np.insert(self.layer_n, 0, 0))
        if band_config is None:
            self.max_pixels_per_run = int(max_pixels_per_run)
        else:
            self.max_pixels_per_run = self.band_config['max_pixels_per_run']
        #print(4, time() - t0) #####

    @property
    def band_config(self):
        return self._band_config

    @property
    def layer_resolution(self):
        return self.band_config['base_resolution'] * np.cumprod(
            np.insert(self.band_config['resolution_boost'], 0, 1))

    @property
    def image_resolution(self):
        return self.layer_resolution[-1]

    @property
    def n_layer(self):
        return self.band_config['resolution_boost'].size + 1

    @property
    def max_pixels_per_run(self):
        return self.band_config['max_pixels_per_run']

    @max_pixels_per_run.setter
    def max_pixels_per_run(self, m):
        self.band_config['max_pixels_per_run'] = int(m)
        if self.band_config['max_pixels_per_run'] > 0:
            self.band_config['n_run'] = int(np.ceil(self.band_config['x_all'].size /
                                                    self.band_config['max_pixels_per_run']))
        else:
            self.band_config['n_run'] = 1

    @staticmethod
    def get_x(camera_width, resolution):
        dx = camera_width / resolution
        return np.arange(-camera_width / 2 + dx / 2, camera_width / 2 + dx / 2, dx)

    def split_config(self):
        all_config = []
        if self.band_config['n_run'] > 1:
            for i in range(self.band_config['n_run']):
                _band_config = deepcopy(self.band_config)
                _band_config['x_all'] = _band_config['x_all'][
                    (i * _band_config['max_pixels_per_run']):
                    ((i + 1) * _band_config['max_pixels_per_run'])]
                _band_config['y_all'] = _band_config['y_all'][
                    (i * _band_config['max_pixels_per_run']):
                    ((i + 1) * _band_config['max_pixels_per_run'])]
                all_config.append(_band_config)
        else:
            all_config.append(self.band_config)
        return all_config

    def save_config(self, path, split=True):
        np.savez(path, **self.band_config)
        if split and self.band_config['n_run'] > 1:
            all_config = self.split_config()
            for i, c in enumerate(all_config):
                with open(path + f'.{i}', 'wb') as _f:
                    np.savez(_f, **c) # avoid duplicate npz extensions

    @staticmethod
    def merge_output(output_list):
        output_list = [np.load(_) for _ in output_list]
        merged = OrderedDict()
        shared_keys = ['mass_msun', 'width', 'frequency', 'adaptive_num_levels']
        for k in shared_keys:
            merged[k] = output_list[0][k]
        for k in (set(output_list[0].keys()) - set(shared_keys)):
            merged[k] = np.concatenate([_[k] for _ in output_list])
        return merged

    def read_image(self, path, target='I_nu'):
        #t0 = time() #####
        self.layer_image_raw = [np.full((_r, _r), -1.) for _r in self.layer_resolution]
        #print(1, time() - t0) #####
        result = np.nan_to_num(np.load(path)[target]) if isinstance(path, str) else path
        #print(2, time() - t0) #####
        self.layer_image_raw[0] = result[self.layer_cumn[0]:self.layer_cumn[1]].reshape(
            (self.layer_resolution[0], self.layer_resolution[0]))
        for i in range(1, self.n_layer):
            self.layer_image_raw[i][tuple(self.band_config[f'layer_{i}'])] = result[
                self.layer_cumn[i]:self.layer_cumn[i + 1]]
        #print(3, time() - t0) #####
        self.layer_image = [jnp.kron(self.layer_image_raw[i],
                                     np.ones((self.image_resolution // self.layer_resolution[i],
                                              self.image_resolution // self.layer_resolution[i])))
                            for i in range(self.n_layer)]
        #print(4, time() - t0) #####
        self.total_image = self.layer_image[-1]
        for img in self.layer_image[::-1][1:]:
            self.total_image = np.where(self.total_image >= 0., self.total_image, img)
        assert np.all(self.total_image >= 0.)
        #print(5, time() - t0) #####
        return self.total_image
