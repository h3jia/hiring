import numpy as np
from collections import OrderedDict, namedtuple
import pickle
from .lensing_bands import *
from ._allocation import _set_mask, _merge_layers, _zoom_layers, _rotate_image

__all__ = ['LayerConfig', 'PixelConfig', 'get_lensing_band_layer', 'get_square_layer',
           'merge_configs', 'save_configs', 'load_layers', 'load_pixels', 'load_image',
           'rotate_image', 'proj_image']


LayerConfig = namedtuple('LayerConfig', ['padding_resolution', 'central_resolution', 'ij'])


PixelConfig = namedtuple('PixelConfig', ['camera_width', 'x_all', 'y_all'])


def get_lensing_band_layer(camera_width, layer_resolution, n, outer_width=None, spin=0.9375,
                           theta_o=163., theta_d=90., n_phi=1000, f_exp=3.):
    camera_width = float(camera_width)
    assert camera_width > 0
    layer_resolution = int(layer_resolution)
    assert layer_resolution > 0
    if outer_width is not None:
        outer_width = float(outer_width)
        assert 0 < outer_width <= camera_width
        central_width = outer_width
    else:
        central_width = camera_width
    central_resolution = int(layer_resolution * central_width / camera_width)
    padding_resolution = (layer_resolution - central_resolution) // 2
    padding_resolution = int(max(0, np.floor(padding_resolution - f_exp)))
    central_resolution = layer_resolution - 2 * padding_resolution
    central_width = camera_width * central_resolution / layer_resolution

    dx = central_width / central_resolution
    _x_1d_all = np.arange(-central_width / 2 + dx / 2, central_width / 2 + dx / 2.01, dx)
    _x_all, _y_all = np.meshgrid(_x_1d_all, _x_1d_all) # shape: (n_y, n_x)

    _mask = np.full_like(_x_all, 0, dtype=np.int8)
    phi_all = np.linspace(0., 360., n_phi)
    r_all = np.array([r_b(p, spin, theta_o, theta_d, n) for p in phi_all])
    r_in_all = r_all[:, 0].copy() - f_exp * dx
    r_out_all = r_all[:, 1].copy() + f_exp * dx
    _set_mask(_mask, _x_all, _y_all, r_in_all, r_out_all, _x_all.shape[0], n_phi, 1)
    mask = np.where(_mask)

    x_all = _x_all[mask]
    y_all = _y_all[mask]
    layer_config = LayerConfig(padding_resolution=padding_resolution,
                               central_resolution=central_resolution, ij=mask)
    return layer_config, camera_width, x_all, y_all


def get_square_layer(camera_width, layer_resolution, inner_width=None, outer_width=None,
                     remove_lensing_band=True, spin=0.9375, theta_o=163., theta_d=90.,
                     n_phi=1000, f_exp=3.):
    camera_width = float(camera_width)
    assert camera_width > 0
    layer_resolution = int(layer_resolution)
    assert layer_resolution > 0
    if outer_width is not None:
        outer_width = float(outer_width)
        assert 0 < outer_width <= camera_width
        central_width = outer_width
    else:
        central_width = camera_width
    central_resolution = int(layer_resolution * central_width / camera_width)
    padding_resolution = (layer_resolution - central_resolution) // 2
    padding_resolution = int(max(0, np.floor(padding_resolution - f_exp)))
    central_resolution = layer_resolution - 2 * padding_resolution
    central_width = camera_width * central_resolution / layer_resolution
    if central_resolution % 2:
        raise NotImplementedError(
            f'central_resolution should be even, instead of {central_resolution}')

    dx = central_width / central_resolution
    _x_1d_all = np.arange(-central_width / 2 + dx / 2, central_width / 2 + dx / 2.01, dx)
    _x_all, _y_all = np.meshgrid(_x_1d_all, _x_1d_all)

    _mask = np.full_like(_x_all, 1, dtype=np.int8)
    if inner_width is not None:
        inner_width = float(inner_width)
        assert 0 < inner_width < central_width
        inner_resolution = int(
            max(0, np.floor(central_resolution * inner_width / central_width - 2 * f_exp)))
        band_resolution = (central_resolution - inner_resolution) // 2
        inner_resolution = central_resolution - 2 * band_resolution
        _mask[band_resolution:(-band_resolution), band_resolution:(-band_resolution)] = 0
    if remove_lensing_band:
        phi_all = np.linspace(0., 360., n_phi)
        r_all = np.array([r_b(p, spin, theta_o, theta_d, 1) for p in phi_all])
        r_in_all = r_all[:, 0].copy() + f_exp * dx
        r_out_all = r_all[:, 1].copy() - f_exp * dx
        _set_mask(_mask, _x_all, _y_all, r_in_all, r_out_all, _x_all.shape[0], n_phi, 0)
    mask = np.where(_mask)

    x_all = _x_all[mask]
    y_all = _y_all[mask]
    layer_config = LayerConfig(padding_resolution=padding_resolution,
                               central_resolution=central_resolution, ij=mask)
    return layer_config, camera_width, x_all, y_all


def merge_configs(layers):
    layer_configs = [l[0] for l in layers]
    pixel_configs = PixelConfig(layers[0][1], np.concatenate([l[2] for l in layers]),
                                np.concatenate([l[3] for l in layers]))
    return layer_configs, pixel_configs


def save_configs(base_name, layer_configs, pixel_configs, max_pixels_per_run=520000):
    with open(base_name + '.layer', 'wb') as _f:
        pickle.dump(layer_configs, _f)
        # np.savez(_f, layer_configs=layer_configs)
    total_pixels = pixel_configs.x_all.shape[0]
    i = 0
    while i * max_pixels_per_run < total_pixels:
        with open(base_name + f'.pixel.{i}', 'wb') as _f:
            np.savez(
                _f, camera_width=pixel_configs.camera_width,
                x_all=pixel_configs.x_all[(i * max_pixels_per_run):((i + 1) * max_pixels_per_run)],
                y_all=pixel_configs.y_all[(i * max_pixels_per_run):((i + 1) * max_pixels_per_run)]
            )
        i += 1


def load_layers(base_name):
    # layer_configs = np.load(base_name + '.layer')['layer_configs']
    with open(base_name + '.layer', 'rb') as _f:
        layer_configs = pickle.load(_f)
    return layer_configs


def load_pixels(base_name):
    raise NotImplementedError


def load_image(file, layer_configs, target='I_nu'):
    if isinstance(file, np.ndarray):
        pass
    elif isinstance(file, str):
        file = np.load(file)[target]
    else:
        file = np.concatenate([np.load(f)[target] for f in file])
    file = np.nan_to_num(file).astype(np.float32)
    i_cum = 0
    # layer_img_all = []
    central_img_all = []
    layer_res_all = []
    for l in layer_configs:
        central_img_now = np.full((l.central_resolution, l.central_resolution), np.nan,
                                  dtype=np.float32)
        central_img_now[l.ij] = file[i_cum:(i_cum + l.ij[0].size)]
        i_cum += l.ij[0].size
        layer_res_now = l.central_resolution + 2 * l.padding_resolution
        # layer_img_now = np.full((layer_res_now, layer_res_now), np.nan, dtype=np.float32)
        # layer_img_now[
        #     l.padding_resolution:(l.padding_resolution + l.central_resolution),
        #     l.padding_resolution:(l.padding_resolution + l.central_resolution)
        # ] = central_img
        central_img_all.append(central_img_now)
        layer_res_all.append(layer_res_now)
        # layer_img_all.append(layer_img_now)
    max_res = np.max(layer_res_all)
    final_img = np.full((max_res, max_res), np.nan, dtype=np.float32)
    for i_l in range(len(layer_res_all)):
        if max_res % layer_res_all[i_l]:
            raise NotImplementedError
        n_zoom = max_res // layer_res_all[i_l]
        if n_zoom > 1:
            if not n_zoom % 2:
                raise NotImplementedError
            layer_img_now = np.full((central_img_all[i_l].shape[0] * n_zoom,
                                     central_img_all[i_l].shape[1] * n_zoom), np.nan,
                                    dtype=np.float32)
            _zoom_layers(central_img_all[i_l], layer_img_now, central_img_all[i_l].shape[0],
                         n_zoom)
        else:
            layer_img_now = central_img_all[i_l]
        _merge_layers(final_img, layer_img_now, layer_img_now.shape[0],
                      layer_configs[i_l].padding_resolution * n_zoom)
    return final_img


def rotate_image(image, angle=0., circle_mask=True):
    image = np.ascontiguousarray(image, dtype=np.float32)
    assert image.ndim == 2
    if not (image.shape[0] == image.shape[1]):
        raise NotImplementedError

    output = np.full_like(image, np.nan)
    _rotate_image(image, output, image.shape[0], angle, int(circle_mask))
    return output


def proj_image(image, angle=0., circle_mask=True):
    image = rotate_image(image, angle, circle_mask)
    return np.mean(image, axis=0)
