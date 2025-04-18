# Based on https://github.com/iAART/aart

import numpy as np
from numpy.lib.scimath import sqrt
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
from scipy.special import ellipk, ellipkinc, ellipj
import warnings


__all__ = ['r_c', 'd_c', 'r_proj', 'd_proj', 'r_b', 'r_s_from_r', 'r_from_r_s',
           'r_proj_from_r_s', 'd_proj_from_r_s']


import mpmath as mp
mp.dps = 50  # Adjust precision (digits) as needed


def _r_c_fun(r, phi, spin, theta_o, target):
    # the critical curve should correspond to the solution to 1910.12873 Eq 37
    # will solve that r for phi
    lam = spin + r / spin * (r - 2 * (r**2 - 2 * r + spin**2) / (r - 1))     # 1910.12873 Eq 38
    eta = r**3 / spin**2 * (4 * (r**2 - 2 * r + spin**2) / (r - 1)**2 - r)   # 1910.12873 Eq 39
    alpha = -lam / np.sin(theta_o)
    beta = eta + spin**2 * np.cos(theta_o)**2 - lam**2 * np.tan(theta_o)**(-2)
    beta = np.sign(beta) * np.sqrt(np.abs(beta))                             # 1910.12873 Eq 55
    # actually beta should be positive here, we extend it to negative values for the optimizer
    if target == 'phi':
        # from -90 to 270, although phi should really range from 0 to 180
        return (np.arctan2(beta, alpha) * 180 / np.pi + 90) % 360 - 90 - phi * 180 / np.pi
    elif target == 'r':
        return np.sqrt(alpha**2 + beta**2)
    elif target == 'xy':
        return alpha, beta


def r_c(phi, spin, theta_o, deg=True, target='r'):
    if deg:
        phi = phi * np.pi / 180
        theta_o = theta_o * np.pi / 180
    if np.pi < phi <= 2 * np.pi:
        return r_c(2 * np.pi - phi, spin, theta_o, False, target)
    assert 0 <= phi <= np.pi
    assert 0 <= theta_o <= np.pi
    theta_o = np.clip(theta_o, 1e-5, np.pi - 1e-5)
    r_m = 2 * (1 + np.cos(2 / 3 * np.arccos(-spin)))
    r_p = 2 * (1 + np.cos(2 / 3 * np.arccos(spin)))                          # 1910.12873 Eq 40
    r_0 = r_m - 0.01 * (r_p - r_m)
    r_1 = r_p + 0.01 * (r_p - r_m)
    _r_c = root_scalar(_r_c_fun, bracket=[r_0, r_1], args=(phi, spin, theta_o, 'phi'))
    if not _r_c.converged:
        warnings.warn('root_scalar may be not converged.', RuntimeWarning)
    return _r_c_fun(_r_c.root, phi, spin, theta_o, target)


def d_c(phi, spin, theta_o, deg=True):
    phi_p = 180 if deg else np.pi
    phi %= phi_p
    return r_c(phi, spin, theta_o, deg) + r_c(phi_p - phi, spin, theta_o, deg)


def r_proj(phi, spin, theta_o, deg=True):
    phi_p = 180 if deg else np.pi
    phi %= (2 * phi_p)
    if phi > phi_p:
        phi = 2 * phi_p - phi
    assert 0 <= phi <= phi_p
    def foo(p, phi):
        x, y = r_c(p, spin, theta_o, deg, 'xy')
        phi = phi * np.pi / 180 if deg else phi
        return -(x * np.cos(phi) + y * np.sin(phi))
    o_c = minimize_scalar(foo, bracket=[0, phi_p], bounds=[0, phi_p], method='bounded', args=(phi,))
    if not o_c.success:
        warnings.warn('minimize_scalar may be not converged.', RuntimeWarning)
    # return r_c(o_c.x, spin, theta_o, deg, 'r')
    return -foo(o_c.x, phi)


def d_proj(phi, spin, theta_o, deg=True):
    phi_p = 180 if deg else np.pi
    phi %= phi_p
    return r_proj(phi, spin, theta_o, deg) + r_proj(phi_p - phi, spin, theta_o, deg)


def cbrt(x):
    if x.imag==0:
        return np.cbrt(x)
    else:
        return x**(1/3)


def _r_b_fun(x, _r_c, phi, spin, theta_o, theta_d, n, target='solve'):
    r = x * _r_c
    # m: theta turning, n: z turning
    m = n + np.heaviside(np.sin(phi) * np.cos(theta_o), 0)                  # 1910.12873 Eq 82
    alpha = r * np.cos(phi)
    beta = r * np.sin(phi)
    lam = -alpha * np.sin(theta_o)                                          # 1910.12873 Eq 58
    eta = (alpha**2 - spin**2) * np.cos(theta_o)**2 + beta**2               # 1910.12873 Eq 59
    nu_theta = np.sign(np.sin(phi))

    # radial roots and integrals
    AA = spin**2 - eta - lam**2                                             # 1910.12873 Eq A1
    BB = 2. * (eta + (lam - spin)**2)                                       # 1910.12873 Eq A2
    CC = -spin**2 * eta                                                     # 1910.12873 Eq A3
    P = -(AA**2 / 12.) - CC                                                 # 1910.12873 Eq A4
    Q = -(AA / 3.) * ((AA / 6.)**2 - CC) - BB**2 / 8.                       # 1910.12873 Eq A5
    delta_3 = -4. * P**3 - 27. * Q**2
    xi_0 = np.real(cbrt(-(Q / 2.) + sqrt(-(delta_3 / 108.))) +
                   cbrt(-(Q / 2.) - sqrt(-(delta_3 / 108.))) - AA / 3)
    z = sqrt(xi_0 / 2)                                                      # 1910.12873 Eq A6
    r_1 = -z - sqrt(-(AA / 2) - z**2 + BB / (4 * z))                        # 1910.12873 Eq A8a
    r_2 = -z + sqrt(-(AA / 2) - z**2 + BB / (4 * z))                        # 1910.12873 Eq A8b
    r_3 = z - sqrt(-(AA / 2) - z**2 - BB / (4 * z))                         # 1910.12873 Eq A8c
    r_4 = z + sqrt(-(AA / 2) - z**2 - BB / (4 * z))                         # 1910.12873 Eq A8d
    delta_theta = 0.5 * (1. - (eta + lam**2) / spin**2)                     # 1910.12873 Eq A11

    # roots of angular potential
    u_p = delta_theta + sqrt(delta_theta**2 + eta / spin**2)                # 1910.12873 Eq A11
    u_m = delta_theta - sqrt(delta_theta**2 + eta / spin**2)                # 1910.12873 Eq A11
    r_21 = r_2 - r_1
    r_31 = r_3 - r_1
    r_32 = r_3 - r_2
    r_41 = r_4 - r_1
    r_42 = r_4 - r_2
    r_43 = r_4 - r_3

    # outer and inner horizons
    r_p = 1 + sqrt(1 - spin**2)
    r_m = 1 - sqrt(1 - spin**2)

    # elliptic parameter
    k = (r_32 * r_41) / (r_31 * r_42)
    _sin_o = np.cos(theta_o) / sqrt(u_p)
    _sin_d = np.cos(theta_d) / sqrt(u_p)
    if -1.1 < _sin_o < 1.1:
        if not (-1. <= _sin_o <= 1.):
            warnings.warn(f'_sin_o={_sin_o} is slightly out of range, probably due to numerical '
                          f'issues.', RuntimeWarning)
            _sin_o = np.clip(_sin_o, -0.99999, 0.99999)
    else:
        return np.nan
    if -1.1 < _sin_d < 1.1:
        if not (-1. <= _sin_d <= 1.):
            warnings.warn(f'_sin_d={_sin_d} is slightly out of range, probably due to numerical '
                          f'issues.', RuntimeWarning)
            _sin_d = np.clip(_sin_d, -0.99999, 0.99999)
    else:
        return np.nan
    g_theta = 1 / (sqrt(-u_m) * spin) * (
        2. * m * ellipk(u_p / u_m) -
        nu_theta * ellipkinc(np.arcsin(_sin_o), u_p / u_m) +
        nu_theta * (-1)**m * ellipkinc(np.arcsin(_sin_d), u_p / u_m)
    )                                                                       # 1910.12873 Eq 20
    # print(np.cos(theta_o), np.cos(theta_d), sqrt(u_p))
    # g_theta = 1 / (sqrt(-u_m) * spin) * (
    #     2. * m * ellipk(u_p / u_m) -
    #     nu_theta * mp.ellipf(mp.asin(np.cos(theta_o) / sqrt(u_p)), u_p / u_m) +
    #     nu_theta * (-1)**m * mp.ellipf(mp.asin(np.cos(theta_d) / sqrt(u_p)), u_p / u_m)
    # )
    # print(mp.asin(np.cos(theta_d) / sqrt(u_p)), mp.asin(0.1), np.arcsin(0.1), np.cos(theta_d), sqrt(u_p))

    if target == 'solve':
        a_1 = sqrt(-(r_43**2 / 4.))
        b_1 = (r_3 + r_4) / 2.
        A = np.real(sqrt(a_1**2 + (b_1 - r_2)**2))
        B = np.real(sqrt(a_1**2 + (b_1 - r_1)**2))
        k_3 = np.real(((A + B)**2 - r_21**2) / (4. * A * B))                # 1910.12873 Eq A12

        if x > 1:
            # Eqs (A10 P2)
            q_1 = 4 / sqrt(r_31 * r_42) * ellipkinc(np.arcsin(sqrt(r_31 / r_41)), k)
            return q_1 - g_theta
        else:
            if -0.1 < k_3 < 1.1:
                # Eqs (A11 P2)
                if not (0. <= k_3 <= 1.):
                    warnings.warn(f'k_3={k_3} is slightly out of range, probably due to numerical '
                                  f'issues.', RuntimeWarning)
                    k_3 = np.clip(k_3, 0.00001, 0.99999)
                q_2 = 1 / sqrt(A * B) * (
                    ellipkinc(np.arccos((A - B) / (A + B)), k_3) -
                    ellipkinc(np.arccos((A * (r_p - r_1) - B * (r_p - r_2)) /
                                        (A * (r_p - r_1) + B * (r_p - r_2))), k_3)
                )
            else:
                q_2 = np.nan
            # print(k_3)
            return q_2 - g_theta

    elif target == 'r_s':
        sn2 = mp.ellipfun('sn', 0.5 * sqrt(r_31 * r_42) * g_theta -
                          mp.ellipf(np.arcsin(sqrt(r_31 / r_41)), k), k)**2
        return float(np.real((r_4 * r_31 - r_3 * r_41 * sn2) / (r_31 - r_41 * sn2)))

    else:
        raise ValueError


def r_b(phi, spin, theta_o, theta_d, n, deg=True):
    if deg:
        phi = phi * np.pi / 180
        theta_o = theta_o * np.pi / 180
        theta_d = theta_d * np.pi / 180
    phi %= (2 * np.pi)
    if abs(phi - 0.) < 2e-5:
        phi = 2e-5
    elif abs(phi - np.pi) < 2e-5:
        phi = np.pi - 2e-5
    elif abs(phi - 2 * np.pi) < 2e-5:
        phi = 2 * np.pi - 2e-5
    _r_c = r_c(phi, spin, theta_o, deg=False)

    r_i_0 = 0.8
    r_i_1 = 0.999
    _r_b_fun_0 = _r_b_fun(r_i_0, _r_c, phi, spin, theta_o, theta_d, n, 'solve')
    _r_b_fun_1 = _r_b_fun(r_i_1, _r_c, phi, spin, theta_o, theta_d, n, 'solve')
    if not (np.isfinite(_r_b_fun_0) and np.isfinite(_r_b_fun_1)):
        raise RuntimeError(f'solving r_i failed at {(_r_c, phi, spin, theta_o, theta_d, n)}.')
    while _r_b_fun_0 * _r_b_fun_1 > 0.:
        r_i_0 -= 0.1
        _r_b_fun_0 = _r_b_fun(r_i_0, _r_c, phi, spin, theta_o, theta_d, n, 'solve')
        if not  (r_i_0 > 0. and np.isfinite(_r_b_fun_0)):
            raise RuntimeError(f'solving r_i failed at {(_r_c, phi, spin, theta_o, theta_d, n)}.')
    r_i = root_scalar(_r_b_fun, bracket=[r_i_0, r_i_1],
                      args=(_r_c, phi, spin, theta_o, theta_d, n, 'solve'), maxiter=100)
    if not r_i.converged:
        raise RuntimeError(f'the solution r_i={r_i.root} at '
                           f'{(_r_c, phi, spin, theta_o, theta_d, n)} is not converged.')

    r_o_0 = 1.001
    r_o_1 = 5.
    _r_b_fun_0 = _r_b_fun(r_o_0, _r_c, phi, spin, theta_o, theta_d, n, 'solve')
    _r_b_fun_1 = _r_b_fun(r_o_1, _r_c, phi, spin, theta_o, theta_d, n, 'solve')
    if not (np.isfinite(_r_b_fun_0) and np.isfinite(_r_b_fun_1)):
        raise RuntimeError(f'solving r_o failed at {(_r_c, phi, spin, theta_o, theta_d, n)}.')
    while _r_b_fun_0 * _r_b_fun_1 > 0.:
        r_o_1 += 5.
        _r_b_fun_1 = _r_b_fun(r_o_1, _r_c, phi, spin, theta_o, theta_d, n, 'solve')
        if not (r_o_1 < 500. and np.isfinite(_r_b_fun_1)):
            raise RuntimeError(f'solving r_o failed at {(_r_c, phi, spin, theta_o, theta_d, n)}.')
    r_o = root_scalar(_r_b_fun, bracket=[r_o_0, r_o_1],
                      args=(_r_c, phi, spin, theta_o, theta_d, n, 'solve'), maxiter=100)
    if not r_o.converged:
        raise RuntimeError(f'the solution r_o={r_o.root} at '
                           f'{(_r_c, phi, spin, theta_o, theta_d, n)} is not converged.')

    return r_i.root * _r_c, r_o.root * _r_c


def r_s_from_r(r, phi, spin, theta_o, theta_d, n, deg=True):
    if deg:
        phi = phi * np.pi / 180
        theta_o = theta_o * np.pi / 180
        theta_d = theta_d * np.pi / 180
    phi %= (2 * np.pi)
    if abs(phi - 0.) < 2e-5:
        phi = 2e-5
    elif abs(phi - np.pi) < 2e-5:
        phi = np.pi - 2e-5
    elif abs(phi - 2 * np.pi) < 2e-5:
        phi = 2 * np.pi - 2e-5
    return _r_b_fun(r, 1., phi, spin, theta_o, theta_d, n, target='r_s')


def r_from_r_s(r_s, phi, spin, theta_o, theta_d, n, deg=True):
    # TODO: check r_s range
    if n >= 1:
        r_i, r_o = r_b(phi, spin, theta_o, theta_d, n, deg)
        _r_s = root_scalar(lambda x: r_s_from_r(x, phi, spin, theta_o, theta_d, n, deg) - r_s,
                           bracket=[r_i + 0.001, r_o - 0.001])
    elif n == 0:
        _r_s = root_scalar(lambda x: r_s_from_r(x, phi, spin, theta_o, theta_d, n, deg) - r_s,
                           bracket=[1.5, 15.])
    assert _r_s.converged
    return _r_s.root


def r_proj_from_r_s(r_s, phi, spin, theta_o, theta_d, n, deg=True):
    # TODO: merge with r_proj
    phi_p = 180 if deg else np.pi
    phi %= (2 * phi_p)
    # if phi > phi_p:
    #     phi = 2 * phi_p - phi
    # assert 0 <= phi <= phi_p
    def foo(p, phi):
        r = r_from_r_s(r_s, p, spin, theta_o, theta_d, n, deg)
        p = p * np.pi / 180 if deg else p
        phi = phi * np.pi / 180 if deg else phi
        x, y = r * np.cos(p), r * np.sin(p)
        return -(x * np.cos(phi) + y * np.sin(phi))
    o_c = minimize_scalar(foo, bracket=[phi - 0.5 * phi_p, phi + 0.5 * phi_p],
                          bounds=[phi - 0.5 * phi_p, phi + 0.5 * phi_p], method='bounded',
                          args=(phi,))
    if not o_c.success:
        warnings.warn('minimize_scalar may be not converged.', RuntimeWarning)
    # return r_c(o_c.x, spin, theta_o, deg, 'r')
    return -foo(o_c.x, phi)


def d_proj_from_r_s(r_s, phi, spin, theta_o, theta_d, n, deg=True):
    phi_p = 180 if deg else np.pi
    return (r_proj_from_r_s(r_s, phi, spin, theta_o, theta_d, n, deg) +
            r_proj_from_r_s(r_s, phi + phi_p, spin, theta_o, theta_d, n, deg))
