# Based on https://github.com/iAART/aart

import numpy as np
from numpy.lib.scimath import sqrt
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
from scipy.special import ellipk, ellipkinc
import warnings


__all__ = ['r_c', 'd_c', 'r_proj', 'd_proj', 'r_b']


def _r_c_fun(r, phi, spin, theta_o, target):
    lam = spin + r / spin * (r - (2 * (r**2 - 2 * r + spin**2)) / (r - 1))
    eta = r**3 / spin**2 *((4 * (r**2 - 2 * r + spin**2)) / (r - 1)**2 - r)
    alpha = -lam / np.sin(theta_o)
    beta = eta + spin**2 * np.cos(theta_o)**2 - lam**2 * np.tan(theta_o)**(-2)
    beta = np.sign(beta) * np.sqrt(np.abs(beta))
    if target == 'phi':
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
    r_p = 2 * (1 + np.cos(2 / 3 * np.arccos(spin)))
    r_0 = r_m - 0.01 * (r_p - r_m)
    r_1 = r_p + 0.01 * (r_p - r_m)
    _r_c = root_scalar(_r_c_fun, bracket=[r_0, r_1], args=(phi, spin, theta_o, 'phi'))
    if not _r_c.converged:
        warnings.warn('root_scalar may be not converged.', RuntimeWarning)
    return _r_c_fun(_r_c.root, phi, spin, theta_o, target)


def d_c(phi, spin, theta_o, deg=True):
    foo = 180 if deg else np.pi
    phi %= foo
    return r_c(phi, spin, theta_o, deg) + r_c(foo - phi, spin, theta_o, deg)


def r_proj(phi, spin, theta_o, deg=True):
    phi_p = 180 if deg else np.pi
    assert 0 <= phi <= phi_p
    # phi %= phi_p
    def foo(p, phi):
        x, y = r_c(p, spin, theta_o, deg, 'xy')
        phi = phi * np.pi / 180 if deg else phi
        return -(x * np.cos(phi) + y * np.sin(phi))
    o_c = minimize_scalar(foo, bracket=[0, phi_p], bounds=[0, phi_p], method='bounded', args=(phi,))
    if not o_c.success:
        warnings.warn('minimize_scalar may be not converged.', RuntimeWarning)
    return r_c(o_c.x, spin, theta_o, deg, 'r')


def d_proj(phi, spin, theta_o, deg=True):
    foo = 180 if deg else np.pi
    phi %= foo
    return r_proj(phi, spin, theta_o, deg) + r_proj(foo - phi, spin, theta_o, deg)


def cbrt(x):
    if x.imag==0:
        return np.cbrt(x)
    else:
        return x**(1/3)


def _r_b_fun(x, _r_c, phi, spin, theta_o, theta_d, n):
    r = x * _r_c
    m = n + np.heaviside(np.sin(phi) * np.cos(theta_o), 0)
    alpha = r * np.cos(phi)
    beta = r * np.sin(phi)
    lam = -alpha * np.sin(theta_o)
    eta = (alpha**2 - spin**2) * np.cos(theta_o)**2 + beta**2
    nu_theta = np.sign(np.sin(phi))

    # radial roots and integrals
    AA = spin**2 - eta - lam**2
    BB = 2. * (eta + (lam - spin)**2)
    CC = -spin**2 * eta
    P = -(AA**2 / 12.) - CC
    Q = -(AA / 3.) * ((AA / 6.)**2 - CC) - BB**2 / 8.
    delta_3 = -4. * P**3 - 27. * Q**2
    xi_0 = np.real(cbrt(-(Q / 2.) + sqrt(-(delta_3 / 108.))) + cbrt(-(Q / 2.) -
                   sqrt(-(delta_3 / 108.))) - AA / 3)
    z = sqrt(xi_0 / 2)
    r_1 = -z - sqrt(-(AA / 2) - z**2 + BB / (4 * z))
    r_2 = -z + sqrt(-(AA / 2) - z**2 + BB / (4 * z))
    r_3 = z - sqrt(-(AA / 2) - z**2 - BB / (4 * z))
    r_4 = z + sqrt(-(AA / 2) - z**2 - BB / (4 * z))
    delta_theta = 0.5 * (1. - (eta + lam**2) / spin**2)

    # roots of angular potential
    u_p = delta_theta + sqrt(delta_theta**2 + eta / spin**2)
    u_m = delta_theta - sqrt(delta_theta**2 + eta / spin**2)
    r_21 = r_2 - r_1
    r_31 = r_3 - r_1
    r_32 = r_3 - r_2
    r_41 = r_4 - r_1
    r_42 = r_4 - r_2
    r_43 = r_4 - r_3

    # outer and inner horizons
    r_p = 1 + sqrt(1 - spin**2)
    r_m = 1 - sqrt(1 - spin**2)

    a_1 = sqrt(-(r_43**2 / 4.))
    b_1 = (r_3 + r_4) / 2.

    # elliptic parameter
    k = (r_32 * r_41) / (r_31 * r_42)
    A = np.real(sqrt(a_1**2 + (b_1 - r_2)**2))
    B = np.real(sqrt(a_1**2 + (b_1 - r_1)**2))
    k_3 = np.real(((A + B)**2 - r_21**2) / (4. * A * B))

    # Eqs (20 P2)
    g_theta = 1 / (sqrt(-u_m) * spin) * (
        2. * m * ellipk(u_p / u_m) - nu_theta * ellipkinc(np.arcsin(np.cos(theta_o) / sqrt(u_p)),
        u_p / u_m) + nu_theta * (-1)**m * ellipkinc(np.arcsin(np.cos(theta_d) / sqrt(u_p)),
        u_p / u_m))

    if x > 1:
        # Eqs (A10 P2)
        q_1 = 4 / sqrt(r_31 * r_42) * ellipkinc(np.arcsin(sqrt(r_31 / r_41)), k)
        return q_1 - g_theta
    else:
        if k_3 < 1:
            # Eqs (A11 P2)
            q_2 = 1 / sqrt(A * B) * (ellipkinc(np.arccos((A - B) / (A + B)), k_3) -
                                     ellipkinc(np.arccos((A * (r_p - r_1) - B * (r_p - r_2)) /
                                                         (A * (r_p - r_1) + B * (r_p - r_2))), k_3))
        else:
            q_2 = np.nan
        return q_2 - g_theta


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
    r_i = root_scalar(_r_b_fun, bracket=[0.3, 0.999], args=(_r_c, phi, spin, theta_o, theta_d, n))
    assert r_i.converged
    r_o = root_scalar(_r_b_fun, bracket=[1.001, 3], args=(_r_c, phi, spin, theta_o, theta_d, n))
    assert r_o.converged
    return r_i.root * _r_c, r_o.root * _r_c
