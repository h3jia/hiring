import numpy as np
from scipy.integrate import simpson, trapezoid
from scipy.special import factorial
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import PchipInterpolator

__all__ = ['linear_phi', 'linear_q', 'slice', 'phi_edge', 'Irrd', 'r_col_from_Irrd_one',
           'M_from_Irrd_one', 'a_from_M', 'a_from_Irrd_one', 'a_from_Irrd', 'A_phi_from_a',
           'r_col_from_a', 'Vu']


def linear_phi(n_phi=60):
    return np.linspace(0., 2. * np.pi, n_phi + 1, endpoint=True)

def linear_q(n_q_base=101, dq_edge_boost=(10, 5, 2), endpoint=False, clip=1e-6):
    q_base = np.linspace(0., 1., n_q_base, endpoint=True)
    q_all = []
    if endpoint:
        q_all.append(q_base[:1])
    for i in range(len(dq_edge_boost)):
        q_all.append(np.linspace(q_base[i], q_base[i + 1], dq_edge_boost[i] + 1)[1:])
    q_all.append(q_base[(len(dq_edge_boost) + 1):-len(dq_edge_boost)])
    for i in range(len(dq_edge_boost))[::-1]:
        q_all.append(np.linspace(q_base[-i - 2], q_base[-i - 1], dq_edge_boost[i] + 1)[1:])
    q_all = np.concatenate(q_all) if endpoint else np.concatenate(q_all)[:-1]
    if clip is not None and 0 < clip < 0.5:
        q_all = np.clip(q_all, clip, 1. - clip)
    return q_all

def slice(img, x, y, phi=None, q=None, r_in=None, r_out=None, dr=0.005, k_interp=1,
          origin=(0., 0.)):
    img = np.asarray(img)
    x = np.asarray(x)
    y = np.asarray(y)
    origin = np.asarray(origin)
    assert img.ndim == 2 and x.ndim == 1 and y.ndim == 1
    assert img.shape[0] == x.shape[0] and img.shape[1] == y.shape[0]
    phi = linear_phi() if phi is None else np.asarray(phi)
    q = linear_q() if q is None else np.asarray(q)
    r_in = 0.01 * np.max(np.abs(x)) if r_in is None else float(r_in)
    r_out = 0.99 * np.max(np.abs(x)) if r_out is None else float(r_out)

    img_interp = RectBivariateSpline(x, y, img, kx=k_interp, ky=k_interp)
    r_all = []
    I_all = []

    r_grid = np.arange(r_in, r_out, dr)
    for phi_now in phi:
        x_grid = origin[0] + r_grid * np.cos(phi_now)
        y_grid = origin[1] + r_grid * np.sin(phi_now)
        I_grid = img_interp.ev(x_grid, y_grid)
        cum_I_grid = np.cumsum(I_grid)
        cum_I_grid /= cum_I_grid[-1]
        cum_I_interp = PchipInterpolator(r_grid, cum_I_grid)
        I_interp = PchipInterpolator(r_grid, I_grid)
        r_now = []
        I_now = []
        for q_now in q:
            r_now.append(cum_I_interp.solve(y=q_now)[0])
            I_now.append(I_interp(r_now[-1]))
        r_all.append(r_now)
        I_all.append(I_now)
    return np.array([r_all, I_all]) # (r or I, # of phi, # of q)

def _choose_root(roots, phi_obs):
    delta = (roots - phi_obs) % (2 * np.pi)
    delta = np.where(delta > np.pi, delta - 2 * np.pi, delta)
    return np.array([roots[np.argmin(np.abs(delta))],
                     roots[np.argmax(np.abs(delta))]]) % (2 * np.pi)

def _check_slice(slice):
    slice = np.asarray(slice)
    if slice.ndim == 4:
        pass
    elif slice.ndim == 3:
        slice = slice[None]
    else:
        raise ValueError
    return slice

def phi_edge(slice, phi_obs, phi=None):
    # slice: (# of img, r or I, # of phi, # of q)
    slice = _check_slice(slice)
    phi = linear_phi() if phi is None else np.asarray(phi)
    proj = (slice[:, 0] * np.cos(phi)[None, :, None] * np.cos(phi_obs) +
            slice[:, 0] * np.sin(phi)[None, :, None] * np.sin(phi_obs))
    proj[:, -1] = proj[:, 0] # avoid numerical issues
    phi_edge_interp = CubicSpline(phi, proj, axis=1, bc_type='periodic').derivative(1).solve()
    # (# of img, # of q, near or far)
    return np.array([[_choose_root(__, phi_obs) for __ in _] for _ in phi_edge_interp])

def Irrd(slice, phi_edge, phi_obs, phi=None, q=None, periodic=True):
    slice = _check_slice(slice) # (# of img, r or I, # of phi, # of q)
    phi_edge = np.asarray(phi_edge) # (# of img, # of q, near or far)
    phi = linear_phi() if phi is None else np.asarray(phi)
    q = linear_q() if q is None else np.asarray(q)

    if periodic:
        r_interp = [[CubicSpline(phi, slice[i, 0, :, j], bc_type='periodic') for j in
                     range(slice.shape[3])] for i in range(slice.shape[0])]
        I_interp = [[CubicSpline(phi, slice[i, 1, :, j], bc_type='periodic') for j in
                     range(slice.shape[3])] for i in range(slice.shape[0])]
        rI_interp = [r_interp, I_interp] # (r or I; # of img; # of q)

        r_edge = np.array([[rI_interp[0][i][j](phi_edge[i][j]) for j in range(slice.shape[3])]
                           for i in range(slice.shape[0])])
        dr_dphi_edge = np.array([[rI_interp[0][i][j](phi_edge[i][j], 1) for j in
                                  range(slice.shape[3])] for i in range(slice.shape[0])])
        dr2_dphi2_edge = np.array([[rI_interp[0][i][j](phi_edge[i][j], 2) for j in
                                    range(slice.shape[3])] for i in range(slice.shape[0])])

        r_curv = (r_edge**2 + dr_dphi_edge**2)**1.5 / np.abs(r_edge**2 + 2. * dr_dphi_edge**2 -
                                                             r_edge * dr2_dphi2_edge)
        I_edge = np.array([[rI_interp[1][i][j](phi_edge[i][j]) for j in range(slice.shape[3])]
                           for i in range(slice.shape[0])])
        r_proj = np.abs(r_edge * np.cos(phi_edge) * np.cos(phi_obs) +
                        r_edge * np.sin(phi_edge) * np.sin(phi_obs))

        r_q_edge = np.moveaxis([[[rI_interp[0][i][k](phi_edge[i][j]) for k in range(slice.shape[3])]
                                 for j in range(slice.shape[3])] for i in range(slice.shape[0])],
                               2, 3)
        # (# of img, # of q, near or far, # of q)
        dr_dq = np.array([[[CubicSpline(q, r_q_edge[i, j, k])(q[j], 1) for k in range(2)] for j in
                           range(r_q_edge.shape[1])] for i in range(r_q_edge.shape[0])])
        dr_dq_cosdphi = dr_dq * np.cos(phi_edge - np.array([phi_obs, phi_obs + np.pi]))

        # (# of img, 4, # of q, near or far)
        return np.moveaxis((I_edge, r_proj, r_curv, dr_dq_cosdphi), 0, 1)

    else:
        r_interp = [RectBivariateSpline(phi, q, _, bbox=[0., 2. * np.pi, 0., 1.]) for _ in
                    slice[:, 0]]
        I_interp = [RectBivariateSpline(phi, q, _, bbox=[0., 2. * np.pi, 0., 1.]) for _ in
                    slice[:, 1]]
        rI_interp = [r_interp, I_interp]

        r_edge = np.array([rI_interp[0][i].ev(phi_edge[i], q[:, None]) for i in
                           range(slice.shape[0])])
        dr_dphi_edge = np.array([rI_interp[0][i].ev(phi_edge[i], q[:, None], dx=1) for i in
                                 range(slice.shape[0])])
        dr2_dphi2_edge = np.array([rI_interp[0][i].ev(phi_edge[i], q[:, None], dx=2) for i in
                                   range(slice.shape[0])])

        r_curv = (r_edge**2 + dr_dphi_edge**2)**1.5 / np.abs(r_edge**2 + 2. * dr_dphi_edge**2 -
                                                             r_edge * dr2_dphi2_edge)
        I_edge = np.array([rI_interp[1][i].ev(phi_edge[i], q[:, None]) for i in
                           range(slice.shape[0])])
        r_proj = np.abs(r_edge * np.cos(phi_edge) * np.cos(phi_obs) +
                        r_edge * np.sin(phi_edge) * np.sin(phi_obs))
        dr_dq_cosdphi = np.array([rI_interp[0][i].ev(phi_edge[i], q[:, None], dy=1) for i in
                                  range(slice.shape[0])])
        dr_dq_cosdphi = dr_dq_cosdphi * np.cos(phi_edge - np.array([phi_obs, phi_obs + np.pi]))
        # (# of img, 4, # of q, near or far)
        return np.moveaxis((I_edge, r_proj, r_curv, dr_dq_cosdphi), 0, 1)

def r_col_from_Irrd_one(I_edge, r_proj, r_curv, q=None, dr_dq_cosdphi=None):
    if q is None:
        q = linear_q()
    if dr_dq_cosdphi is None:
        dr_dq_cosdphi = np.ones_like(I_edge)
    return simpson(I_edge * np.sqrt(r_curv) * r_proj * dr_dq_cosdphi, q) / simpson(
        I_edge * np.sqrt(r_curv) * dr_dq_cosdphi, q)

def _M_from_Irrd_one(I_edge, r_proj, r_curv, n, q=None, dr_dq_cosdphi=None):
    if q is None:
        q = linear_q()
    if dr_dq_cosdphi is None:
        dr_dq_cosdphi = np.ones_like(I_edge)
    r_col = r_col_from_Irrd_one(I_edge, r_proj, r_curv, q, dr_dq_cosdphi)
    if n == 1:
        return r_col * (2 * np.pi * 1e9 / 206265 / 1e6)
    else:
        M = simpson(I_edge * np.sqrt(r_curv) * (2 * np.pi * (r_proj - r_col))**n / factorial(n) *
                    dr_dq_cosdphi, q)
        M *= (1e9 / 206265 / 1e6)**n * (1 / 206265 / 1e6)**1.5
        return M

def M_from_Irrd_one(I_edge, r_proj, r_curv, n=13, q=None, dr_dq_cosdphi=None):
    assert n >= 0
    return np.array([_M_from_Irrd_one(I_edge, r_proj, r_curv, i, q, dr_dq_cosdphi) for i in
                     range(n + 1)])

def _a_from_M(M, i):
    match i:
        case 0:
            return M[0]
        case 1:
            return M[1]
        case 2:
            return -M[2]
        case 3:
            return -M[3] / M[0]
        case 4:
            return M[4]
        case 5:
            return (-M[2] * M[3] + M[0] * M[5]) / M[0]**2
        case 6:
            return M[3]**2 / 2. / M[0] - M[6]
        case 7:
            return (-M[2]**2 * M[3] + M[0] * M[2] * M[5] +
                    M[0] * (M[3] * M[4] - M[0] * M[7])) / M[0]**3
        case 8:
            return M[3] * (M[2] * M[3] - 2. * M[0] * M[5]) / 2. / M[0]**2 + M[8]
        case 9:
            return (-3. * M[2]**3 * M[3] + 3. * M[0] * M[2]**2 * M[5] -
                    3. * M[0] * M[2] * (-2. * M[3] * M[4] + M[0] * M[7]) +
                    M[0] * (M[3]**3 - 3. * M[0] * M[3] * M[6] +
                            3. * M[0] * (-M[4] * M[5] + M[0] * M[9]))) / 3. / M[0]**4
        case 10:
            return (M[2]**2 * M[3]**2 - 2. * M[0] * M[2] * M[3] * M[5] +
                    M[0] * (-M[3]**2 * M[4] + M[0] * M[5]**2 +
                            2. * M[0] * M[3] * M[7])) / 2. / M[0]**3 - M[10]
        case 11:
            return (-M[2]**4 * M[3] + M[0] * M[2]**3 * M[5] +
                    M[0] * M[2]**2 * (3. * M[3] * M[4] - M[0] * M[7]) +
                    M[0] * M[2] * (M[3]**3 - 2 * M[0] * M[3] * M[6] +
                                   M[0] * (-2. * M[4] * M[5] + M[0] * M[9])) +
                    M[0]**2 * (-M[3]**2 * M[5] + M[3] * (-M[4]**2 + M[0] * M[8]) +
                               M[0] * (M[5] * M[6] + M[4] * M[7] - M[0] * M[11]))) / M[0]**5
        case 12:
            return (4. * M[2]**3 * M[3]**2 - 8. * M[0] * M[2]**2 * M[3] * M[5] +
                    4. * M[0] * M[2] * (-2. * M[3]**2 * M[4] + M[0] * M[5]**2 +
                                        2. * M[0] * M[3] * M[7]) +
                    M[0] * (-M[3]**4 + 4. * M[0] * M[3]**2 * M[6] +
                            8. * M[0] * M[3] * (M[4] * M[5] - M[0] * M[9]) +
                            8. * M[0]**2 * (-M[5] * M[7] + M[0] * M[12]))) / 8. / M[0]**4
        case 13:
            return (-M[2]**5 * M[3] + M[0] * M[2]**4 * M[5] +
                    M[0] * M[2]**3 * (4. * M[3] * M[4] - M[0] * M[7]) +
                    M[0] * M[2]**2 * (2. * M[3]**3 - 3. * M[0] * M[3] * M[6] +
                                      M[0] * (-3. * M[4] * M[5] + M[0] * M[9])) +
                    M[0]**2 * M[2] * (-3. * M[3]**2 * M[5] +
                                      M[3] * (-3. * M[4]**2 + 2. * M[0] * M[8]) +
                                      M[0] * (2. * M[5] * M[6] + 2. * M[4] * M[7] - M[0] * M[11])) +
                    M[0]**2 * (-M[3]**3 * M[4] + M[0] * M[3]**2 * M[7] -
                               M[0] * (-M[4]**2 * M[5] + M[0] * (M[6] * M[7] + M[5] * M[8]) +
                                       M[0] * M[4] * M[9]) +
                               M[0] * M[3] * (M[5]**2 + 2. * M[4] * M[6] - M[0] * M[10]) +
                               M[0]**3 * M[13]))  / M[0]**6
        case _:
            raise NotImplementedError

def a_from_M(M):
    return np.array([_a_from_M(M, i) for i in range(len(M))])

def a_from_Irrd_one(I_edge, r_proj, r_curv, n=13, q=None, dr_dq_cosdphi=None):
    return a_from_M(M_from_Irrd_one(I_edge, r_proj, r_curv, n, q, dr_dq_cosdphi))

def a_from_Irrd(Irrd, n=13, q=None):
    Irrd = np.asarray(Irrd) # (# of img, 4, # of q, near or far)
    assert Irrd.ndim == 4
    return np.asarray([[a_from_Irrd_one(Irrd[i, 0, :, j], Irrd[i, 1, :, j], Irrd[i, 2, :, j], n, q,
                                        Irrd[i, 3, :, j]) for j in range(2)]
                       for i in range(Irrd.shape[0])])

def A_phi_from_a(u, a, n=None):
    assert len(a) >= 2
    if n is None:
        n = len(a) - 1
    else:
        n = int(n)
        assert len(a) >= n + 1
    A = a[0]
    phi = a[1] * u
    for i in range(2, n + 1):
        if i % 2:
            phi += a[i] * u**i
        else:
            A += a[i] * u**i
    return A, phi

def r_col_from_a(a):
    return a[..., 1] / (2 * np.pi * 1e9 / 206265 / 1e6)

# def d_col_from_a(a_near, a_far):
#     return r_col_from_a(a_near) + r_col_from_a(a_far)

def Vu(u, a_near, a_far=None, n=None, A=None, B=None):
    A_near, phi_near = A_phi_from_a(u, a_near, n)
    if a_far is None:
        A_far, phi_far = A_near, phi_near
    else:
        A_far, phi_far = A_phi_from_a(u, a_far, n)
    if A is None:
        A = (A_near**2 + A_far**2) / u
    if B is None:
        B = 2 * A_near * A_far / u
    return np.sqrt(A + B * np.sin(phi_near + phi_far))
