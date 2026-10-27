"""Shared feedforward ANC loop for LMS, NLMS, FxLMS and FxNLMS.

Signal model and sign convention (used by every algorithm):
    x[n]      reference signal
    y[n]      controller output, y = w . [x[n], x[n-1], ...]
    anti[n]   y passed through the TRUE secondary path s
    d[n]      primary-path disturbance at the error mic (clean)
    e[n]      measured error = d[n] + sensor_noise[n] - anti[n]
So y is the cancelling signal and it is subtracted at the error mic.

All algorithms run through the same plant, so differences between them come from
the update rule only. LMS/NLMS use the raw reference as the regressor (no
secondary-path compensation). FxLMS/FxNLMS filter the reference through the
secondary-path ESTIMATE first.

Update order at sample n: output is computed with the weights from n-1,
the error is formed from that output, then the weights are updated.
"""
import numpy as np
from numba import njit
from scipy.signal import lfilter


@njit(cache=True)
def _loop(x, regressor, d_measured, num_taps, step_size, secondary_path, normalize, eps):
    n = len(x)
    num_s = len(secondary_path)
    w = np.zeros(num_taps)
    y = np.zeros(n)
    anti = np.zeros(n)
    e = np.zeros(n)
    xp = np.concatenate((np.zeros(num_taps - 1), x))
    rp = np.concatenate((np.zeros(num_taps - 1), regressor))
    y_hist = np.zeros(num_s)  # newest first

    for i in range(n):
        xr = xp[i:i + num_taps][::-1]
        rr = rp[i:i + num_taps][::-1]

        y[i] = np.dot(w, xr)
        for k in range(num_s - 1, 0, -1):
            y_hist[k] = y_hist[k - 1]
        y_hist[0] = y[i]
        anti[i] = np.dot(secondary_path, y_hist)

        e[i] = d_measured[i] - anti[i]

        mu = step_size
        if normalize:
            mu = step_size / (np.dot(rr, rr) + eps)
        w += mu * e[i] * rr
    return w, y, anti, e


def run_anc(mode, x, d_clean, sensor_noise, num_taps, step_size,
            secondary_path, secondary_path_estimate=None, eps=1e-6):
    """Run one adaptive experiment. mode in {'LMS','NLMS','FxLMS','FxNLMS'}.

    Returns a dict with final weights, controller output y, anti-noise at the
    error mic, measured error e, and the acoustic residual (d_clean - anti),
    which is what a listener at the error mic would hear during adaptation.
    """
    if mode not in ('LMS', 'NLMS', 'FxLMS', 'FxNLMS'):
        raise ValueError(f'Unknown mode: {mode}')
    s_true = np.asarray(secondary_path, dtype=np.float64)
    s_hat = s_true if secondary_path_estimate is None else np.asarray(secondary_path_estimate, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    regressor = lfilter(s_hat, 1, x) if mode.startswith('Fx') else x
    d_measured = d_clean + sensor_noise
    with np.errstate(over='ignore', invalid='ignore'):
        w, y, anti, e = _loop(x, regressor, d_measured, num_taps, step_size,
                              s_true, mode.endswith('NLMS'), eps)
    return {'weights': w, 'y': y, 'anti': anti, 'error': e, 'residual': d_clean - anti}


def reduction_db(before, after, start=0, stop=None, eps=1e-20):
    before = before[start:stop]
    after = after[start:stop]
    return 10 * np.log10((np.mean(before ** 2) + eps) / (np.mean(after ** 2) + eps))
