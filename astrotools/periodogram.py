"""Lomb-Scargle and BLS periodogram helpers.

Frequency-grid construction and quality-factor parameters match the GPU
pipeline in Period_Finding.py (cuvarbase / eebls_gpu_fast), so CPU
periodograms computed here have the same frequency axis as the bulk-search
results stored in the .result files.

Key design choices mirrored from Period_Finding.py
---------------------------------------------------
* **Linear frequency spacing**: df = df_scale / baseline, not log in period.
  A log-spaced period grid over-samples short periods and under-samples long
  ones; uniform df gives the same resolution per resolution element across
  the whole search range.
* **BLS df_scale = qmin**: matches `df = qmin / baseline` in Period_Finding.BLS.
* **LS df_scale = 1 / oversample_factor**: matches LS_Full (oversample=3) and
  LS (oversample=7); default here is 5 (astropy autopower convention).
* **BLS q range**: qmin=0.02, qmax=0.12 (Period_Finding.py defaults).
* **BLS duration grid**: log-spaced durations between qmin×P and qmax×P with
  uniform spacing dlogq=0.1 in log(q), matching the geomspace(qmin,qmax,100)
  grid in Period_Finding.BLS (log(0.12/0.02)/0.1 ≈ 18 trial durations).
* **Harmonic removal**: sub-harmonics of the 200 s TESS cadence
  (86400/200/i cycles/day for i=2..n) are masked, matching LS_Astropy and
  remove_harmonics in Period_Finding.py.
* **Significance metric**: (max−mean)/std, matching Period_Finding.LS_Full /
  LS_Astropy.
"""

from __future__ import annotations

import numpy as np
from astropy.timeseries import BoxLeastSquares, LombScargle


# ── internal helpers ──────────────────────────────────────────────────────────

def _resolve_period_range(time, min_period, max_period):
    """Return (min_p, max_p) in days, filling in defaults from the data.

    Default fmin = 4/baseline (at least 4 complete cycles), matching
    Period_Finding.py's ``fmin = 4/baseline``.
    Default fmax corresponds to pmin = 2 min.
    """
    baseline = float(np.ptp(time))
    if max_period is None:
        max_period = baseline / 4.0
    if min_period is None:
        min_period = 2.0 / 1440.0   # 2 minutes
    return float(min_period), float(max_period)


def _linear_freq_grid(min_period, max_period, baseline, df_scale, n_max=None):
    """Build a linearly-spaced frequency array (cycles/day).

    ``df = df_scale / baseline`` matches Period_Finding.py:
    * LS:  df_scale = 1/oversample_factor
    * BLS: df_scale = qmin

    If ``n_max`` is given and the natural grid exceeds it, the grid is
    uniformly subsampled (preserving linear-in-frequency structure).
    """
    df = df_scale / baseline
    fmin = 1.0 / max_period
    fmax = 1.0 / min_period
    nf = int(np.ceil((fmax - fmin) / df))
    freqs = fmin + df * np.arange(nf)
    if n_max is not None and nf > n_max:
        idx = np.round(np.linspace(0, nf - 1, n_max)).astype(int)
        freqs = freqs[idx]
    return freqs


# ── harmonic removal ──────────────────────────────────────────────────────────

def remove_cadence_harmonics(freqs, power, cadence_s=200.0, n_harmonics=7,
                              tolerance=0.1):
    """Mask spikes at the cadence period and its sub-harmonics.

    Removes frequency windows centred on ``86400 / cadence_s / i`` cycles/day
    for i = 1 … n_harmonics.  i = 1 is the fundamental cadence period (e.g.
    200 s = 3.33 min for TESS 200 s FFIs); i ≥ 2 are sub-harmonics.  Matches
    ``rm_freq_tess`` in ``lc_utils.py`` which also removes the fundamental.

    Parameters
    ----------
    freqs : ndarray
        Frequency array in cycles/day.
    power : ndarray
        Corresponding power values (same length as freqs).
    cadence_s : float
        Sampling cadence in seconds.  200 for TESS 200 s FFIs.
    n_harmonics : int
        Remove i = 1 … n_harmonics harmonics (default: 7).
    tolerance : float
        Half-width of each removal window in cycles/day (0.1 matches
        ``rm_freq_tess`` / ``LS_Astropy``).

    Returns
    -------
    freqs, power : ndarray
        Arrays with harmonic windows excised.
    """
    f_fund = 86400.0 / cadence_s   # fundamental cadence frequency (cycles/day)
    keep = np.ones(len(freqs), dtype=bool)
    for i in range(1, n_harmonics + 1):
        centre = f_fund / i
        keep &= (freqs < centre - tolerance) | (freqs > centre + tolerance)
    return freqs[keep], power[keep]


# ── main periodogram functions ────────────────────────────────────────────────

def compute_ls(
    time,
    flux,
    flux_err=None,
    min_period=None,
    max_period=None,
    oversample_factor=5.0,
    remove_tess_harmonics=False,
    normalization="standard",
):
    """Compute a Lomb-Scargle periodogram on a linearly-spaced frequency grid.

    Frequency spacing ``df = 1 / (oversample_factor × baseline)`` matches
    Period_Finding.LS_Full (oversample=3) / LS (oversample=7).  The default
    oversample=5 aligns with astropy's ``samples_per_peak`` convention.

    The full natural grid is always used; no downsampling is applied.

    Parameters
    ----------
    time : array-like
        Observation times in days.
    flux : array-like
        Flux values (astropy fits and removes the mean internally).
    flux_err : array-like, optional
        Per-point uncertainties.
    min_period, max_period : float, optional
        Period range in days.  Defaults: 2 min and baseline/4.
    oversample_factor : float
        Frequency grid oversampling relative to the resolution element.
    remove_tess_harmonics : bool
        If True, excise the 200 s TESS cadence sub-harmonics.
    normalization : str
        Passed to ``astropy.timeseries.LombScargle``.  ``"standard"``
        normalises peak power to [0, 1].

    Returns
    -------
    periods : ndarray
        Test periods in days, sorted ascending.
    power : ndarray
        LS power at each period.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    min_p, max_p = _resolve_period_range(time, min_period, max_period)
    baseline = float(np.ptp(time))
    freqs = _linear_freq_grid(min_p, max_p, baseline,
                               df_scale=1.0 / oversample_factor)

    ls = LombScargle(time, flux, flux_err, normalization=normalization)
    power = ls.power(freqs)

    if remove_tess_harmonics:
        freqs, power = remove_cadence_harmonics(freqs, power)

    # freqs is already ascending; flip to get ascending period
    return 1.0 / freqs[::-1], power[::-1]


def compute_bls(
    time,
    flux,
    flux_err=None,
    min_period=None,
    max_period=None,
    qmin=0.02,
    qmax=0.12,
    dlogq=0.1,
    noverlap=3,
    remove_tess_harmonics=False,
    _n_max=None,
    use_gpu=True,
):
    """Compute a BLS periodogram on a linearly-spaced frequency grid.

    Tries to use ``cuvarbase.bls.eebls_gpu_fast`` (GPU) when available, and
    falls back to astropy ``BoxLeastSquares`` otherwise.

    The GPU implementation matches ``Period_Finding.BLS`` exactly:
    * frequency spacing ``df = qmin / baseline``
    * per-period duration grid ``[qmin*P, qmax*P]`` with log-spacing ``dlogq``
    * returns ``1 − χ²(model)/χ²(constant)`` ∈ [0, 1]

    The astropy fallback uses period chunks so that astropy's global
    ``max(duration) < min(period)`` constraint is satisfied while keeping
    ``d_max / P_min ≤ 0.5`` to avoid the BLS singularity as q → 1.

    Parameters
    ----------
    time : array-like
        Observation times in days.
    flux : array-like
        Normalized flux (transit candidates appear as dips below 1).
    flux_err : array-like, optional
        Per-point uncertainties.  GPU path uses unit weights when None.
    min_period, max_period : float, optional
        Period range in days.  Defaults: 2 min and baseline/4.
    qmin, qmax : float
        Min/max transit duration as a fraction of the period.
    dlogq : float
        Log-spacing of trial durations.  Default 0.1.
    noverlap : int
        Phase-grid overlap factor passed to ``eebls_gpu_fast`` (GPU only).
    remove_tess_harmonics : bool
        If True, excise the TESS 200 s cadence period and its sub-harmonics.
    use_gpu : bool
        If True (default), attempt to use the GPU path first.

    Returns
    -------
    periods : ndarray
        Test periods in days, sorted ascending.
    power : ndarray
        BLS power at each period.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    min_p, max_p = _resolve_period_range(time, min_period, max_period)
    baseline = float(np.ptp(time))
    freqs = _linear_freq_grid(min_p, max_p, baseline, df_scale=qmin,
                               n_max=_n_max)

    # ── GPU path (cuvarbase eebls_gpu_fast) ───────────────────────────────────
    if use_gpu:
        try:
            import cuvarbase.bls as cv_bls
            dy = (flux_err if flux_err is not None
                  else np.ones(len(flux), dtype=np.float64))
            # Centre times for numerical stability (matches Period_Finding.BLS)
            t_c = time - np.mean(time)
            power_gpu = cv_bls.eebls_gpu_fast(
                t_c, flux, dy, freqs,
                qmin=qmin, qmax=qmax, dlogq=dlogq, noverlap=noverlap,
            )
            power = np.asarray(power_gpu, dtype=float)
            if remove_tess_harmonics:
                freqs, power = remove_cadence_harmonics(freqs, power)
            # freqs is ascending → periods descending; flip to ascending period
            return 1.0 / freqs[::-1], power[::-1]
        except Exception:
            pass   # GPU unavailable or failed — fall through to astropy

    # ── CPU fallback (astropy BoxLeastSquares) ────────────────────────────────
    periods = 1.0 / freqs   # descending period order (ascending freq)

    # astropy requires max(duration) < min(period) as a *global* constraint.
    # Split into chunks with ratio 0.5/qmax so d_max = 0.5 × chunk_lo, keeping
    # q_eff = d/P ≤ 0.5 for the shortest period in every chunk.  The previous
    # ratio of 1/qmax set d_max ≈ P_min → q_eff ≈ 1, causing a BLS singularity
    # (n/(n−n_in) → ∞) that produced large spurious spikes at chunk boundaries.
    power = np.zeros(len(periods))
    chunk_lo = min_p
    while chunk_lo < max_p:
        chunk_hi = min(chunk_lo * 0.5 / qmax, max_p)
        mask = (periods >= chunk_lo) & (periods <= chunk_hi)
        if mask.any():
            d_min = qmin * chunk_lo
            d_max = qmax * chunk_hi   # = 0.5 × chunk_lo → d/P ≤ 0.5 always
            n_dur = max(2, round(np.log(d_max / d_min) / dlogq) + 1)
            durations = np.geomspace(d_min, d_max, n_dur)
            result = BoxLeastSquares(time, flux, flux_err).power(
                periods[mask], durations
            )
            power[mask] = np.asarray(result.power)
        chunk_lo = chunk_hi

    if remove_tess_harmonics:
        freqs, power = remove_cadence_harmonics(freqs, power)
        periods = 1.0 / freqs

    # flip to ascending period order
    return periods[::-1], power[::-1]


# ── peak finding ──────────────────────────────────────────────────────────────

def find_peaks(periods, power, n=3, min_separation_factor=0.1):
    """Return the top-*n* peaks in a periodogram.

    Peaks are selected greedily from highest to lowest power, requiring each
    new peak to be separated from all already-selected peaks by at least
    ``min_separation_factor × min(p_new, p_selected)``.

    Parameters
    ----------
    periods, power : array-like
    n : int
        Maximum number of peaks to return.
    min_separation_factor : float
        Minimum fractional separation between peaks.

    Returns
    -------
    peak_periods, peak_powers : ndarray
        Sorted by descending power.
    """
    periods = np.asarray(periods, dtype=float)
    power = np.asarray(power, dtype=float)

    order = np.argsort(power)[::-1]
    sel_periods, sel_powers = [], []

    for idx in order:
        p, pw = periods[idx], power[idx]
        too_close = any(
            abs(p - p2) / min(p, p2) < min_separation_factor
            for p2 in sel_periods
        )
        if not too_close:
            sel_periods.append(p)
            sel_powers.append(pw)
        if len(sel_periods) >= n:
            break

    return np.array(sel_periods), np.array(sel_powers)


# ── significance ──────────────────────────────────────────────────────────────

def significance(power):
    """Return the peak SNR significance of a periodogram.

    Computes ``(max − mean) / std``, matching ``Period_Finding.LS_Full``
    and ``LS_Astropy``.

    Parameters
    ----------
    power : array-like

    Returns
    -------
    float
    """
    p = np.asarray(power, dtype=float)
    return float((np.max(p) - np.mean(p)) / np.std(p))
