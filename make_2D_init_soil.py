#!/usr/bin/env python3
"""
make_2D_init_soil.py  (v4 — NCL-fidelity improvements)
=======================================================
Unified Python replacement for 7 NCL soil initialization scripts.

Interpolation & fill backends (--interp_backend, --fill_backend):
  python_ncl     : NCL-faithful implementations (default)
                    - linint2_like_vec: bilinear interp matching NCL linint2
                    - poisson_grid_fill_ncl: Gauss-Seidel SOR with spherical
                      Laplacian, cos(lat) weighting, cyclic longitude
  python_ncl_fast: Faster but less NCL-faithful
                    - Same linint2 as above
                    - poisson_grid_fill_jacobi: Jacobi simultaneous update
                      (same stencil, different iteration order → different
                      convergence path → slightly different fill values)
  scipy          : Legacy fallback (not NCL-like)
                    - RegularGridInterpolator (extrap instead of NaN for OOD)
                    - Cartesian Laplacian (no spherical weighting)

Default behavior:
  - Out-of-domain interpolation points → NaN (matching NCL _FillValue)
  - NaN source corners → NaN output (matching NCL)
  - Post-interpolation NaN are NOT silently filled unless --fill_interp_nan
  - Poisson fill uses Gauss-Seidel SOR matching NCL's iteration order

Remaining known differences from NCL (see function docstrings for details):
  - Floating-point: O(1e-7) differences from Fortran vs Python arithmetic
  - Polar row treatment: cos(φ) clamped at 0.01; NCL clamp undocumented
  - Deep-ocean fill: O(0.1K) differences far from land; negligible near coast

Modes: auto, era5_ncar, era5_ncar_lsm, era5_ncar_delta, era5_notncar,
       era5_wetness, gfs, cesm_cc
"""
import argparse
import calendar
import os
import socket
import struct
import sys
import warnings

import numpy as np
import xarray as xr

try:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import laplace
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
#  Global config — set by CLI or callers
# ---------------------------------------------------------------------------
_INTERP_BACKEND = "python_ncl"       # python_ncl | python_ncl_fast | scipy
_FILL_BACKEND = "python_ncl"         # python_ncl | python_ncl_fast | scipy
_FILL_INTERP_NAN = False             # if True, NN-fill NaN after interpolation

_DERECHO_DEFAULT_FILEPATH = (
    "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc"
)

ZSOIL_ERA5 = np.array([0.035, 0.175, 0.64, 1.945], dtype=np.float32)
ZSOIL_GFS = np.array([0.05, 0.25, 0.7, 1.5], dtype=np.float32)
NSOIL = 4

STL_CODES = [
    "128_139_stl1", "128_170_stl2", "128_183_stl3", "128_236_stl4"
]
SWVL_CODES = [
    "128_039_swvl1", "128_040_swvl2", "128_041_swvl3", "128_042_swvl4"
]

POROSITY_TABLE = {
    1: 0.403, 2: 0.439, 3: 0.430, 4: 0.520,
    5: 0.614, 6: 0.766, 7: 0.472,
}
POROSITY_DEFAULT = 0.4


# ---------------------------------------------------------------------------
def is_on_derecho():
    fqdn = socket.getfqdn().lower()
    scratch = os.environ.get("SCRATCH", "")
    return ("derecho.hpc.ucar.edu" in fqdn
            or scratch.startswith("/glade/derecho/scratch/"))


# ============================= Validation ==================================
def _require_file(p, label="File"):
    if not p:
        raise ValueError(f"{label}: path is empty.")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{label}: {p}")


def _require_dir(p, label="Dir"):
    if not p:
        raise ValueError(f"{label}: path is empty.")
    os.makedirs(p, exist_ok=True)


def _require_var(ds, v, path=""):
    if v not in ds.data_vars and v not in ds.coords:
        avail = list(ds.data_vars) + list(ds.coords)
        raise KeyError(f"'{v}' not in {path}. Available: {avail}")


def _resolve_coord(ds, cands, label="coord"):
    for c in cands:
        if c in ds.coords or c in ds.dims:
            return c
    avail = list(ds.coords) + list(ds.dims)
    raise KeyError(f"No {label} coord. Tried {cands}, have: {avail}")


def _grids_aligned(lat_a, lon_a, lat_b, lon_b, atol=0.01):
    """Check shape + coordinate values within tolerance."""
    if lat_a.shape != lat_b.shape or lon_a.shape != lon_b.shape:
        return False
    return (np.allclose(lat_a, lat_b, atol=atol)
            and np.allclose(lon_a, lon_b, atol=atol))


# ======================= NCL-like linint2 ==================================
def linint2_like_vec(src_lon, src_lat, data_2d, cyclic, dst_lon, dst_lat):
    """
    Vectorized bilinear interpolation matching NCL linint2 / linint2_Wrap.

    NCL-matched behavior:
    - src_lon, src_lat: monotonically increasing 1D arrays.
    - Out-of-domain target points → NaN (NCL: _FillValue).
    - Any NaN among the 4 surrounding source points → NaN output.
    - cyclic=True: longitude wraps across the last→first column.

    Remaining differences from NCL:
    - Fortran vs Python floating-point: O(1e-7) differences possible.
    - NCL "_Wrap" suffix copies coordinate metadata; callers do that here.
    """
    src_lon = np.asarray(src_lon, dtype=np.float64)
    src_lat = np.asarray(src_lat, dtype=np.float64)
    data = np.asarray(data_2d, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    ny_s, nx_s = data.shape

    # --- Longitude indexing ---
    if cyclic:
        dlon = src_lon[1] - src_lon[0] if nx_s > 1 else 360.0
        period = src_lon[-1] + dlon - src_lon[0]
        dl_norm = src_lon[0] + np.mod(dst_lon - src_lon[0], period)
        ix = np.searchsorted(src_lon, dl_norm, side='right') - 1
        ix = np.clip(ix, 0, nx_s - 1)
        ix_r = np.where(ix == nx_s - 1, 0, ix + 1)
        # Safe index for gap computation (only used when ix < nx_s-1)
        safe_next = np.clip(ix + 1, 0, nx_s - 1)
        gap = np.where(
            ix == nx_s - 1,
            src_lon[0] + period - src_lon[ix],
            src_lon[safe_next] - src_lon[ix],
        )
        gap = np.maximum(gap, 1e-15)
        wx = (dl_norm - src_lon[ix]) / gap
        lon_ok = np.ones(len(dst_lon), dtype=bool)
    else:
        ix = np.searchsorted(src_lon, dst_lon, side='right') - 1
        lon_ok = (ix >= 0) & (ix < nx_s - 1)
        ix = np.clip(ix, 0, nx_s - 2)
        ix_r = ix + 1
        gap = np.maximum(src_lon[ix_r] - src_lon[ix], 1e-15)
        wx = (dst_lon - src_lon[ix]) / gap

    # --- Latitude indexing (never cyclic) ---
    jy = np.searchsorted(src_lat, dst_lat, side='right') - 1
    lat_ok = (jy >= 0) & (jy < ny_s - 1)
    jy = np.clip(jy, 0, ny_s - 2)
    jy_t = jy + 1
    gap_lat = np.maximum(src_lat[jy_t] - src_lat[jy], 1e-15)
    wy = (dst_lat - src_lat[jy]) / gap_lat

    # --- Build meshgrid of indices ---
    JY, IX = np.meshgrid(jy, ix, indexing='ij')
    JYT, IXR = np.meshgrid(jy_t, ix_r, indexing='ij')
    WY2, WX2 = np.meshgrid(wy, wx, indexing='ij')
    LOK, VOK = np.meshgrid(lat_ok, lon_ok, indexing='ij')
    valid = LOK & VOK

    q00 = data[JY, IX]
    q10 = data[JY, IXR]
    q01 = data[JYT, IX]
    q11 = data[JYT, IXR]

    out = (q00 * (1 - WX2) * (1 - WY2) + q10 * WX2 * (1 - WY2)
           + q01 * (1 - WX2) * WY2 + q11 * WX2 * WY2)

    any_nan = np.isnan(q00) | np.isnan(q10) | np.isnan(q01) | np.isnan(q11)
    out[any_nan | ~valid] = np.nan
    return out


# ==================== NCL-like poisson_grid_fill ===========================
def poisson_grid_fill_ncl(field, lat_deg, is_cyclic=True, guess=1,
                          nscan=2000, eps=1e-2, relc=0.6, opt=0,
                          dlon_deg=None):
    """
    Gauss-Seidel SOR solver for Laplace's equation on a lat-lon sphere.

    This is the most NCL-faithful implementation: scalar point-by-point
    updates in row-major order, matching NCL's Fortran iteration pattern.

    Spherical Laplacian discretization at grid point (j,i):
        α  = (Δφ/Δλ)² / cos²(φ_j)
        β_n = 1 + tan(φ_j)·Δφ/2
        β_s = 1 - tan(φ_j)·Δφ/2
        D   = 2α + β_n + β_s
        f* = [α·(f(j,i+1)+f(j,i-1)) + β_n·f(j+1,i) + β_s·f(j-1,i)] / D
        f_new = relc·f* + (1-relc)·f_old

    Parameters match NCL's poisson_grid_fill(x, is_cyclic, guess, nscan,
    eps, relc, opt).

    Remaining differences from NCL:
    - Exact polar-row treatment may differ slightly from NCL Fortran.
    - cos(φ) clamped to >=0.01 to avoid div-by-zero; NCL's exact clamp
      value is undocumented.
    - Convergence checked as max absolute change per scan (same as NCL).
    - Expect O(0.1K) differences in deep-ocean fill far from coast;
      negligible differences on land or near-coast fill.
    """
    data = field.copy().astype(np.float64)
    mask = np.isnan(data)
    if not mask.any():
        return data

    ny, nx = data.shape
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    if len(lat_rad) != ny:
        raise ValueError(f"lat_deg length {len(lat_rad)} != data rows {ny}")

    dlat = abs(lat_rad[1] - lat_rad[0]) if ny > 1 else np.deg2rad(0.25)
    if is_cyclic:
        dlon = np.deg2rad(360.0 / nx)
    elif dlon_deg is not None:
        dlon = np.deg2rad(dlon_deg)
    else:
        # For non-cyclic (cropped) domains without explicit dlon,
        # assume equal to dlat (isotropic). This is exact for ERA5 0.25°.
        dlon = dlat

    # Performance note for Gauss-Seidel on large grids
    n_fill = int(mask.sum())
    if n_fill > 100000:
        print(f"    [poisson_ncl] Gauss-Seidel on {ny}x{nx} grid, "
              f"{n_fill} fill points, {nscan} max scans — this may be slow. "
              f"Use --fill_backend python_ncl_fast for ~10-50x speedup "
              f"(less NCL-faithful).")

    cosph = np.clip(np.cos(lat_rad), 0.01, None)
    tanph = np.clip(np.tan(lat_rad), -100.0, 100.0)

    ratio2 = (dlat / dlon) ** 2
    alpha = ratio2 / (cosph ** 2)
    beta_n = 1.0 + tanph * dlat / 2.0
    beta_s = 1.0 - tanph * dlat / 2.0
    denom = 2.0 * alpha + beta_n + beta_s

    valid_vals = data[~mask]
    if len(valid_vals) == 0:
        warnings.warn("poisson_grid_fill_ncl: all NaN, returning zeros")
        data[:] = 0.0
        return data
    data[mask] = np.mean(valid_vals) if guess == 1 else 0.0

    for scan in range(nscan):
        maxdiff = 0.0
        for j in range(ny):
            a = alpha[j]
            bn = beta_n[j]
            bs = beta_s[j]
            d = denom[j]
            jn = min(j + 1, ny - 1)
            js = max(j - 1, 0)
            for i in range(nx):
                if not mask[j, i]:
                    continue
                if is_cyclic:
                    ie = (i + 1) % nx
                    iw = (i - 1) % nx
                else:
                    ie = min(i + 1, nx - 1)
                    iw = max(i - 1, 0)
                fstar = (a * (data[j, ie] + data[j, iw])
                         + bn * data[jn, i]
                         + bs * data[js, i]) / d
                fnew = relc * fstar + (1.0 - relc) * data[j, i]
                diff = abs(fnew - data[j, i])
                if diff > maxdiff:
                    maxdiff = diff
                data[j, i] = fnew
        if maxdiff < eps:
            break
    return data


def poisson_grid_fill_jacobi(field, lat_deg, is_cyclic=True, guess=1,
                             nscan=2000, eps=1e-2, relc=0.6, opt=0,
                             dlon_deg=None):
    """
    Jacobi-style (simultaneous update) vectorized solver. Same stencil as
    poisson_grid_fill_ncl but updates all fill points at once per scan.

    Trade-off: ~10-50x faster than the Gauss-Seidel version for large grids,
    but Jacobi converges slower and iteration pattern differs from NCL.
    May need more scans and produces slightly different fill values than
    NCL in practice.
    """
    data = field.copy().astype(np.float64)
    mask = np.isnan(data)
    if not mask.any():
        return data

    ny, nx = data.shape
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    dlat = abs(lat_rad[1] - lat_rad[0]) if ny > 1 else np.deg2rad(0.25)
    if is_cyclic:
        dlon = np.deg2rad(360.0 / nx)
    elif dlon_deg is not None:
        dlon = np.deg2rad(dlon_deg)
    else:
        dlon = dlat  # assume isotropic for non-global cropped domains

    cosph = np.clip(np.cos(lat_rad), 0.01, None)
    tanph = np.clip(np.tan(lat_rad), -100.0, 100.0)
    ratio2 = (dlat / dlon) ** 2

    a = (ratio2 / cosph ** 2)[:, None] * np.ones((1, nx))
    bn = (1.0 + tanph * dlat / 2.0)[:, None] * np.ones((1, nx))
    bs = (1.0 - tanph * dlat / 2.0)[:, None] * np.ones((1, nx))
    d = 2.0 * a + bn + bs

    valid_mean = np.nanmean(data) if guess == 1 else 0.0
    if np.isnan(valid_mean):
        valid_mean = 0.0
    data[mask] = valid_mean

    for scan in range(nscan):
        if is_cyclic:
            fe = np.roll(data, -1, axis=1)
            fw = np.roll(data, 1, axis=1)
        else:
            fe = np.empty_like(data)
            fe[:, :-1] = data[:, 1:]
            fe[:, -1] = data[:, -1]
            fw = np.empty_like(data)
            fw[:, 1:] = data[:, :-1]
            fw[:, 0] = data[:, 0]
        fn = np.empty_like(data)
        fn[:-1] = data[1:]
        fn[-1] = data[-1]
        fs = np.empty_like(data)
        fs[1:] = data[:-1]
        fs[0] = data[0]

        fstar = (a * (fe + fw) + bn * fn + bs * fs) / d
        update = relc * fstar + (1.0 - relc) * data
        maxdiff = np.abs(update - data)[mask].max() if mask.any() else 0.0
        data[mask] = update[mask]
        if maxdiff < eps:
            break
    return data


# ==================== Scipy fallbacks ======================================
def _scipy_bilinear(data_2d, src_lat, src_lon, dst_lat, dst_lon):
    """scipy RegularGridInterpolator (fill_value=None → nearest extrap).
    NOTE: This does NOT match NCL — out-of-domain points get extrapolated
    instead of being set to NaN/_FillValue."""
    if not _HAS_SCIPY:
        raise ImportError("scipy not available; use --interp_backend python_ncl")
    sl = np.asarray(src_lat, dtype=np.float64)
    slo = np.asarray(src_lon, dtype=np.float64)
    d = np.asarray(data_2d, dtype=np.float64)
    if sl[0] > sl[-1]:
        sl = sl[::-1]
        d = d[::-1, :]
    interp = RegularGridInterpolator(
        (sl, slo), d, method='linear', bounds_error=False, fill_value=None
    )
    la2, lo2 = np.meshgrid(dst_lat, dst_lon, indexing='ij')
    pts = np.column_stack([la2.ravel(), lo2.ravel()])
    return interp(pts).reshape(la2.shape).astype(np.float32)


def _scipy_poisson(field, mask=None, niter=2000, tol=1e-2, relax=0.6):
    """scipy.ndimage.laplace Cartesian fallback. NOT spherical, does NOT
    match NCL's poisson_grid_fill."""
    if not _HAS_SCIPY:
        raise ImportError("scipy not available; use --fill_backend python_ncl")
    data = field.copy().astype(np.float64)
    if mask is None:
        mask = np.isnan(data)
    if not mask.any():
        return data
    vm = np.nanmean(data)
    if np.isnan(vm):
        data[:] = 0.0
        return data
    data[mask] = vm
    for _ in range(niter):
        lap = laplace(data)
        upd = relax * lap
        old = data[mask].copy()
        data[mask] += upd[mask]
        if np.max(np.abs(data[mask] - old)) < tol:
            break
    return data


# ===================== Dispatch wrappers ===================================
def interp_to_grid(data_2d, src_lat, src_lon, dst_lat, dst_lon, cyclic=False):
    """
    Dispatch interpolation to the configured backend.

    Default (python_ncl / python_ncl_fast): NCL-like linint2.
      Out-of-domain → NaN. NaN source corners → NaN.
      NaN points are preserved unless --fill_interp_nan is set.

    scipy backend: RegularGridInterpolator with nearest-boundary extrap.
      Out-of-domain → extrapolated (NOT NCL-like).
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    d = np.asarray(data_2d, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)

    # Ensure S→N
    if src_lat.size > 1 and src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        d = d[::-1, :]

    if _INTERP_BACKEND == "scipy":
        return _scipy_bilinear(d, src_lat, src_lon, dst_lat, dst_lon)

    # python_ncl or python_ncl_fast — both use the same vectorized linint2
    result = linint2_like_vec(
        src_lon, src_lat, d, cyclic, dst_lon, dst_lat
    )
    nan_count = int(np.isnan(result).sum())

    if nan_count > 0:
        if _FILL_INTERP_NAN:
            # Opt-in nearest-neighbor backfill for practical use cases
            # where source is global but target extends slightly beyond.
            # This is NOT what NCL linint2 does.
            print(f"    [interp] {nan_count} NaN after linint2; "
                  f"applying NN backfill (--fill_interp_nan)")
            _nn_fill_2d(result)
        else:
            print(f"    [interp] {nan_count} NaN after linint2 "
                  f"(out-of-domain or NaN source; preserved as NaN)")

    return result.astype(np.float32)


def _nn_fill_2d(arr):
    """In-place nearest-neighbor fill of NaN points from valid neighbors.
    Simple iterative expansion; NOT part of NCL pipeline."""
    mask = np.isnan(arr)
    if not mask.any():
        return
    ny, nx = arr.shape
    # Iterate until no more NaN can be filled (or 50 passes max)
    for _ in range(50):
        filled_any = False
        new = arr.copy()
        idx = np.where(mask)
        for j, i in zip(idx[0], idx[1]):
            neighbors = []
            if j > 0 and not np.isnan(arr[j - 1, i]):
                neighbors.append(arr[j - 1, i])
            if j < ny - 1 and not np.isnan(arr[j + 1, i]):
                neighbors.append(arr[j + 1, i])
            if i > 0 and not np.isnan(arr[j, i - 1]):
                neighbors.append(arr[j, i - 1])
            if i < nx - 1 and not np.isnan(arr[j, i + 1]):
                neighbors.append(arr[j, i + 1])
            if neighbors:
                new[j, i] = np.mean(neighbors)
                filled_any = True
        arr[:] = new
        mask = np.isnan(arr)
        if not mask.any() or not filled_any:
            break


def fill_poisson(field, lat_deg=None, lon_deg=None, is_cyclic=True,
                 nscan=2000, eps=1e-2, relc=0.6):
    """
    Dispatch Poisson fill to the configured backend.

    python_ncl (default): Gauss-Seidel SOR, most NCL-faithful, slow.
    python_ncl_fast: Jacobi vectorized, faster, less faithful.
    scipy: Cartesian laplacian, not NCL-like at all.

    Parameters
    ----------
    lon_deg : array-like, optional
        Source longitude array (degrees). Used to compute dlon for
        non-cyclic (cropped) domains. If None and not cyclic,
        dlon is assumed equal to dlat (isotropic approximation).
    """
    if lat_deg is None:
        ny = field.shape[0]
        lat_deg = np.linspace(-90, 90, ny)
        warnings.warn(
            "fill_poisson: lat_deg not provided, assuming uniform -90..90"
        )

    # Compute dlon from longitude array if available
    dlon_deg = None
    if lon_deg is not None and len(lon_deg) > 1:
        dlon_deg = abs(float(lon_deg[1] - lon_deg[0]))

    if _FILL_BACKEND == "scipy":
        return _scipy_poisson(field, niter=nscan, tol=eps, relax=relc)
    elif _FILL_BACKEND == "python_ncl_fast":
        return poisson_grid_fill_jacobi(
            field, lat_deg, is_cyclic=is_cyclic,
            guess=1, nscan=nscan, eps=eps, relc=relc,
            dlon_deg=dlon_deg,
        )
    else:
        # python_ncl — Gauss-Seidel (most faithful to NCL)
        return poisson_grid_fill_ncl(
            field, lat_deg, is_cyclic=is_cyclic,
            guess=1, nscan=nscan, eps=eps, relc=relc,
            dlon_deg=dlon_deg,
        )


# ===================== Common utilities ====================================
def flip_lat_if_needed(data, lat):
    """Ensure latitude runs S→N. Corresponds to NCL nnn:0:1 reversal."""
    if lat.size > 1 and lat[0] > lat[-1]:
        return data[..., ::-1, :], lat[::-1].copy()
    return data, lat


def flip_lon_if_needed(data, lon, target_lon):
    """Shift longitude convention (0..360 ↔ -180..180) to match target."""
    if np.any(target_lon < 0) and np.all(lon >= 0):
        lon = np.where(lon > 180, lon - 360, lon).copy()
        ix = np.argsort(lon)
        lon = lon[ix]
        data = data[..., ix]
    elif np.any(target_lon >= 0) and np.any(lon < 0):
        lon = np.where(lon < 0, lon + 360, lon).copy()
        ix = np.argsort(lon)
        lon = lon[ix]
        data = data[..., ix]
    return data, lon


def apply_land_mask(f2d, lm):
    """Set ocean/ice (landmask==0 or 15) to 0; clamp negatives to 0."""
    f2d = np.where((lm == 0) | (lm == 15), 0.0, f2d)
    return np.where(f2d < 0, 0.0, f2d)


def crop_to_domain(data, slat, slon, dlat, dlon, margin=1.0):
    """
    Crop source field to target domain ± margin degrees.

    Corresponds to NCL: fld1 = fld0({latmin:latmax},{lonmin:lonmax})

    Longitude handling:
    - Normalizes target domain bounds to source longitude convention
      before computing crop window.
    - If target domain straddles the source wrap point (e.g. target
      crosses 0° but source is in [0,360]), longitude crop is skipped
      and the full source longitude range is used. This is safe but
      means the fill operates on the full range (slower, same result).
    """
    la0 = float(dlat.min()) - margin
    la1 = float(dlat.max()) + margin
    lat_mask = (slat >= la0) & (slat <= la1)

    # Normalize target lon bounds to source convention
    slo_min, slo_max = float(slon.min()), float(slon.max())
    tlo_min, tlo_max = float(dlon.min()) - margin, float(dlon.max()) + margin

    # Shift target bounds toward source range
    if slo_min >= 0 and tlo_min < 0:
        # Source is [0,360], target has negative lons
        tlo_min += 360.0
        tlo_max += 360.0
    elif slo_min < 0 and tlo_min >= 180:
        # Source is [-180,180], target has lons > 180
        tlo_min -= 360.0
        tlo_max -= 360.0

    # Check if (after normalization) target is contiguous within source
    if tlo_min >= slo_min and tlo_max <= slo_max:
        lon_mask = (slon >= tlo_min) & (slon <= tlo_max)
    elif tlo_min < slo_min and tlo_max > slo_min:
        # Wraps past source minimum — skip lon crop (safe fallback)
        lon_mask = np.ones(slon.shape, dtype=bool)
        print(f"    [crop] target lon [{tlo_min:.1f},{tlo_max:.1f}] wraps "
              f"past source [{slo_min:.1f},{slo_max:.1f}]; no lon crop")
    elif tlo_max > slo_max and tlo_min < slo_max:
        # Wraps past source maximum
        lon_mask = np.ones(slon.shape, dtype=bool)
        print(f"    [crop] target lon [{tlo_min:.1f},{tlo_max:.1f}] wraps "
              f"past source [{slo_min:.1f},{slo_max:.1f}]; no lon crop")
    else:
        # No overlap or fully outside — don't crop
        lon_mask = np.ones(slon.shape, dtype=bool)

    if lat_mask.sum() < 3 or lon_mask.sum() < 3:
        return data, slat, slon

    return data[np.ix_(lat_mask, lon_mask)], slat[lat_mask], slon[lon_mask]


# ===================== Fortran binary I/O ==================================
def fbinrecwrite(f, arr):
    """Write one Fortran unformatted sequential record (little-endian)."""
    raw = np.ascontiguousarray(arr).tobytes()
    n = len(raw)
    f.write(struct.pack('<i', n))
    f.write(raw)
    f.write(struct.pack('<i', n))


def fbinrecread(f, dt, count=1):
    """Read one Fortran unformatted sequential record (little-endian)."""
    h = f.read(4)
    if len(h) < 4:
        raise EOFError("Unexpected EOF in record header")
    n = struct.unpack('<i', h)[0]
    raw = f.read(n)
    if len(raw) < n:
        raise EOFError(f"Short read: {len(raw)}/{n} bytes")
    t = f.read(4)
    n2 = struct.unpack('<i', t)[0]
    if n != n2:
        raise ValueError(f"Fortran record len mismatch: {n} vs {n2}")
    return np.frombuffer(raw, dtype=dt, count=count).copy()


def write_binary_output(fname, nsoil, nlon, nlat, zsoil, soilt, soilw):
    _require_dir(os.path.dirname(fname) or '.', "binary output dir")
    with open(fname, 'wb') as f:
        fbinrecwrite(f, np.array([nsoil], dtype=np.int32))
        fbinrecwrite(f, np.array([nlon], dtype=np.int32))
        fbinrecwrite(f, np.array([nlat], dtype=np.int32))
        fbinrecwrite(f, zsoil.astype(np.float32))
        for i in range(nsoil):
            fbinrecwrite(f, soilt[i].astype(np.float32))
        for i in range(nsoil):
            fbinrecwrite(f, soilw[i].astype(np.float32))
    print(f"  Written binary: {fname}")


def write_netcdf_output(fname, soilt, soilw, zsoil, lat, lon,
                        soilw_attrs=None, extra_vars=None):
    _require_dir(os.path.dirname(fname) or '.', "NetCDF output dir")
    ds = xr.Dataset(
        {'soilt': (['zsoil', 'lat', 'lon'], soilt),
         'soilw': (['zsoil', 'lat', 'lon'], soilw)},
        coords={'zsoil': (['zsoil'], zsoil),
                'lat': (['lat'], lat),
                'lon': (['lon'], lon)},
    )
    ds['zsoil'].attrs['units'] = 'm'
    ds['soilt'].attrs.update({'long_name': 'Soil temperature', 'units': 'K'})
    sw = {'long_name': 'Soil volumetric water content', 'units': 'm3/m3'}
    if soilw_attrs:
        sw.update(soilw_attrs)
    ds['soilw'].attrs.update(sw)
    if extra_vars:
        for k, v in extra_vars.items():
            ds[k] = v
    if os.path.exists(fname):
        os.remove(fname)
    ds.to_netcdf(fname)
    print(f"  Written NetCDF: {fname}")


# ===================== Data readers ========================================
def load_landmask(grid, indir='NC_D'):
    p = os.path.join(indir, f"landmask_{grid}.nc")
    _require_file(p, f"Landmask '{grid}'")
    ds = xr.open_dataset(p)
    for v in ['lat', 'lon', 'LANDMASK']:
        _require_var(ds, v, p)
    la, lo, lm = ds['lat'].values, ds['lon'].values, ds['LANDMASK'].values
    ds.close()
    return la, lo, lm


def parse_date(d):
    y = d // 1000000
    m = (d - y * 1000000) // 10000
    dy = (d - y * 1000000 - m * 10000) // 100
    h = d % 100
    if not (1 <= m <= 12 and 1 <= dy <= 31 and 0 <= h <= 23):
        raise ValueError(f"Bad date {d}: Y={y} M={m} D={dy} H={h}")
    return y, m, dy, h


def find_time_index(ds, date_int):
    """
    Find the time index matching date_int (YYYYMMDDHH).
    Supports numpy.datetime64 and cftime objects.
    cftime: exact match only (no fuzzy search).
    datetime64: exact match first, then nearest within 1 hour.
    """
    if 'time' not in ds.coords and 'time' not in ds.dims:
        raise KeyError(f"No 'time' coordinate. Available: {list(ds.coords)}")

    y, m, d, h = parse_date(date_int)
    times = ds['time'].values
    nt = len(times)
    if nt == 0:
        raise ValueError("Time coordinate is empty.")

    # --- cftime objects ---
    if hasattr(times[0], 'year'):
        for i, t in enumerate(times):
            if (t.year == y and t.month == m
                    and t.day == d and t.hour == h):
                print(f"  Time index {i}: {t} (cftime exact)")
                return i
        samp = [str(times[j]) for j in range(min(3, nt))]
        raise ValueError(
            f"Date {date_int} not found in {nt} cftime steps. "
            f"Samples: {samp}"
        )

    # --- numpy datetime64 ---
    tgt = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:00:00")
    try:
        th = times.astype('datetime64[h]')
        tgh = tgt.astype('datetime64[h]')
        ex = np.where(th == tgh)[0]
        if len(ex) > 0:
            i = int(ex[0])
            print(f"  Time index {i}: {times[i]} (exact)")
            return i
        df = np.abs(th - tgh)
        i = int(np.argmin(df))
        if df[i] <= np.timedelta64(1, 'h'):
            print(f"  Time index {i}: {times[i]} (nearest, diff={df[i]})")
            return i
    except (TypeError, OverflowError):
        try:
            ts = times.astype('datetime64[s]').astype(np.int64)
            tt = tgt.astype('datetime64[s]').astype(np.int64)
            df = np.abs(ts - tt)
            i = int(np.argmin(df))
            if df[i] <= 3600:
                print(f"  Time index {i}: {times[i]} (fallback, "
                      f"diff={df[i]}s)")
                return i
        except Exception:
            pass

    samp = [str(times[j]) for j in range(min(3, nt))]
    raise ValueError(
        f"Date {date_int} not found ({nt} time steps, "
        f"type={type(times[0]).__name__}). Samples: {samp}"
    )


def build_ncar_era5_paths(fp, date_int):
    y, m, _, _ = parse_date(date_int)
    ym = f"{y:04d}{m:02d}"
    ld = calendar.monthrange(y, m)[1]
    dobs = f"{y:04d}{m:02d}0100_{y:04d}{m:02d}{ld:02d}23"
    sp = [os.path.join(fp, ym,
          f"e5.oper.an.sfc.{c}.ll025sc.{dobs}.nc") for c in STL_CODES]
    wp = [os.path.join(fp, ym,
          f"e5.oper.an.sfc.{c}.ll025sc.{dobs}.nc") for c in SWVL_CODES]
    for p in sp + wp:
        _require_file(p, "ERA5 NCAR RDA file")
    return sp, wp


# ===================== Core processing =====================================
def process_soil_layer(raw_2d, src_lat, src_lon, dst_lat, dst_lon,
                       landmask=None, is_moisture=False,
                       do_poisson=True, era5_lsm=None, label="",
                       cyclic_lon=False):
    """
    Process one soil layer:
      flip → (LSM mask) → (poisson fill) → lon flip → interp → (land mask).
    """
    data = raw_2d.astype(np.float64)
    lat_s = src_lat.copy()
    lon_s = src_lon.copy()

    data, lat_s = flip_lat_if_needed(data, lat_s)

    if era5_lsm is not None:
        lsm_w, _ = flip_lat_if_needed(era5_lsm.copy(), src_lat.copy())
        n_masked = int((lsm_w < 0.5).sum())
        data = np.where(lsm_w < 0.5, np.nan, data)
        if label:
            print(f"    [{label}] LSM: {n_masked} ocean→NaN")

    if do_poisson:
        nm = np.isnan(data)
        n_fill = int(nm.sum())
        if nm.any():
            data = fill_poisson(
                data, lat_deg=lat_s, lon_deg=lon_s,
                is_cyclic=cyclic_lon,
                nscan=2000, eps=1e-2, relc=0.6,
            )
            if label:
                print(f"    [{label}] poisson_fill: {n_fill} pts")

    data, lon_s = flip_lon_if_needed(data, lon_s, dst_lon)

    result = interp_to_grid(
        data, lat_s, lon_s, dst_lat, dst_lon, cyclic=cyclic_lon
    )

    if is_moisture and landmask is not None:
        result = apply_land_mask(result, landmask)

    if label:
        print(f"    [{label}] {raw_2d.shape}→{result.shape} "
              f"min={np.nanmin(result):.4f} max={np.nanmax(result):.4f}")

    return result


# ======================== Mode functions ===================================
def run_era5_ncar(grid, date_int, filepath, indir='NC_D', outdir='BIN_D',
                  outdir_nc='NC_D', netcdf_out=True):
    """Mode: era5_ncar (baseline)."""
    dataset = "era5"
    zsoil = ZSOIL_ERA5
    nsoil = NSOIL
    dst_lat, dst_lon, landmask = load_landmask(grid, indir)
    nlat, nlon = len(dst_lat), len(dst_lon)
    print(f"  Grid: {nlon}x{nlat}")

    stl_p, swvl_p = build_ncar_era5_paths(filepath, date_int)
    ds0 = xr.open_dataset(stl_p[0])
    itime = find_time_index(ds0, date_int)
    ds0.close()

    soilt = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        ds = xr.open_dataset(stl_p[n])
        vn = f"STL{n+1}"
        _require_var(ds, vn, stl_p[n])
        raw = ds[vn].values[itime]
        sl, slo = ds['latitude'].values, ds['longitude'].values
        ds.close()
        print(f"  {vn}: min={np.nanmin(raw):.2f} max={np.nanmax(raw):.2f}")
        soilt[n] = process_soil_layer(
            raw, sl, slo, dst_lat, dst_lon,
            do_poisson=False, label=vn,
        )

    soilw = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        ds = xr.open_dataset(swvl_p[n])
        vn = f"SWVL{n+1}"
        _require_var(ds, vn, swvl_p[n])
        raw = ds[vn].values[itime]
        sl, slo = ds['latitude'].values, ds['longitude'].values
        ds.close()
        print(f"  {vn}: min={np.nanmin(raw):.4f} max={np.nanmax(raw):.4f}")
        soilw[n] = process_soil_layer(
            raw, sl, slo, dst_lat, dst_lon,
            landmask=landmask, is_moisture=True,
            do_poisson=True, label=vn, cyclic_lon=True,
        )

    print(f"  soilt: {soilt.min():.2f}..{soilt.max():.2f}  "
          f"soilw: {soilw.min():.4f}..{soilw.max():.4f}")
    _require_dir(outdir)
    write_binary_output(
        os.path.join(outdir, f"soil_init_{date_int}_{grid}_{dataset}.bin"),
        nsoil, nlon, nlat, zsoil, soilt, soilw,
    )
    if netcdf_out:
        _require_dir(outdir_nc)
        write_netcdf_output(
            os.path.join(outdir_nc,
                         f"soil_init_{date_int}_{grid}_{dataset}.nc"),
            soilt, soilw, zsoil, dst_lat, dst_lon,
        )
    return soilt, soilw


def run_era5_ncar_lsm(grid, date_int, filepath, era5_lsm_path,
                       indir='NC_D', outdir='BIN_D', outdir_nc='NC_D',
                       netcdf_out=True):
    """Mode: era5_ncar_lsm — ERA5 + LSM mask + crop + poisson + interp."""
    dataset = "era5_lsm"
    zsoil = ZSOIL_ERA5
    nsoil = NSOIL
    dst_lat, dst_lon, landmask = load_landmask(grid, indir)
    nlat, nlon = len(dst_lat), len(dst_lon)
    print(f"  Grid: {nlon}x{nlat} [LSM]")

    _require_file(era5_lsm_path, "ERA5 LSM")
    ds_l = xr.open_dataset(era5_lsm_path)
    _require_var(ds_l, 'LSM', era5_lsm_path)
    lsm_full = ds_l['LSM'].values
    while lsm_full.ndim > 2:
        lsm_full = lsm_full[0]
    ds_l.close()

    stl_p, swvl_p = build_ncar_era5_paths(filepath, date_int)
    ds0 = xr.open_dataset(stl_p[0])
    itime = find_time_index(ds0, date_int)
    ds0.close()

    def _lsm_layer(raw, sl, slo, lsm, vn):
        d = raw.astype(np.float64)
        d, lat_s = flip_lat_if_needed(d, sl.copy())
        lw, _ = flip_lat_if_needed(lsm.copy(), sl.copy())
        lon_s = slo.copy()

        d = np.where(lw < 0.5, np.nan, d)
        dc, lc, loc = crop_to_domain(
            d, lat_s, lon_s, dst_lat, dst_lon, margin=1.0
        )
        print(f"    [{vn}] crop {d.shape}→{dc.shape}")

        nm = np.isnan(dc)
        if nm.any():
            dc = fill_poisson(
                dc, lat_deg=lc, lon_deg=loc, is_cyclic=False,
                nscan=6000, eps=1e-2, relc=0.6,
            )
            print(f"    [{vn}] poisson_fill: {int(nm.sum())} pts")

        dc, loc = flip_lon_if_needed(dc, loc, dst_lon)
        return interp_to_grid(dc, lc, loc, dst_lat, dst_lon, cyclic=False)

    soilt = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        ds = xr.open_dataset(stl_p[n])
        vn = f"STL{n+1}"
        _require_var(ds, vn, stl_p[n])
        raw = ds[vn].values[itime]
        sl, slo = ds['latitude'].values, ds['longitude'].values
        ds.close()
        soilt[n] = _lsm_layer(raw, sl, slo, lsm_full, vn)
        print(f"  {vn}(LSM): {soilt[n].min():.2f}..{soilt[n].max():.2f}")

    soilw = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        ds = xr.open_dataset(swvl_p[n])
        vn = f"SWVL{n+1}"
        _require_var(ds, vn, swvl_p[n])
        raw = ds[vn].values[itime]
        sl, slo = ds['latitude'].values, ds['longitude'].values
        ds.close()
        soilw[n] = apply_land_mask(
            _lsm_layer(raw, sl, slo, lsm_full, vn), landmask
        )
        print(f"  {vn}(LSM): {soilw[n].min():.4f}..{soilw[n].max():.4f}")

    _require_dir(outdir)
    write_binary_output(
        os.path.join(outdir, f"soil_init_{date_int}_{grid}_{dataset}.bin"),
        nsoil, nlon, nlat, zsoil, soilt, soilw,
    )
    if netcdf_out:
        _require_dir(outdir_nc)
        write_netcdf_output(
            os.path.join(outdir_nc,
                         f"soil_init_{date_int}_{grid}_{dataset}.nc"),
            soilt, soilw, zsoil, dst_lat, dst_lon,
        )
    return soilt, soilw


def _load_delta_skt(delta_file):
    _require_file(delta_file, "Delta SKT")
    ds = xr.open_dataset(delta_file)
    vn = None
    for c in ['skt_delta', 'skt', 'SKT', 'dts', 'DTS']:
        if c in ds.data_vars:
            vn = c
            break
    if vn is None:
        av = list(ds.data_vars)
        ds.close()
        raise KeyError(f"No recognized variable in delta file. Have: {av}")
    print(f"  Delta var: '{vn}'")
    d = ds[vn].values
    while d.ndim > 2:
        d = d[0]
    latn = _resolve_coord(ds, ['latitude', 'lat'], "latitude")
    lonn = _resolve_coord(ds, ['longitude', 'lon'], "longitude")
    la, lo = ds[latn].values, ds[lonn].values
    ds.close()
    return d, la, lo


def run_era5_ncar_delta(grid, date_int, filepath, delta_file,
                        dataset_tag="era5_DELTA",
                        indir='NC_D', outdir='BIN_D', outdir_nc='NC_D',
                        netcdf_out=True):
    """Mode: era5_ncar_delta (PGW)."""
    zsoil = ZSOIL_ERA5
    nsoil = NSOIL
    dst_lat, dst_lon, landmask = load_landmask(grid, indir)
    nlat, nlon = len(dst_lat), len(dst_lon)
    print(f"  Grid: {nlon}x{nlat} [DELTA tag={dataset_tag}]")

    sr, dla, dlo = _load_delta_skt(delta_file)
    sr, dla = flip_lat_if_needed(sr, dla)
    sr, dlo = flip_lon_if_needed(sr, dlo, dst_lon)
    skt_d = interp_to_grid(sr, dla, dlo, dst_lat, dst_lon)
    print(f"  SKT delta: {np.nanmin(skt_d):.3f}..{np.nanmax(skt_d):.3f}")

    stl_p, swvl_p = build_ncar_era5_paths(filepath, date_int)
    ds0 = xr.open_dataset(stl_p[0])
    itime = find_time_index(ds0, date_int)
    ds0.close()

    soilt = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        ds = xr.open_dataset(stl_p[n])
        vn = f"STL{n+1}"
        _require_var(ds, vn, stl_p[n])
        raw = ds[vn].values[itime]
        sl, slo = ds['latitude'].values, ds['longitude'].values
        ds.close()
        soilt[n] = process_soil_layer(
            raw, sl, slo, dst_lat, dst_lon,
            do_poisson=False, label=vn,
        )
        soilt[n] += skt_d
        print(f"  {vn}+Δ: {soilt[n].min():.2f}..{soilt[n].max():.2f}")

    soilw = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        ds = xr.open_dataset(swvl_p[n])
        vn = f"SWVL{n+1}"
        _require_var(ds, vn, swvl_p[n])
        raw = ds[vn].values[itime]
        sl, slo = ds['latitude'].values, ds['longitude'].values
        ds.close()
        soilw[n] = process_soil_layer(
            raw, sl, slo, dst_lat, dst_lon,
            landmask=landmask, is_moisture=True,
            do_poisson=True, label=vn, cyclic_lon=True,
        )

    _require_dir(outdir)
    write_binary_output(
        os.path.join(outdir,
                     f"soil_init_{date_int}_{grid}_{dataset_tag}.bin"),
        nsoil, nlon, nlat, zsoil, soilt, soilw,
    )
    if netcdf_out:
        _require_dir(outdir_nc)
        write_netcdf_output(
            os.path.join(outdir_nc,
                         f"soil_init_{date_int}_{grid}_{dataset_tag}.nc"),
            soilt, soilw, zsoil, dst_lat, dst_lon,
        )
    return soilt, soilw


def run_era5_notncar(grid, date_int, filedata,
                     indir='NC_D', outdir='BIN_D', outdir_nc='NC_D',
                     netcdf_out=True):
    """Mode: era5_notncar — user-downloaded single ERA5 file."""
    dataset = "era5"
    zsoil = ZSOIL_ERA5
    nsoil = NSOIL
    _require_file(filedata, "ERA5 notncar data file")
    dst_lat, dst_lon, landmask = load_landmask(grid, indir)
    nlat, nlon = len(dst_lat), len(dst_lon)
    print(f"  Grid: {nlon}x{nlat} [notncar]")

    ds = xr.open_dataset(filedata)
    latn = _resolve_coord(ds, ['latitude', 'lat'], "latitude")
    lonn = _resolve_coord(ds, ['longitude', 'lon'], "longitude")
    itime = find_time_index(ds, date_int)

    for n in range(nsoil):
        _require_var(ds, f"stl{n+1}", filedata)
        _require_var(ds, f"swvl{n+1}", filedata)
    v1 = ds['stl1']
    if v1.ndim < 3:
        ds.close()
        raise ValueError(f"stl1 ndim={v1.ndim}, need >=3 (time, lat, lon)")
    if np.issubdtype(v1.dtype, np.integer):
        ds.close()
        raise TypeError(f"stl1 dtype={v1.dtype} — CF decode may have failed")

    sl = ds[latn].values
    slo = ds[lonn].values.copy()
    if np.max(slo) < 0:
        slo += 360.0
        print("  Shifted longitude +360")

    soilt = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        vn = f"stl{n+1}"
        raw = ds[vn].values[itime]
        soilt[n] = process_soil_layer(
            raw, sl, slo, dst_lat, dst_lon,
            do_poisson=False, label=vn,
        )

    soilw = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        vn = f"swvl{n+1}"
        raw = ds[vn].values[itime]
        soilw[n] = process_soil_layer(
            raw, sl, slo, dst_lat, dst_lon,
            landmask=landmask, is_moisture=True,
            do_poisson=False, label=vn,
        )
    ds.close()

    _require_dir(outdir)
    write_binary_output(
        os.path.join(outdir, f"soil_init_{date_int}_{grid}_{dataset}.bin"),
        nsoil, nlon, nlat, zsoil, soilt, soilw,
    )
    if netcdf_out:
        _require_dir(outdir_nc)
        write_netcdf_output(
            os.path.join(outdir_nc,
                         f"soil_init_{date_int}_{grid}_{dataset}.nc"),
            soilt, soilw, zsoil, dst_lat, dst_lon,
        )
    return soilt, soilw


def run_era5_wetness(grid, date_int, filepath, porosity_file,
                     indir='NC_D', outdir='BIN_D', outdir_nc='NC_D',
                     netcdf_out=True):
    """Mode: era5_wetness — swvl/porosity → negative wetness (gSAM)."""
    dataset = "era5"
    zsoil = ZSOIL_ERA5
    nsoil = NSOIL
    _require_file(porosity_file, "Porosity file")
    dst_lat, dst_lon, landmask = load_landmask(grid, indir)
    nlat, nlon = len(dst_lat), len(dst_lon)
    print(f"  Grid: {nlon}x{nlat} [wetness]")

    ds_p = xr.open_dataset(porosity_file)
    _require_var(ds_p, 'SLT', porosity_file)
    slt_r = ds_p['SLT'].values
    while slt_r.ndim > 2:
        slt_r = slt_r[0]
    pln = _resolve_coord(ds_p, ['latitude', 'lat'], "latitude")
    pon = _resolve_coord(ds_p, ['longitude', 'lon'], "longitude")
    plr, plo = ds_p[pln].values, ds_p[pon].values
    ds_p.close()

    slt, pla = flip_lat_if_needed(slt_r, plr.copy())
    poro = np.full_like(slt, POROSITY_DEFAULT, dtype=np.float64)
    for st, pv in POROSITY_TABLE.items():
        poro = np.where(slt == st, pv, poro)
    print(f"  Porosity: {poro.min():.3f}..{poro.max():.3f} "
          f"shape={poro.shape}")

    stl_p, swvl_p = build_ncar_era5_paths(filepath, date_int)
    ds0 = xr.open_dataset(stl_p[0])
    itime = find_time_index(ds0, date_int)
    e5la, e5lo = ds0['latitude'].values, ds0['longitude'].values
    ds0.close()
    e5la_sn = e5la.copy()
    if e5la_sn[0] > e5la_sn[-1]:
        e5la_sn = e5la_sn[::-1]

    if not _grids_aligned(pla, plo, e5la_sn, e5lo):
        print(f"  Porosity grid != ERA5 → interpolating")
        poro = interp_to_grid(
            poro, pla, plo, e5la_sn, e5lo
        ).astype(np.float64)
        poro = np.where(poro < 0.05, POROSITY_DEFAULT, poro)
        print(f"  Porosity(interp): {poro.min():.3f}..{poro.max():.3f}")
    else:
        print(f"  Porosity grid matches ERA5")

    soilt = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        ds = xr.open_dataset(stl_p[n])
        vn = f"STL{n+1}"
        _require_var(ds, vn, stl_p[n])
        raw = ds[vn].values[itime]
        sl, slo = ds['latitude'].values, ds['longitude'].values
        ds.close()
        soilt[n] = process_soil_layer(
            raw, sl, slo, dst_lat, dst_lon,
            do_poisson=False, label=vn,
        )

    soilw = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        ds = xr.open_dataset(swvl_p[n])
        vn = f"SWVL{n+1}"
        _require_var(ds, vn, swvl_p[n])
        raw = ds[vn].values[itime]
        sl = ds['latitude'].values
        slo = ds['longitude'].values
        ds.close()
        raw_sn, lat_sn = flip_lat_if_needed(raw.copy(), sl.copy())
        raw_sn = raw_sn / poro
        soilw[n] = process_soil_layer(
            raw_sn, lat_sn, slo, dst_lat, dst_lon,
            landmask=landmask, is_moisture=True,
            do_poisson=False, label=f"{vn}/poro",
        )

    soilw = -soilw
    print(f"  soilw(neg wetness): {soilw.min():.4f}..{soilw.max():.4f}")
    _require_dir(outdir)
    write_binary_output(
        os.path.join(outdir,
                     f"soilwetness_init_{date_int}_{grid}_{dataset}.bin"),
        nsoil, nlon, nlat, zsoil, soilt, soilw,
    )
    if netcdf_out:
        _require_dir(outdir_nc)
        write_netcdf_output(
            os.path.join(outdir_nc,
                         f"soilwetness_init_{date_int}_{grid}_{dataset}.nc"),
            soilt, soilw, zsoil, dst_lat, dst_lon,
            soilw_attrs={
                'long_name': 'Soil wetness (neg, gSAM convention)',
                'units': '1',
                'comment': '-(SWVL/porosity)',
            },
        )
    return soilt, soilw


def run_gfs(grid, date_int, gfs_file,
            indir='NC_D', outdir='BIN_D', outdir_nc='NC_D',
            netcdf_out=True):
    """
    Mode: gfs — from GFS GRIB2 file.

    Time handling:
    - The original NCL script opens a single GFS f000 file (named by date),
      which contains one forecast step — no time selection needed.
    - If the file does contain a time dimension (4D data), this function
      uses date_int to select the correct time step via find_time_index.
    - If there is no time dimension (3D data, the common case), the data
      is used as-is and date_int is used only for output naming.
    """
    dataset = "GFS"
    zsoil = ZSOIL_GFS
    nsoil = NSOIL
    _require_file(gfs_file, "GFS data file")
    dst_lat, dst_lon, landmask = load_landmask(grid, indir)
    nlat, nlon = len(dst_lat), len(dst_lon)
    print(f"  Grid: {nlon}x{nlat} [GFS]")

    tsoil_raw = None
    soilw_raw = None
    gfs_lat = None
    gfs_lon = None

    # --- Method 1: cfgrib ---
    try:
        ds_t = xr.open_dataset(
            gfs_file, engine='cfgrib',
            backend_kwargs={'filter_by_keys': {
                'typeOfLevel': 'depthBelowLandLayer',
                'shortName': 'st',
            }},
        )
        tv = list(ds_t.data_vars)[0]
        print(f"  cfgrib: tsoil var='{tv}', "
              f"dims={ds_t[tv].dims}, shape={ds_t[tv].shape}")
        tsoil_raw = ds_t[tv].values
        gfs_lat = ds_t['latitude'].values
        gfs_lon = ds_t['longitude'].values
        ds_t.close()

        ds_w = xr.open_dataset(
            gfs_file, engine='cfgrib',
            backend_kwargs={'filter_by_keys': {
                'typeOfLevel': 'depthBelowLandLayer',
                'shortName': 'soilw',
            }},
        )
        sv = list(ds_w.data_vars)[0]
        soilw_raw = ds_w[sv].values
        ds_w.close()
    except Exception as e:
        print(f"  cfgrib failed ({e}), trying NCL-style variable names")
        ds = xr.open_dataset(gfs_file)
        for v in ['TSOIL_P0_2L106_GLL0', 'TSOIL_P0_L106_GLL0']:
            if v in ds.data_vars:
                tsoil_raw = ds[v].values
                print(f"  GFS tsoil: '{v}', shape={tsoil_raw.shape}")
                break
        for v in ['SOILW_P0_2L106_GLL0', 'SOILW_P0_L106_GLL0']:
            if v in ds.data_vars:
                soilw_raw = ds[v].values
                break
        ln = _resolve_coord(ds, ['lat_0', 'latitude', 'lat'], "latitude")
        on = _resolve_coord(ds, ['lon_0', 'longitude', 'lon'], "longitude")
        gfs_lat = ds[ln].values
        gfs_lon = ds[on].values
        ds.close()

    if tsoil_raw is None or soilw_raw is None:
        raise RuntimeError(f"Cannot load soil variables from {gfs_file}")

    # --- Time selection for GFS ---
    # If data has a leading time dimension, select the correct time step.
    if tsoil_raw.ndim == 4:
        # Shape is (time, depth, lat, lon).
        # The NCL original uses single-step f000 files so this is atypical.
        # Try to match date_int against the file's time coordinate.
        ntime_gfs = tsoil_raw.shape[0]
        itime_gfs = None

        # Try to open with default engine to read time coord
        for engine in [None, 'cfgrib']:
            try:
                kw = {'engine': engine} if engine else {}
                ds_check = xr.open_dataset(gfs_file, **kw)
                time_cands = ['time', 'initial_time0_hours',
                              'forecast_time0', 'ref_time']
                for tc in time_cands:
                    if tc in ds_check.coords or tc in ds_check.dims:
                        itime_gfs = find_time_index(ds_check, date_int)
                        break
                ds_check.close()
                if itime_gfs is not None:
                    break
            except Exception:
                continue

        if itime_gfs is not None:
            print(f"  GFS time selection: index {itime_gfs} "
                  f"of {ntime_gfs} steps")
            tsoil_raw = tsoil_raw[itime_gfs]
            soilw_raw = soilw_raw[itime_gfs]
        else:
            if ntime_gfs == 1:
                print(f"  GFS: 4D with 1 time step; using it")
                tsoil_raw = tsoil_raw[0]
                soilw_raw = soilw_raw[0]
            else:
                raise ValueError(
                    f"GFS file has {ntime_gfs} time steps but could not "
                    f"match date {date_int} to any time coordinate. "
                    f"Cannot determine which step to use."
                )
    elif tsoil_raw.ndim == 3:
        # (depth, lat, lon) — no time dim, typical for f000 files
        print(f"  GFS: no time dimension (single-step file)")
    elif tsoil_raw.ndim == 2:
        raise ValueError(
            f"GFS data is 2D {tsoil_raw.shape} — expected "
            f"(depth,lat,lon) or (time,depth,lat,lon)"
        )
    else:
        raise ValueError(
            f"GFS data has unexpected {tsoil_raw.ndim}D shape "
            f"{tsoil_raw.shape}"
        )

    if tsoil_raw.shape[0] != nsoil:
        raise ValueError(
            f"GFS first dimension is {tsoil_raw.shape[0]}, "
            f"expected {nsoil} soil layers"
        )
    print(f"  GFS final: tsoil={tsoil_raw.shape} soilw={soilw_raw.shape}")

    soilt = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        soilt[n] = process_soil_layer(
            tsoil_raw[n], gfs_lat, gfs_lon, dst_lat, dst_lon,
            do_poisson=True, label=f"TSOIL[{n}]", cyclic_lon=True,
        )

    soilw = np.zeros((nsoil, nlat, nlon), dtype=np.float32)
    for n in range(nsoil):
        soilw[n] = process_soil_layer(
            soilw_raw[n], gfs_lat, gfs_lon, dst_lat, dst_lon,
            landmask=landmask, is_moisture=True,
            do_poisson=True, label=f"SOILW[{n}]", cyclic_lon=True,
        )

    _require_dir(outdir)
    write_binary_output(
        os.path.join(outdir, f"soil_init_{date_int}_{grid}_{dataset}.bin"),
        nsoil, nlon, nlat, zsoil, soilt, soilw,
    )
    if netcdf_out:
        _require_dir(outdir_nc)
        write_netcdf_output(
            os.path.join(outdir_nc,
                         f"soil_init_{date_int}_{grid}_{dataset}.nc"),
            soilt, soilw, zsoil, dst_lat, dst_lon,
        )
    return soilt, soilw


def run_cesm_cc(grid, base_bin_file, output_bin_file,
                cesm_current_file, cesm_future_file,
                landfrac_file=None, nday=122,
                netcdf_out=True, outdir_nc='NC_D', indir='NC_D'):
    """Mode: cesm_cc — overlay CESM ΔTS onto existing soil init."""
    dataset = "era5_CESM2090s"
    _require_file(base_bin_file, "base binary file")
    _require_file(cesm_current_file, "CESM current file")
    _require_file(cesm_future_file, "CESM future file")
    if not output_bin_file:
        raise ValueError("--output_bin is required for cesm_cc mode")

    dst_lat, dst_lon, _ = load_landmask(grid, indir)
    gnlat, gnlon = len(dst_lat), len(dst_lon)
    print(f"  Grid: {gnlon}x{gnlat} [CESM CC]")

    ds1 = xr.open_dataset(cesm_current_file)
    ds2 = xr.open_dataset(cesm_future_file)
    _require_var(ds1, 'TS', cesm_current_file)
    _require_var(ds2, 'TS', cesm_future_file)

    t1f = ds1['TS'].values
    t2f = ds2['TS'].values
    ti = nday * 4 - 1
    if ti < 0 or ti >= t1f.shape[0]:
        raise IndexError(
            f"nday={nday} → index {ti} out of bounds "
            f"(current file has {t1f.shape[0]} steps)"
        )
    if ti >= t2f.shape[0]:
        raise IndexError(
            f"nday={nday} → index {ti} out of bounds "
            f"(future file has {t2f.shape[0]} steps)"
        )

    ts1 = t1f[ti]
    ts2 = t2f[ti]
    cesm_lat = ds1['lat'].values
    cesm_lon = ds1['lon'].values
    ds1.close()
    ds2.close()

    # --- landfrac resolution ---
    if landfrac_file:
        _require_file(landfrac_file, "landfrac (--landfrac)")
    else:
        landfrac_file = os.path.join(
            os.path.dirname(cesm_current_file), "..", "landfrac.nc"
        )
        if not os.path.isfile(landfrac_file):
            raise FileNotFoundError(
                f"landfrac not found at {landfrac_file}; use --landfrac"
            )
        print(f"  Using default landfrac: {landfrac_file}")

    lf_ds = xr.open_dataset(landfrac_file)
    _require_var(lf_ds, 'LANDFRAC', landfrac_file)
    lf_vals = lf_ds['LANDFRAC'].values
    # Squeeze extra dims (e.g. leading time dim)
    while lf_vals.ndim > 2:
        lf_vals = lf_vals[0]

    # --- Grid consistency: shape ---
    if lf_vals.shape != ts1.shape:
        lf_ds.close()
        raise ValueError(
            f"LANDFRAC shape {lf_vals.shape} != TS shape {ts1.shape}"
        )

    # --- Grid consistency: coordinate values ---
    try:
        lf_lat = lf_ds['lat'].values
        lf_lon = lf_ds['lon'].values
        if lf_lat.shape != cesm_lat.shape or lf_lon.shape != cesm_lon.shape:
            warnings.warn(
                f"LANDFRAC coord shapes (lat={lf_lat.shape}, lon={lf_lon.shape})"
                f" differ from TS coord shapes (lat={cesm_lat.shape}, "
                f"lon={cesm_lon.shape}). Proceeding on shape match only."
            )
        elif not _grids_aligned(lf_lat, lf_lon, cesm_lat, cesm_lon, atol=0.01):
            warnings.warn(
                f"LANDFRAC lat/lon values differ from TS lat/lon "
                f"(max Δlat={np.max(np.abs(lf_lat - cesm_lat)):.4f}°, "
                f"max Δlon={np.max(np.abs(lf_lon - cesm_lon)):.4f}°). "
                f"Proceeding because shapes match, but results may be wrong."
            )
        else:
            print("  LANDFRAC grid matches TS grid (shape + coords)")
    except KeyError:
        print("  LANDFRAC has no lat/lon coords; shape-only validation")

    lf_ds.close()

    cesm_landmask = np.where(lf_vals == 0, 0.0, 1.0)
    ts1 = np.where(cesm_landmask > 0, ts1, np.nan)
    ts1 = fill_poisson(ts1, lat_deg=cesm_lat)
    ts2 = np.where(cesm_landmask > 0, ts2, np.nan)
    ts2 = fill_poisson(ts2, lat_deg=cesm_lat)
    dts = ts2 - ts1

    dts_interp = interp_to_grid(dts, cesm_lat, cesm_lon, dst_lat, dst_lon)
    print(f"  DTS: {np.nanmin(dts_interp):.2f}..{np.nanmax(dts_interp):.2f}")

    # --- Read base binary ---
    with open(base_bin_file, 'rb') as f:
        nsoil_bin = fbinrecread(f, np.int32, 1)[0]
        nlon_bin = fbinrecread(f, np.int32, 1)[0]
        nlat_bin = fbinrecread(f, np.int32, 1)[0]
        zsoil = fbinrecread(f, np.float32, nsoil_bin)

        if nlat_bin != gnlat or nlon_bin != gnlon:
            raise ValueError(
                f"base_bin grid ({nlat_bin},{nlon_bin}) != "
                f"landmask grid ({gnlat},{gnlon})"
            )

        soilt = np.zeros((nsoil_bin, nlat_bin, nlon_bin), dtype=np.float32)
        for i in range(nsoil_bin):
            soilt[i] = fbinrecread(
                f, np.float32, nlat_bin * nlon_bin
            ).reshape(nlat_bin, nlon_bin)
        soilw = np.zeros((nsoil_bin, nlat_bin, nlon_bin), dtype=np.float32)
        for i in range(nsoil_bin):
            soilw[i] = fbinrecread(
                f, np.float32, nlat_bin * nlon_bin
            ).reshape(nlat_bin, nlon_bin)

    for i in range(nsoil_bin):
        soilt[i] += dts_interp
        print(f"  tsoil+dts[{i}]: {soilt[i].min():.2f}..{soilt[i].max():.2f}")

    write_binary_output(
        output_bin_file, nsoil_bin, nlon_bin, nlat_bin, zsoil, soilt, soilw
    )

    if netcdf_out:
        _require_dir(outdir_nc)
        nc_name = os.path.join(outdir_nc, f"dts_{grid}_{dataset}.nc")
        ds_out = xr.Dataset(
            {'dts_grid': (['lat', 'lon'], dts_interp)},
            coords={'lat': dst_lat, 'lon': dst_lon},
        )
        ds_out['dts_grid'].attrs = {'long_name': 'DTS', 'units': 'K'}
        if os.path.exists(nc_name):
            os.remove(nc_name)
        ds_out.to_netcdf(nc_name)
        print(f"  Written {nc_name}")

    return soilt, soilw


# ========================== CLI ============================================
def main():
    global _INTERP_BACKEND, _FILL_BACKEND, _FILL_INTERP_NAN

    p = argparse.ArgumentParser(
        description="Soil init for SAM/gSAM (v4, NCL-fidelity)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Modes: auto era5_ncar era5_ncar_lsm era5_ncar_delta "
               "era5_notncar era5_wetness gfs cesm_cc",
    )
    p.add_argument('--mode', required=True, choices=[
        'auto', 'era5_ncar', 'era5_ncar_lsm', 'era5_ncar_delta',
        'era5_notncar', 'era5_wetness', 'gfs', 'cesm_cc',
    ])
    p.add_argument('--grid', required=True)
    p.add_argument('--date', type=int, default=None,
                   help='YYYYMMDDHH (not needed for cesm_cc)')
    p.add_argument('--filepath', default='')
    p.add_argument('--filedata', default='')
    p.add_argument('--delta_file', default='')
    p.add_argument('--dataset_tag', default='era5_DELTA')
    p.add_argument('--era5_lsm_path', default='')
    p.add_argument('--porosity_file', default='')
    p.add_argument('--base_bin', default='')
    p.add_argument('--output_bin', default='')
    p.add_argument('--cesm_current', default='')
    p.add_argument('--cesm_future', default='')
    p.add_argument('--landfrac', default='')
    p.add_argument('--nday', type=int, default=122)
    p.add_argument('--indir', default='NC_D')
    p.add_argument('--outdir', default='BIN_D')
    p.add_argument('--outdir_nc', default='NC_D')
    p.add_argument('--no_netcdf', action='store_true')
    p.add_argument('--interp_backend', default='python_ncl',
                   choices=['python_ncl', 'python_ncl_fast', 'scipy'],
                   help='Interpolation backend')
    p.add_argument('--fill_backend', default='python_ncl',
                   choices=['python_ncl', 'python_ncl_fast', 'scipy'],
                   help='Poisson fill backend')
    p.add_argument('--fill_interp_nan', action='store_true',
                   help='Fill post-interpolation NaN with nearest neighbor '
                        '(NOT NCL-like; off by default)')

    a = p.parse_args()
    _INTERP_BACKEND = a.interp_backend
    _FILL_BACKEND = a.fill_backend
    _FILL_INTERP_NAN = a.fill_interp_nan
    nc_out = not a.no_netcdf

    print(f"  Backends: interp={_INTERP_BACKEND}, fill={_FILL_BACKEND}"
          + (", fill_interp_nan=ON" if _FILL_INTERP_NAN else ""))

    # --- auto mode ---
    if a.mode == 'auto':
        if is_on_derecho():
            a.mode = 'era5_ncar'
            if not a.filepath:
                a.filepath = _DERECHO_DEFAULT_FILEPATH
            if not os.path.isdir(a.filepath):
                print(f"ERROR: Derecho detected but {a.filepath} missing")
                sys.exit(1)
            print(f"  Derecho → era5_ncar")
        else:
            a.mode = 'era5_notncar'
            print("  Not Derecho → era5_notncar")
            if not a.filedata:
                print("ERROR: --filedata required")
                sys.exit(1)

    if a.mode != 'cesm_cc' and a.date is None:
        print(f"ERROR: --date required for mode '{a.mode}'")
        sys.exit(1)

    print(f"Mode: {a.mode}  Grid: {a.grid}"
          + (f"  Date: {a.date}" if a.date else ""))

    # --- Dispatch ---
    if a.mode == 'era5_ncar':
        if not a.filepath:
            print("ERROR: --filepath required"); sys.exit(1)
        run_era5_ncar(a.grid, a.date, a.filepath,
                      a.indir, a.outdir, a.outdir_nc, nc_out)
    elif a.mode == 'era5_ncar_lsm':
        if not a.filepath or not a.era5_lsm_path:
            print("ERROR: --filepath and --era5_lsm_path required")
            sys.exit(1)
        run_era5_ncar_lsm(a.grid, a.date, a.filepath, a.era5_lsm_path,
                          a.indir, a.outdir, a.outdir_nc, nc_out)
    elif a.mode == 'era5_ncar_delta':
        if not a.filepath or not a.delta_file:
            print("ERROR: --filepath and --delta_file required")
            sys.exit(1)
        run_era5_ncar_delta(a.grid, a.date, a.filepath, a.delta_file,
                            a.dataset_tag, a.indir, a.outdir,
                            a.outdir_nc, nc_out)
    elif a.mode == 'era5_notncar':
        if not a.filedata:
            print("ERROR: --filedata required"); sys.exit(1)
        run_era5_notncar(a.grid, a.date, a.filedata,
                         a.indir, a.outdir, a.outdir_nc, nc_out)
    elif a.mode == 'era5_wetness':
        if not a.filepath or not a.porosity_file:
            print("ERROR: --filepath and --porosity_file required")
            sys.exit(1)
        run_era5_wetness(a.grid, a.date, a.filepath, a.porosity_file,
                         a.indir, a.outdir, a.outdir_nc, nc_out)
    elif a.mode == 'gfs':
        if not a.filedata:
            print("ERROR: --filedata required"); sys.exit(1)
        run_gfs(a.grid, a.date, a.filedata,
                a.indir, a.outdir, a.outdir_nc, nc_out)
    elif a.mode == 'cesm_cc':
        if not all([a.base_bin, a.output_bin,
                    a.cesm_current, a.cesm_future]):
            print("ERROR: --base_bin, --output_bin, --cesm_current, "
                  "--cesm_future all required")
            sys.exit(1)
        run_cesm_cc(a.grid, a.base_bin, a.output_bin,
                    a.cesm_current, a.cesm_future,
                    landfrac_file=a.landfrac or None,
                    nday=a.nday, netcdf_out=nc_out,
                    outdir_nc=a.outdir_nc, indir=a.indir)

    print("Done.")


if __name__ == '__main__':
    main()
