#!/usr/bin/env python3
"""
make_2D_init_soil.py (v7 — Simplified modes & unified path resolution)
======================================================================
Unified Python replacement for NCL soil initialization scripts.

v7 Key Changes:
- SIMPLIFIED: Modes reduced to era5, era5_lsm, era5_delta, era5_wetness, gfs, cesm_cc
- UNIFIED: Single era5 mode handles both NCAR directory and single-file inputs
- IMPROVED: Startup dependency checking with clear install instructions
- IMPROVED: Path priority: user-provided → Derecho default → error
- IMPROVED: Separate validation for input paths (must exist) vs output paths (parent must exist)

Modes:
  era5          Standard ERA5 soil initialization
  era5_lsm      ERA5 with LSM mask and extended fill
  era5_delta    ERA5 + climate delta (PGW)
  era5_wetness  ERA5 → soil wetness (gSAM format)
  gfs           GFS GRIB2 data
  cesm_cc       CESM climate change overlay

Path Resolution Priority:
  1. User-provided path (--filepath or --filedata) → must exist
  2. If not provided AND on Derecho → use default path
  3. If not provided AND not on Derecho → error
"""

import argparse
import calendar
import importlib.util
import os
import socket
import struct
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, List, Any

import numpy as np

# =============================================================================
# SECTION 1: Fixed Default Paths (DO NOT MODIFY)
# =============================================================================

_DERECHO_DEFAULT_FILEPATH = (
    "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc"
)

_DERECHO_DEFAULT_LSM_PATH = (
    "/glade/campaign/collections/rda/data/d633000/e5.oper.invariant/197901/"
    "e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc"
)

# Additional default paths (can be modified if needed)
_DERECHO_DEFAULT_POROSITY_PATH = None  # No reliable default; user must provide


# =============================================================================
# SECTION 2: Constants
# =============================================================================

ZSOIL_ERA5 = np.array([0.035, 0.175, 0.64, 1.945], dtype=np.float32)
ZSOIL_GFS = np.array([0.05, 0.25, 0.7, 1.5], dtype=np.float32)
NSOIL = 4

STL_CODES = ["128_139_stl1", "128_170_stl2", "128_183_stl3", "128_236_stl4"]
SWVL_CODES = ["128_039_swvl1", "128_040_swvl2", "128_041_swvl3", "128_042_swvl4"]

POROSITY_TABLE = {1: 0.403, 2: 0.439, 3: 0.430, 4: 0.520,
                  5: 0.614, 6: 0.766, 7: 0.472}
POROSITY_DEFAULT = 0.4

# HRRR soil layer depths (mid-layer, metres).
# HRRR (RUC LSM) native 9-level depths differ from Noah LSM 4-level depths.
# When HRRR GRIB2 is filtered to the 4 standard depthBelowLandLayer records
# that WRF typically outputs, the layer boundaries are 0–0.1, 0.1–0.4,
# 0.4–1.0, 1.0–2.0 m, giving these mid-points:
ZSOIL_HRRR = np.array([0.05, 0.25, 0.70, 1.50], dtype=np.float32)

# Candidate variable names for HRRR soil fields (checked in order).
# cfgrib typically exposes 'st' / 'soilw'; NCL-decoded GRIB may use longer names.
_HRRR_TSOIL_CANDIDATES = ['st', 'ST', 'tsoil', 'TSOIL', 'soilt', 'SOILT', 't']
_HRRR_SOILW_CANDIDATES = ['soilw', 'SOILW', 'sm', 'SM', 'vsw', 'VSW', 'SOIL_M']

# Known HRRR 4-layer boundary pairs (top_m, bot_m) used to derive mid-layer
# depths.  Order is shallowest → deepest.
_HRRR_KNOWN_LAYER_BOUNDS = [
    (0.0, 0.1),
    (0.1, 0.4),
    (0.4, 1.0),
    (1.0, 2.0),
]


# =============================================================================
# SECTION 3: Mode Definitions
# =============================================================================

class Mode(Enum):
    """Supported processing modes."""
    ERA5 = "era5"
    ERA5_LSM = "era5_lsm"
    ERA5_DELTA = "era5_delta"
    ERA5_WETNESS = "era5_wetness"
    GFS = "gfs"
    HRRR = "hrrr"
    CESM_CC = "cesm_cc"

# Legacy mode mapping for backward compatibility.
# These are accepted by normalize_mode(), but are not exposed in CLI help.
_LEGACY_MODE_MAP = {
    "era5_ncar": Mode.ERA5,
    "era5_notncar": Mode.ERA5,
    "era5_ncar_lsm": Mode.ERA5_LSM,
    "era5_ncar_delta": Mode.ERA5_DELTA,
}


def normalize_mode(mode_str: str) -> Mode:
    """
    Normalize a mode string to the canonical Mode enum.

    Public interface supports only the simplified modes. A small set of legacy
    names is accepted for backward compatibility, but is not advertised.
    """
    mode_lower = mode_str.lower().strip()

    if mode_lower == 'auto':
        raise ValueError(
            "Mode 'auto' is no longer supported. "
            "Please choose one of: era5, era5_lsm, era5_delta, era5_wetness, gfs, hrrr, cesm_cc."
        )

    for m in Mode:
        if m.value == mode_lower:
            return m

    if mode_lower in _LEGACY_MODE_MAP:
        mapped = _LEGACY_MODE_MAP[mode_lower]
        print(f"  NOTE: Legacy mode '{mode_str}' mapped to '{mapped.value}'.")
        return mapped

    valid_modes = [m.value for m in Mode]
    raise ValueError(
        f"Unknown mode: '{mode_str}'\n"
        f"Valid modes: {', '.join(valid_modes)}"
    )



# =============================================================================
# SECTION 4: Dependency Checking
# =============================================================================

@dataclass
class DependencyStatus:
    """Status of a dependency check."""
    name: str
    available: bool
    required_for: List[str] = field(default_factory=list)
    pip_install: str = ""
    conda_install: str = ""


def check_module_available(module_name: str) -> bool:
    """Check if a Python module is available."""
    return importlib.util.find_spec(module_name) is not None


def check_dependencies() -> Dict[str, DependencyStatus]:
    """
    Check all dependencies and return their status.
    
    Returns:
        Dictionary mapping module names to their DependencyStatus.
    """
    deps = {}
    
    # Core required dependencies
    deps['numpy'] = DependencyStatus(
        name='numpy',
        available=check_module_available('numpy'),
        required_for=['all modes'],
        pip_install='pip install numpy',
        conda_install='conda install -c conda-forge numpy'
    )
    
    deps['xarray'] = DependencyStatus(
        name='xarray',
        available=check_module_available('xarray'),
        required_for=['all modes'],
        pip_install='pip install xarray',
        conda_install='conda install -c conda-forge xarray'
    )
    
    deps['scipy'] = DependencyStatus(
        name='scipy',
        available=check_module_available('scipy'),
        required_for=['scipy interpolation/fill backends'],
        pip_install='pip install scipy',
        conda_install='conda install -c conda-forge scipy'
    )
    
    # Optional dependencies
    deps['cfgrib'] = DependencyStatus(
        name='cfgrib',
        available=check_module_available('cfgrib'),
        required_for=['gfs mode (GRIB2 reading)'],
        pip_install='pip install cfgrib eccodes',
        conda_install='conda install -c conda-forge cfgrib eccodes'
    )
    
    deps['netCDF4'] = DependencyStatus(
        name='netCDF4',
        available=check_module_available('netCDF4'),
        required_for=['NetCDF output (usually included with xarray)'],
        pip_install='pip install netCDF4',
        conda_install='conda install -c conda-forge netcdf4'
    )
    
    deps['cftime'] = DependencyStatus(
        name='cftime',
        available=check_module_available('cftime'),
        required_for=['handling non-standard calendar dates'],
        pip_install='pip install cftime',
        conda_install='conda install -c conda-forge cftime'
    )
    
    return deps


def validate_core_dependencies(deps: Dict[str, DependencyStatus]) -> None:
    """
    Validate that core dependencies are available.
    Raises SystemExit if critical dependencies are missing.
    """
    core_deps = ['numpy', 'xarray']
    missing_core = [d for d in core_deps if not deps[d].available]
    
    if missing_core:
        print("\n" + "=" * 60)
        print("ERROR: Missing critical dependencies")
        print("=" * 60)
        for name in missing_core:
            d = deps[name]
            print(f"\n  {name}:")
            print(f"    Required for: {', '.join(d.required_for)}")
            print(f"    Install with pip:   {d.pip_install}")
            print(f"    Install with conda: {d.conda_install}")
        print("\nPlease install missing dependencies and try again.")
        print("Quick install: pip install numpy xarray scipy")
        print("Or:            conda install -c conda-forge numpy xarray scipy")
        sys.exit(1)


def validate_mode_dependencies(mode: Mode, deps: Dict[str, DependencyStatus]) -> None:
    """
    Validate dependencies required for the selected mode and runtime options.

    This runs after CONFIG is populated so backend-specific requirements can be
    enforced before any data processing starts.
    """
    if CONFIG.interp_backend == 'scipy' or CONFIG.fill_backend == 'scipy':
        if not deps['scipy'].available:
            d = deps['scipy']
            print("\n" + "=" * 60)
            print("ERROR: scipy is required for the selected backend configuration")
            print("=" * 60)
            print(f"  interp_backend = {CONFIG.interp_backend}")
            print(f"  fill_backend   = {CONFIG.fill_backend}")
            print(f"  Missing module: {d.name}")
            print(f"  Install with pip:   {d.pip_install}")
            print(f"  Install with conda: {d.conda_install}")
            sys.exit(1)

    if mode == Mode.GFS and not deps['cfgrib'].available:
        print("\n" + "=" * 60)
        print("WARNING: cfgrib not available")
        print("=" * 60)
        print("  gfs mode will still try xarray fallback paths, but GRIB2 support may fail.")
        print(f"  Install with pip:   {deps['cfgrib'].pip_install}")
        print(f"  Install with conda: {deps['cfgrib'].conda_install}")
        print()

    if mode == Mode.HRRR and not deps['cfgrib'].available:
        d = deps['cfgrib']
        print("\n" + "=" * 60)
        print("ERROR: cfgrib is required for HRRR mode")
        print("=" * 60)
        print("  HRRR GRIB2 files use projected grids and non-standard variable")
        print("  naming that cfgrib handles reliably.  Without cfgrib, HRRR")
        print("  decoding will almost certainly fail.")
        print(f"  Install with pip:   {d.pip_install}")
        print(f"  Install with conda: {d.conda_install}")
        sys.exit(1)



def log_dependency_status(deps: Dict[str, DependencyStatus]) -> None:
    """Print dependency status summary."""
    print("  Dependencies:")
    core = ['numpy', 'xarray', 'scipy']
    optional = ['cfgrib', 'netCDF4', 'cftime']
    
    for name in core:
        status = "✓" if deps[name].available else "✗"
        print(f"    [{status}] {name} (core)")
    
    for name in optional:
        status = "✓" if deps[name].available else "-"
        print(f"    [{status}] {name} (optional)")


# =============================================================================
# SECTION 5: Environment Detection
# =============================================================================

def is_on_derecho() -> bool:
    """
    Detect whether the script is running on the NCAR Derecho HPC system.
    
    Checks:
    1. FQDN contains 'derecho.hpc.ucar.edu'
    2. SCRATCH env var starts with '/glade/derecho/scratch/'
    """
    fqdn = socket.getfqdn().lower()
    scratch = os.environ.get("SCRATCH", "")
    
    return (
        "derecho.hpc.ucar.edu" in fqdn or
        scratch.startswith("/glade/derecho/scratch/")
    )


# =============================================================================
# SECTION 6: Path Resolution
# =============================================================================

class PathSource(Enum):
    """Indicates where a resolved path came from."""
    USER_PROVIDED = "user-provided"
    DERECHO_DEFAULT = "derecho-default"
    INFERRED = "inferred"
    NOT_SET = "not-set"


@dataclass
class ResolvedPath:
    """Container for a resolved path with metadata."""
    path: Optional[str]
    source: PathSource
    label: str
    
    def __bool__(self):
        return self.path is not None and len(self.path) > 0
    
    def log(self) -> str:
        if self.source == PathSource.USER_PROVIDED:
            return f"[{self.label}] Using user-provided: {self.path}"
        elif self.source == PathSource.DERECHO_DEFAULT:
            return f"[{self.label}] Using Derecho default: {self.path}"
        elif self.source == PathSource.INFERRED:
            return f"[{self.label}] Using inferred path: {self.path}"
        else:
            return f"[{self.label}] Not set"


def resolve_input_path(
    user_value: Optional[str],
    default_path: Optional[str],
    label: str,
    on_derecho: bool,
    required: bool = True,
    check_file: bool = True,
    check_dir: bool = False,
) -> ResolvedPath:
    """
    Resolve an input path with strict priority rules.
    
    Priority:
    1. User-provided path → validate existence → use it
    2. User path doesn't exist → ERROR (never fallback)
    3. No user path + on Derecho + default exists → use default
    4. No user path + not on Derecho → error if required
    
    Args:
        user_value: User-provided path (may be None or empty)
        default_path: Derecho default path (may be None)
        label: Human-readable description for error messages
        on_derecho: Whether running on Derecho
        required: If True, raise error when path cannot be resolved
        check_file: Validate as file
        check_dir: Validate as directory
    
    Returns:
        ResolvedPath with final path and source info
    
    Raises:
        FileNotFoundError: User-provided path doesn't exist
        ValueError: Required path not resolvable
    """
    # Case 1: User provided a path
    if user_value and user_value.strip():
        path = user_value.strip()
        
        # Validate user path exists
        if check_file and not os.path.isfile(path):
            raise FileNotFoundError(
                f"[{label}] User-provided file does not exist: {path}\n"
                f"  You explicitly provided this path, but the file was not found.\n"
                f"  Please verify the path is correct.\n"
                f"  NOTE: User-provided paths are never replaced with defaults."
            )
        if check_dir and not os.path.isdir(path):
            raise FileNotFoundError(
                f"[{label}] User-provided directory does not exist: {path}\n"
                f"  You explicitly provided this path, but the directory was not found.\n"
                f"  Please verify the path is correct.\n"
                f"  NOTE: User-provided paths are never replaced with defaults."
            )
        
        return ResolvedPath(path=path, source=PathSource.USER_PROVIDED, label=label)
    
    # Case 2: No user path → try Derecho default
    if on_derecho and default_path:
        exists = (
            (check_file and os.path.isfile(default_path)) or
            (check_dir and os.path.isdir(default_path)) or
            (not check_file and not check_dir and os.path.exists(default_path))
        )
        if exists:
            return ResolvedPath(path=default_path, source=PathSource.DERECHO_DEFAULT, label=label)
        else:
            # Default doesn't exist
            if required:
                raise FileNotFoundError(
                    f"[{label}] Derecho default path not found: {default_path}\n"
                    f"  You did not provide this path, and the Derecho default does not exist.\n"
                    f"  Please provide the path explicitly."
                )
    
    # Case 3: No path available
    if required:
        env_note = "You are on Derecho" if on_derecho else "You are NOT on Derecho"
        default_note = f"Default path: {default_path}" if default_path else "No default path available"
        raise ValueError(
            f"[{label}] Required path not provided.\n"
            f"  {env_note}\n"
            f"  {default_note}\n"
            f"  Please provide this path explicitly via the appropriate CLI argument."
        )
    
    return ResolvedPath(path=None, source=PathSource.NOT_SET, label=label)


def resolve_output_path(
    user_value: str,
    label: str,
) -> ResolvedPath:
    """
    Resolve an output path.
    
    Output files don't need to exist, but their parent directory must
    exist or be creatable.
    
    Args:
        user_value: User-provided output path
        label: Human-readable description
    
    Returns:
        ResolvedPath (always USER_PROVIDED for outputs)
    
    Raises:
        ValueError: Path is empty or parent directory cannot be created
    """
    if not user_value or not user_value.strip():
        raise ValueError(f"[{label}] Output path is required but not provided.")
    
    path = user_value.strip()
    parent_dir = os.path.dirname(path) or '.'
    
    # Try to create parent directory if it doesn't exist
    try:
        os.makedirs(parent_dir, exist_ok=True)
    except OSError as e:
        raise ValueError(
            f"[{label}] Cannot create parent directory for output: {parent_dir}\n"
            f"  Error: {e}"
        )
    
    return ResolvedPath(path=path, source=PathSource.USER_PROVIDED, label=label)


def ensure_output_dir(dir_path: str, label: str = "Output directory") -> None:
    """Ensure output directory exists, create if needed."""
    if not dir_path:
        dir_path = '.'
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        raise ValueError(f"[{label}] Cannot create directory: {dir_path}\n  Error: {e}")


# =============================================================================
# SECTION 7: Runtime Configuration
# =============================================================================

@dataclass
class RuntimeConfig:
    """Global runtime configuration."""
    interp_backend: str = "python_ncl"
    fill_backend: str = "python_ncl"
    fill_interp_nan: bool = False
    strict_time: bool = False
    on_derecho: bool = False
    
    def log(self):
        env = "Derecho" if self.on_derecho else "Non-Derecho"
        print(f"  Environment: {env}")
        print(f"  Backends: interp={self.interp_backend}, fill={self.fill_backend}")
        if self.fill_interp_nan:
            print(f"  fill_interp_nan: ON")
        if self.strict_time:
            print(f"  strict_time: ON")


CONFIG = RuntimeConfig()


# =============================================================================
# SECTION 8: Delayed Imports (after dependency check)
# =============================================================================

# These are imported after dependency check passes
xr = None
RegularGridInterpolator = None
laplace = None
griddata = None  # scipy.interpolate.griddata — needed for projected-grid regridding

def do_delayed_imports():
    """Import heavier dependencies after validation."""
    global xr, RegularGridInterpolator, laplace, griddata
    import xarray
    xr = xarray
    try:
        from scipy.interpolate import RegularGridInterpolator as RGI
        from scipy.interpolate import griddata as _griddata
        from scipy.ndimage import laplace as lap
        RegularGridInterpolator = RGI
        laplace = lap
        griddata = _griddata
    except ImportError:
        pass


# =============================================================================
# SECTION 9: Numerical Processing - Interpolation
# =============================================================================

def linint2_like_vec(src_lon, src_lat, data_2d, cyclic, dst_lon, dst_lat):
    """
    Vectorized bilinear interpolation matching NCL linint2.
    """
    src_lon = np.asarray(src_lon, dtype=np.float64)
    src_lat = np.asarray(src_lat, dtype=np.float64)
    data = np.asarray(data_2d, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)

    ny_s, nx_s = data.shape

    if cyclic:
        dlon = src_lon[1] - src_lon[0] if nx_s > 1 else 360.0
        period = src_lon[-1] + dlon - src_lon[0]
        dl_norm = src_lon[0] + np.mod(dst_lon - src_lon[0], period)
        ix = np.searchsorted(src_lon, dl_norm, side='right') - 1
        ix = np.clip(ix, 0, nx_s - 1)
        ix_r = np.where(ix == nx_s - 1, 0, ix + 1)
        safe_next = np.clip(ix + 1, 0, nx_s - 1)
        gap = np.where(ix == nx_s - 1, src_lon[0] + period - src_lon[ix],
                       src_lon[safe_next] - src_lon[ix])
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

    jy = np.searchsorted(src_lat, dst_lat, side='right') - 1
    lat_ok = (jy >= 0) & (jy < ny_s - 1)
    jy = np.clip(jy, 0, ny_s - 2)
    jy_t = jy + 1
    gap_lat = np.maximum(src_lat[jy_t] - src_lat[jy], 1e-15)
    wy = (dst_lat - src_lat[jy]) / gap_lat

    JY, IX = np.meshgrid(jy, ix, indexing='ij')
    JYT, IXR = np.meshgrid(jy_t, ix_r, indexing='ij')
    WY2, WX2 = np.meshgrid(wy, wx, indexing='ij')
    LOK, VOK = np.meshgrid(lat_ok, lon_ok, indexing='ij')
    valid = LOK & VOK

    q00, q10 = data[JY, IX], data[JY, IXR]
    q01, q11 = data[JYT, IX], data[JYT, IXR]

    out = (q00 * (1 - WX2) * (1 - WY2) + q10 * WX2 * (1 - WY2) +
           q01 * (1 - WX2) * WY2 + q11 * WX2 * WY2)

    any_nan = np.isnan(q00) | np.isnan(q10) | np.isnan(q01) | np.isnan(q11)
    out[any_nan | ~valid] = np.nan
    return out


# =============================================================================
# SECTION 10: Numerical Processing - Poisson Fill
# =============================================================================

def poisson_grid_fill_ncl(field, lat_deg, is_cyclic=True, guess=1,
                          nscan=2000, eps=1e-2, relc=0.6, dlon_deg=None):
    """Gauss-Seidel SOR solver for Laplace's equation (NCL-faithful)."""
    data = field.copy().astype(np.float64)
    mask = np.isnan(data)
    if not mask.any():
        return data

    ny, nx = data.shape
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    if len(lat_rad) != ny:
        raise ValueError(f"lat_deg length {len(lat_rad)} != data rows {ny}")

    dlat = abs(lat_rad[1] - lat_rad[0]) if ny > 1 else np.deg2rad(0.25)
    dlon = np.deg2rad(360.0 / nx) if is_cyclic else (np.deg2rad(dlon_deg) if dlon_deg else dlat)

    n_fill = int(mask.sum())
    if n_fill > 100000:
        print(f"  [poisson_ncl] {ny}x{nx} grid, {n_fill} fill pts, {nscan} scans")

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

    for _ in range(nscan):
        maxdiff = 0.0
        for j in range(ny):
            a, bn, bs, d = alpha[j], beta_n[j], beta_s[j], denom[j]
            jn, js = min(j + 1, ny - 1), max(j - 1, 0)
            for i in range(nx):
                if not mask[j, i]:
                    continue
                ie = (i + 1) % nx if is_cyclic else min(i + 1, nx - 1)
                iw = (i - 1) % nx if is_cyclic else max(i - 1, 0)
                fstar = (a * (data[j, ie] + data[j, iw]) + bn * data[jn, i] + bs * data[js, i]) / d
                fnew = relc * fstar + (1.0 - relc) * data[j, i]
                maxdiff = max(maxdiff, abs(fnew - data[j, i]))
                data[j, i] = fnew
        if maxdiff < eps:
            break
    return data


def poisson_grid_fill_jacobi(field, lat_deg, is_cyclic=True, guess=1,
                             nscan=2000, eps=1e-2, relc=0.6, dlon_deg=None):
    """Jacobi-style vectorized solver (faster but less NCL-faithful)."""
    data = field.copy().astype(np.float64)
    mask = np.isnan(data)
    if not mask.any():
        return data

    ny, nx = data.shape
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    dlat = abs(lat_rad[1] - lat_rad[0]) if ny > 1 else np.deg2rad(0.25)
    dlon = np.deg2rad(360.0 / nx) if is_cyclic else (np.deg2rad(dlon_deg) if dlon_deg else dlat)

    cosph = np.clip(np.cos(lat_rad), 0.01, None)
    tanph = np.clip(np.tan(lat_rad), -100.0, 100.0)
    ratio2 = (dlat / dlon) ** 2
    a = (ratio2 / cosph ** 2)[:, None] * np.ones((1, nx))
    bn = (1.0 + tanph * dlat / 2.0)[:, None] * np.ones((1, nx))
    bs = (1.0 - tanph * dlat / 2.0)[:, None] * np.ones((1, nx))
    d = 2.0 * a + bn + bs

    valid_mean = np.nanmean(data)
    data[mask] = (valid_mean if guess == 1 and not np.isnan(valid_mean) else 0.0)

    for _ in range(nscan):
        if is_cyclic:
            fe, fw = np.roll(data, -1, axis=1), np.roll(data, 1, axis=1)
        else:
            fe = np.empty_like(data)
            fe[:, :-1], fe[:, -1] = data[:, 1:], data[:, -1]
            fw = np.empty_like(data)
            fw[:, 1:], fw[:, 0] = data[:, :-1], data[:, 0]
        fn = np.empty_like(data)
        fn[:-1], fn[-1] = data[1:], data[-1]
        fs = np.empty_like(data)
        fs[1:], fs[0] = data[:-1], data[0]

        fstar = (a * (fe + fw) + bn * fn + bs * fs) / d
        update = relc * fstar + (1.0 - relc) * data
        maxdiff = np.abs(update - data)[mask].max() if mask.any() else 0.0
        data[mask] = update[mask]
        if maxdiff < eps:
            break
    return data


def scipy_poisson(field, niter=2000, tol=1e-2, relax=0.6, guess=1):
    """scipy.ndimage.laplace Cartesian fallback."""
    if laplace is None:
        raise ImportError("scipy not available for poisson fill")
    data = field.copy().astype(np.float64)
    mask = np.isnan(data)
    if not mask.any():
        return data
    vm = np.nanmean(data)
    data[mask] = (vm if guess == 1 and not np.isnan(vm) else 0.0)
    for _ in range(niter):
        lap = laplace(data)
        old = data[mask].copy()
        data[mask] += relax * lap[mask]
        if np.max(np.abs(data[mask] - old)) < tol:
            break
    return data


# =============================================================================
# SECTION 11: Dispatch Wrappers
# =============================================================================

def interp_to_grid(data_2d, src_lat, src_lon, dst_lat, dst_lon, cyclic=False):
    """Dispatch interpolation to configured backend."""
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    d = np.asarray(data_2d, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)

    if src_lat.size > 1 and src_lat[0] > src_lat[-1]:
        src_lat, d = src_lat[::-1], d[::-1, :]

    if CONFIG.interp_backend == "scipy":
        if RegularGridInterpolator is None:
            raise ImportError("scipy not available for interpolation")
        sl = src_lat[::-1] if src_lat[0] > src_lat[-1] else src_lat
        dd = d[::-1, :] if src_lat[0] > src_lat[-1] else d
        interp = RegularGridInterpolator((sl, src_lon), dd, method='linear',
                                          bounds_error=False, fill_value=None)
        la2, lo2 = np.meshgrid(dst_lat, dst_lon, indexing='ij')
        return interp(np.column_stack([la2.ravel(), lo2.ravel()])).reshape(la2.shape).astype(np.float32)

    result = linint2_like_vec(src_lon, src_lat, d, cyclic, dst_lon, dst_lat)
    nan_count = int(np.isnan(result).sum())
    if nan_count > 0:
        if CONFIG.fill_interp_nan:
            print(f"  [interp] {nan_count} NaN → NN backfill")
            nn_fill_2d(result)
        else:
            print(f"  [interp] {nan_count} NaN preserved")
    return result.astype(np.float32)


def nn_fill_2d(arr):
    """In-place nearest-neighbor fill of NaN."""
    mask = np.isnan(arr)
    if not mask.any():
        return
    ny, nx = arr.shape
    for _ in range(50):
        filled_any = False
        new = arr.copy()
        for j, i in zip(*np.where(mask)):
            neighbors = []
            if j > 0 and not np.isnan(arr[j-1, i]): neighbors.append(arr[j-1, i])
            if j < ny-1 and not np.isnan(arr[j+1, i]): neighbors.append(arr[j+1, i])
            if i > 0 and not np.isnan(arr[j, i-1]): neighbors.append(arr[j, i-1])
            if i < nx-1 and not np.isnan(arr[j, i+1]): neighbors.append(arr[j, i+1])
            if neighbors:
                new[j, i] = np.mean(neighbors)
                filled_any = True
        arr[:] = new
        mask = np.isnan(arr)
        if not mask.any() or not filled_any:
            break


def fill_poisson(field, lat_deg=None, lon_deg=None, is_cyclic=True,
                 nscan=2000, eps=1e-2, relc=0.6, guess=1):
    """Dispatch Poisson fill to configured backend."""
    if lat_deg is None:
        lat_deg = np.linspace(-90, 90, field.shape[0])
        warnings.warn("fill_poisson: lat_deg not provided, assuming uniform -90..90")

    dlon_deg = abs(float(lon_deg[1] - lon_deg[0])) if lon_deg is not None and len(lon_deg) > 1 else None

    if CONFIG.fill_backend == "scipy":
        return scipy_poisson(field, niter=nscan, tol=eps, relax=relc, guess=guess)
    elif CONFIG.fill_backend == "python_ncl_fast":
        return poisson_grid_fill_jacobi(field, lat_deg, is_cyclic, guess, nscan, eps, relc, dlon_deg)
    else:
        return poisson_grid_fill_ncl(field, lat_deg, is_cyclic, guess, nscan, eps, relc, dlon_deg)


# =============================================================================
# SECTION 12: Data Processing Utilities
# =============================================================================

def flip_lat_if_needed(data, lat):
    """Ensure latitude runs S→N."""
    if lat.size > 1 and lat[0] > lat[-1]:
        return data[..., ::-1, :], lat[::-1].copy()
    return data, lat


def flip_lon_if_needed(data, lon, target_lon):
    """Shift longitude convention to match target."""
    if np.any(target_lon < 0) and np.all(lon >= 0):
        lon = np.where(lon > 180, lon - 360, lon).copy()
        ix = np.argsort(lon)
        return data[..., ix], lon[ix]
    elif np.any(target_lon >= 0) and np.any(lon < 0):
        lon = np.where(lon < 0, lon + 360, lon).copy()
        ix = np.argsort(lon)
        return data[..., ix], lon[ix]
    return data, lon


def apply_land_mask(f2d, lm):
    """Set ocean/ice (landmask==0 or 15) to 0; clamp negatives."""
    f2d = np.where((lm == 0) | (lm == 15), 0.0, f2d)
    return np.where(f2d < 0, 0.0, f2d)


def crop_to_domain(data, slat, slon, dlat, dlon, margin=1.0):
    """Crop source field to target domain ± margin degrees."""
    la0, la1 = float(dlat.min()) - margin, float(dlat.max()) + margin
    lat_mask = (slat >= la0) & (slat <= la1)

    slo_min, slo_max = float(slon.min()), float(slon.max())
    tlo_min, tlo_max = float(dlon.min()) - margin, float(dlon.max()) + margin

    if slo_min >= 0 and tlo_min < 0:
        tlo_min, tlo_max = tlo_min + 360.0, tlo_max + 360.0
    elif slo_min < 0 and tlo_min >= 180:
        tlo_min, tlo_max = tlo_min - 360.0, tlo_max - 360.0

    if tlo_min >= slo_min and tlo_max <= slo_max:
        lon_mask = (slon >= tlo_min) & (slon <= tlo_max)
    else:
        lon_mask = np.ones(slon.shape, dtype=bool)

    if lat_mask.sum() < 3 or lon_mask.sum() < 3:
        return data, slat, slon
    return data[np.ix_(lat_mask, lon_mask)], slat[lat_mask], slon[lon_mask]


def grids_aligned(lat_a, lon_a, lat_b, lon_b, atol: float = 0.01) -> bool:
    """Check if two grids have matching shape and coordinates."""
    if lat_a.shape != lat_b.shape or lon_a.shape != lon_b.shape:
        return False
    return np.allclose(lat_a, lat_b, atol=atol) and np.allclose(lon_a, lon_b, atol=atol)


# =============================================================================
# SECTION 13: Binary I/O
# =============================================================================

def fbinrecwrite(f, arr):
    """Write one Fortran unformatted sequential record."""
    raw = np.ascontiguousarray(arr).tobytes()
    n = len(raw)
    f.write(struct.pack('<i', n))
    f.write(raw)
    f.write(struct.pack('<i', n))


def fbinrecread(f, dt, count=1):
    """Read one Fortran unformatted sequential record."""
    h = f.read(4)
    if len(h) < 4:
        raise EOFError("Unexpected EOF")
    n = struct.unpack('<i', h)[0]
    raw = f.read(n)
    if len(raw) < n:
        raise EOFError(f"Short read: {len(raw)}/{n}")
    n2 = struct.unpack('<i', f.read(4))[0]
    if n != n2:
        raise ValueError(f"Fortran record len mismatch: {n} vs {n2}")
    return np.frombuffer(raw, dtype=dt, count=count).copy()


def write_binary_output(fname, nsoil, nlon, nlat, zsoil, soilt, soilw):
    """Write soil init in Fortran binary format."""
    ensure_output_dir(os.path.dirname(fname) or '.')
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
    """Write soil init in NetCDF format."""
    ensure_output_dir(os.path.dirname(fname) or '.')
    ds = xr.Dataset(
        {'soilt': (['zsoil', 'lat', 'lon'], soilt),
         'soilw': (['zsoil', 'lat', 'lon'], soilw)},
        coords={'zsoil': (['zsoil'], zsoil), 'lat': (['lat'], lat), 'lon': (['lon'], lon)},
    )
    ds['zsoil'].attrs['units'] = 'm'
    ds['soilt'].attrs.update({'long_name': 'Soil temperature', 'units': 'K'})
    sw_attrs = {'long_name': 'Soil volumetric water content', 'units': 'm3/m3'}
    if soilw_attrs:
        sw_attrs.update(soilw_attrs)
    ds['soilw'].attrs.update(sw_attrs)
    if extra_vars:
        for k, v in extra_vars.items():
            ds[k] = v
    if os.path.exists(fname):
        os.remove(fname)
    ds.to_netcdf(fname)
    print(f"  Written NetCDF: {fname}")


# =============================================================================
# SECTION 14: Data Loading Utilities
# =============================================================================

def load_landmask(grid: str, indir: str):
    """Load landmask for target grid."""
    p = os.path.join(indir, f"landmask_{grid}.nc")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Landmask file not found: {p}")
    ds = xr.open_dataset(p)
    for v in ['lat', 'lon', 'LANDMASK']:
        if v not in ds.data_vars and v not in ds.coords:
            raise KeyError(f"Variable '{v}' not in {p}")
    lat, lon, lm = ds['lat'].values, ds['lon'].values, ds['LANDMASK'].values
    ds.close()
    return lat, lon, lm


def parse_date(d: int) -> Tuple[int, int, int, int]:
    """Parse YYYYMMDDHH to (year, month, day, hour)."""
    y = d // 1000000
    m = (d - y * 1000000) // 10000
    dy = (d - y * 1000000 - m * 10000) // 100
    h = d % 100
    if not (1 <= m <= 12 and 1 <= dy <= 31 and 0 <= h <= 23):
        raise ValueError(f"Invalid date {d}: Y={y} M={m} D={dy} H={h}")
    return y, m, dy, h


def find_time_index(ds, date_int: int) -> int:
    """Find time index matching date_int."""
    if 'time' not in ds.coords and 'time' not in ds.dims:
        raise KeyError(f"No 'time' coordinate in dataset")

    y, m, d, h = parse_date(date_int)
    times = ds['time'].values
    nt = len(times)

    if hasattr(times[0], 'year'):
        for i, t in enumerate(times):
            if t.year == y and t.month == m and t.day == d and t.hour == h:
                print(f"  Time index {i}: {t}")
                return i
        raise ValueError(f"Date {date_int} not found in {nt} cftime steps")

    tgt = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:00:00")
    try:
        th = times.astype('datetime64[h]')
        tgh = tgt.astype('datetime64[h]')
        ex = np.where(th == tgh)[0]
        if len(ex) > 0:
            print(f"  Time index {int(ex[0])}: {times[ex[0]]}")
            return int(ex[0])

        if CONFIG.strict_time:
            raise ValueError(f"Date {date_int} not found (strict mode)")

        df = np.abs(th - tgh)
        i = int(np.argmin(df))
        if df[i] <= np.timedelta64(1, 'h'):
            print(f"  Time index {i}: {times[i]} (nearest)")
            return i
    except (TypeError, OverflowError):
        pass

    raise ValueError(f"Date {date_int} not found ({nt} time steps)")


def find_var_case_insensitive(ds, base_name: str) -> str:
    """Find variable with case-insensitive matching."""
    candidates = [base_name, base_name.lower(), base_name.upper(), base_name.capitalize()]
    for vn in candidates:
        if vn in ds.data_vars:
            return vn
    avail = list(ds.data_vars)
    raise KeyError(f"Variable '{base_name}' (any case) not found. Available: {avail}")


def resolve_coord(ds, candidates: list, label: str) -> str:
    """Find a coordinate by trying multiple names."""
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    avail = list(ds.coords) + list(ds.dims)
    raise KeyError(f"No {label} coordinate found. Tried {candidates}; available: {avail}")


# =============================================================================
# SECTION 15: ERA5 File Path Construction
# =============================================================================

def build_ncar_era5_paths(filepath: str, date_int: int) -> Tuple[list, list]:
    """Build NCAR RDA ERA5 file paths for STL and SWVL variables."""
    y, m, _, _ = parse_date(date_int)
    ym = f"{y:04d}{m:02d}"
    ld = calendar.monthrange(y, m)[1]
    dobs = f"{y:04d}{m:02d}0100_{y:04d}{m:02d}{ld:02d}23"
    
    stl_paths = [os.path.join(filepath, ym, f"e5.oper.an.sfc.{c}.ll025sc.{dobs}.nc") 
                 for c in STL_CODES]
    swvl_paths = [os.path.join(filepath, ym, f"e5.oper.an.sfc.{c}.ll025sc.{dobs}.nc") 
                  for c in SWVL_CODES]
    
    for p in stl_paths + swvl_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"ERA5 file not found: {p}")
    
    return stl_paths, swvl_paths


# =============================================================================
# SECTION 16: Unified Layer Processing
# =============================================================================

def process_soil_layer(raw_2d, src_lat, src_lon, dst_lat, dst_lon,
                       landmask=None, is_moisture=False, do_poisson=True,
                       era5_lsm=None, label="", cyclic_lon=False,
                       poisson_nscan=2000, poisson_guess=1):
    """
    Process one soil layer:
    flip → (LSM mask) → (Poisson fill) → longitude flip → interpolation → (land mask)
    """
    data = raw_2d.astype(np.float64)
    lat_s, lon_s = src_lat.copy(), src_lon.copy()
    data, lat_s = flip_lat_if_needed(data, lat_s)

    if era5_lsm is not None:
        lsm_w, _ = flip_lat_if_needed(era5_lsm.copy(), src_lat.copy())
        data = np.where(lsm_w < 0.5, np.nan, data)
        if label:
            print(f"  [{label}] LSM: {int((lsm_w < 0.5).sum())} ocean→NaN")

    if do_poisson:
        n_fill = int(np.isnan(data).sum())
        if n_fill > 0:
            data = fill_poisson(data, lat_deg=lat_s, lon_deg=lon_s, is_cyclic=cyclic_lon,
                               nscan=poisson_nscan, guess=poisson_guess)
            if label:
                print(f"  [{label}] poisson_fill: {n_fill} pts")

    data, lon_s = flip_lon_if_needed(data, lon_s, dst_lon)
    result = interp_to_grid(data, lat_s, lon_s, dst_lat, dst_lon, cyclic=cyclic_lon)

    if is_moisture and landmask is not None:
        result = apply_land_mask(result, landmask)

    if label:
        print(f"  [{label}] {raw_2d.shape}→{result.shape} "
              f"min={np.nanmin(result):.4f} max={np.nanmax(result):.4f}")
    return result


def prepare_common_params(grid: str, indir: str):
    """Load the land mask and prepare common parameters."""
    dst_lat, dst_lon, landmask = load_landmask(grid, indir)
    nlat, nlon = len(dst_lat), len(dst_lon)
    print(f"  Grid: {nlon}x{nlat}")
    return dst_lat, dst_lon, landmask, nlat, nlon


def write_outputs(outdir: str, outdir_nc: str, netcdf_out: bool,
                  fname_base: str, nsoil: int, nlon: int, nlat: int,
                  zsoil, soilt, soilw, dst_lat, dst_lon, soilw_attrs=None):
    """Write binary and optional NetCDF outputs."""
    ensure_output_dir(outdir)
    write_binary_output(
        os.path.join(outdir, f"{fname_base}.bin"),
        nsoil, nlon, nlat, zsoil, soilt, soilw
    )
    if netcdf_out:
        ensure_output_dir(outdir_nc)
        write_netcdf_output(
            os.path.join(outdir_nc, f"{fname_base}.nc"),
            soilt, soilw, zsoil, dst_lat, dst_lon, soilw_attrs
        )


# =============================================================================
# SECTION 16A: HRRR Data Structures
# =============================================================================

@dataclass
class HRRRSoilData:
    """Container for loaded and inspected HRRR soil fields."""
    soilt_raw: np.ndarray          # (nsoil, y, x)
    soilw_raw: np.ndarray          # (nsoil, y, x)
    src_lat: np.ndarray            # 1-D (regular_ll) or 2-D (projected)
    src_lon: np.ndarray            # 1-D (regular_ll) or 2-D (projected)
    latlon_is_2d: bool             # True when lat/lon are 2-D arrays
    grid_kind: str                 # "regular_ll" or "projected"
    zsoil_native: np.ndarray       # mid-layer depths in metres (nsoil,)
    time_info: Optional[str]       # human-readable time label or None
    var_names: Dict[str, str]      # {'tsoil': actual_name, 'soilw': actual_name}


# =============================================================================
# SECTION 16B: HRRR Depth Metadata Extraction
# =============================================================================

def _extract_hrrr_depth_metadata(ds) -> np.ndarray:
    """
    Derive mid-layer soil depths (metres) from an opened HRRR dataset.

    Strategy (tried in order):
    1. Look for coordinate variables that encode layer boundaries
       (e.g. 'topLevel'/'bottomLevel', or 'depthBelowLandLayer' with bounds).
    2. If the dataset exposes explicit depth values as a coordinate, use them
       directly as mid-layer depths.
    3. If the number of layers is 4 and matches the known HRRR 4-layer
       boundaries, return ZSOIL_HRRR.
    4. Otherwise raise an error.
    """
    nlayers = None
    # Determine how many soil layers are present from the first soil variable.
    for vn in list(ds.data_vars):
        shape = ds[vn].shape
        if len(shape) >= 2:
            # The shallowest dimension that is exactly 4 (or small) is likely
            # the soil-layer axis.  cfgrib puts it first when time is absent.
            nlayers = shape[0] if shape[0] <= 12 else None
            break

    # --- Strategy 1: explicit top/bottom coordinate pairs ---
    top_names = ['topLevel', 'top', 'depthBelowLandLayer_top']
    bot_names = ['bottomLevel', 'bottom', 'depthBelowLandLayer_bottom']
    top_coord = bot_coord = None
    for tn in top_names:
        if tn in ds.coords:
            top_coord = ds[tn].values
            break
    for bn in bot_names:
        if bn in ds.coords:
            bot_coord = ds[bn].values
            break

    if top_coord is not None and bot_coord is not None:
        mids = (np.asarray(top_coord, dtype=np.float64) +
                np.asarray(bot_coord, dtype=np.float64)) / 2.0
        if np.all(mids > 0):
            print(f"  [HRRR depth] from top/bottom bounds: {mids}")
            return mids.astype(np.float32)

    # --- Strategy 2: single depth coordinate ---
    depth_names = ['depthBelowLandLayer', 'depth', 'soilLayer', 'level']
    for dn in depth_names:
        if dn in ds.coords:
            vals = np.asarray(ds[dn].values, dtype=np.float64)
            if vals.ndim == 1 and vals.size >= 2 and np.all(vals > 0):
                print(f"  [HRRR depth] from coordinate '{dn}': {vals}")
                return vals.astype(np.float32)

    # --- Strategy 3: match known layer count ---
    if nlayers == NSOIL:
        print(f"  [HRRR depth] no explicit depth metadata; "
              f"{NSOIL} layers matches default ZSOIL_HRRR")
        return ZSOIL_HRRR.copy()

    raise ValueError(
        f"Cannot determine HRRR soil layer depths.\n"
        f"  Detected {nlayers} layers but found no depth coordinate.\n"
        f"  Please verify the GRIB2 file contains standard HRRR soil fields."
    )


# =============================================================================
# SECTION 16C: HRRR Grid Classification
# =============================================================================

def _classify_hrrr_grid(lat: np.ndarray, lon: np.ndarray) -> Tuple[bool, str]:
    """
    Classify whether HRRR lat/lon arrays represent a regular lat-lon grid
    or a projected (curvilinear / 2-D) grid.

    Returns:
        (latlon_is_2d, grid_kind)
        latlon_is_2d: True if arrays are 2-D
        grid_kind:    "regular_ll" or "projected"
    """
    if lat.ndim == 1 and lon.ndim == 1:
        # 1-D arrays — check monotonicity to confirm regularity.
        lat_mono = np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0)
        lon_mono = np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0)
        if lat_mono and lon_mono:
            return False, "regular_ll"
        # 1-D but not monotonic is unusual — treat as projected to be safe.
        print("  [HRRR grid] WARNING: 1-D lat/lon but non-monotonic; "
              "treating as projected")
        return False, "projected"

    if lat.ndim == 2 and lon.ndim == 2:
        # 2-D arrays — could still be a rectilinear grid stored as 2-D
        # meshgrids.  Check whether each row of lon is constant and each
        # column of lat is constant (within tolerance).
        if lat.shape == lon.shape and lat.shape[0] > 1 and lat.shape[1] > 1:
            lat_col_spread = np.max(np.ptp(lat, axis=1))   # variation along x
            lon_row_spread = np.max(np.ptp(lon, axis=0))   # variation along y
            tol = 1e-4  # ~11 m at equator
            if lat_col_spread < tol and lon_row_spread < tol:
                print("  [HRRR grid] 2-D arrays are rectilinear → regular_ll")
                return True, "regular_ll"
        return True, "projected"

    raise ValueError(
        f"Unexpected lat/lon dimensionality: lat.ndim={lat.ndim}, "
        f"lon.ndim={lon.ndim}"
    )


# =============================================================================
# SECTION 16D: HRRR Loader
# =============================================================================

def _find_var_in_dataset(ds, candidates: List[str], label: str) -> str:
    """Find the first matching variable name from a list of candidates."""
    for c in candidates:
        if c in ds.data_vars:
            return c
    avail = list(ds.data_vars)
    raise KeyError(
        f"[HRRR] No {label} variable found.\n"
        f"  Tried: {candidates}\n"
        f"  Available: {avail}"
    )


def _load_hrrr_soil_fields(hrrr_file: str, date_int: Optional[int]) -> HRRRSoilData:
    """
    Load and inspect HRRR GRIB2 soil fields.

    Two-stage design:
      Stage 1 — Load: open with cfgrib, discover variables, extract arrays.
      Stage 2 — Inspect: classify grid geometry, extract depth metadata.
    """
    print(f"  Loading HRRR from: {hrrr_file}")

    # ------------------------------------------------------------------
    # Stage 1: Load with cfgrib
    # ------------------------------------------------------------------
    # cfgrib filter: soil temperature lives under depthBelowLandLayer.
    # We open two datasets — one for temperature, one for moisture —
    # because cfgrib may split them into separate hypercubes.

    tsoil_ds = None
    soilw_ds = None

    # Try opening with shortName filters first (most reliable for HRRR).
    for sn_temp in ['st', 'soilt', 'tsoil', 't']:
        try:
            tsoil_ds = xr.open_dataset(
                hrrr_file, engine='cfgrib',
                backend_kwargs={'filter_by_keys': {
                    'typeOfLevel': 'depthBelowLandLayer',
                    'shortName': sn_temp,
                }},
            )
            print(f"  [HRRR] Soil temperature dataset opened (shortName='{sn_temp}')")
            break
        except Exception:
            continue

    for sn_mois in ['soilw', 'sm', 'vsw']:
        try:
            soilw_ds = xr.open_dataset(
                hrrr_file, engine='cfgrib',
                backend_kwargs={'filter_by_keys': {
                    'typeOfLevel': 'depthBelowLandLayer',
                    'shortName': sn_mois,
                }},
            )
            print(f"  [HRRR] Soil moisture dataset opened (shortName='{sn_mois}')")
            break
        except Exception:
            continue

    # Fallback: open without shortName filter and find variables manually.
    if tsoil_ds is None or soilw_ds is None:
        print("  [HRRR] shortName filters failed; opening with typeOfLevel only")
        try:
            ds_all = xr.open_dataset(
                hrrr_file, engine='cfgrib',
                backend_kwargs={'filter_by_keys': {
                    'typeOfLevel': 'depthBelowLandLayer',
                }},
            )
        except Exception as e:
            raise RuntimeError(
                f"[HRRR] Cannot open {hrrr_file} with cfgrib.\n"
                f"  Ensure the file is a valid GRIB2 with depthBelowLandLayer "
                f"soil fields.\n  cfgrib error: {e}"
            )
        if tsoil_ds is None:
            tsoil_ds = ds_all
        if soilw_ds is None:
            soilw_ds = ds_all

    # --- Discover variable names ---
    tsoil_vn = _find_var_in_dataset(tsoil_ds, _HRRR_TSOIL_CANDIDATES,
                                    "soil temperature")
    soilw_vn = _find_var_in_dataset(soilw_ds, _HRRR_SOILW_CANDIDATES,
                                    "soil moisture")
    print(f"  [HRRR] Variables: tsoil='{tsoil_vn}', soilw='{soilw_vn}'")

    # --- Extract raw arrays ---
    tsoil_raw = tsoil_ds[tsoil_vn].values.copy()
    soilw_raw = soilw_ds[soilw_vn].values.copy()

    # --- Handle time dimension if present ---
    time_info = None
    if 'time' in tsoil_ds.coords:
        tvals = tsoil_ds['time'].values
        if tvals.ndim == 0:
            time_info = str(tvals)
            print(f"  [HRRR] Single time step: {time_info}")
        elif date_int is not None:
            idx = find_time_index(tsoil_ds, date_int)
            tsoil_raw = tsoil_ds[tsoil_vn].isel(time=idx).values.copy()
            soilw_raw = soilw_ds[soilw_vn].isel(time=idx).values.copy()
            time_info = str(tvals[idx])
        else:
            # No date requested, take first time step.
            tsoil_raw = tsoil_ds[tsoil_vn].isel(time=0).values.copy()
            soilw_raw = soilw_ds[soilw_vn].isel(time=0).values.copy()
            time_info = str(tvals[0])
            print(f"  [HRRR] No --date; using first time step: {time_info}")
    elif 'valid_time' in tsoil_ds.coords:
        time_info = str(tsoil_ds['valid_time'].values)
        print(f"  [HRRR] valid_time: {time_info}")

    # --- Extract coordinates ---
    lat_cand = ['latitude', 'lat', 'XLAT', 'gridlat_0']
    lon_cand = ['longitude', 'lon', 'XLONG', 'gridlon_0']
    lat_name = resolve_coord(tsoil_ds, lat_cand, "latitude")
    lon_name = resolve_coord(tsoil_ds, lon_cand, "longitude")
    src_lat = tsoil_ds[lat_name].values.copy()
    src_lon = tsoil_ds[lon_name].values.copy()

    # --- Extract depth metadata ---
    zsoil = _extract_hrrr_depth_metadata(tsoil_ds)

    # Close datasets.
    tsoil_ds.close()
    if soilw_ds is not tsoil_ds:
        soilw_ds.close()

    # ------------------------------------------------------------------
    # Stage 2: Inspect structure
    # ------------------------------------------------------------------

    # Ensure arrays are at least 2-D spatial (nsoil, y, x) or (y, x).
    # cfgrib may return (nsoil, y, x) or (y, x) depending on layer count.
    if tsoil_raw.ndim == 2:
        tsoil_raw = tsoil_raw[np.newaxis, ...]
    if soilw_raw.ndim == 2:
        soilw_raw = soilw_raw[np.newaxis, ...]

    nsoil_t = tsoil_raw.shape[0]
    nsoil_w = soilw_raw.shape[0]

    if nsoil_t != NSOIL:
        raise ValueError(
            f"[HRRR] Expected {NSOIL} soil temperature layers, got {nsoil_t}.\n"
            f"  Shape: {tsoil_raw.shape}\n"
            f"  If HRRR uses a different layer count, the mapping logic must "
            f"be extended."
        )
    if nsoil_w != NSOIL:
        raise ValueError(
            f"[HRRR] Expected {NSOIL} soil moisture layers, got {nsoil_w}.\n"
            f"  Shape: {soilw_raw.shape}"
        )

    # Classify grid geometry.
    latlon_is_2d, grid_kind = _classify_hrrr_grid(src_lat, src_lon)
    print(f"  [HRRR] Grid: {grid_kind} "
          f"(lat {'2D' if latlon_is_2d else '1D'} {src_lat.shape}, "
          f"lon {'2D' if latlon_is_2d else '1D'} {src_lon.shape})")
    print(f"  [HRRR] Soil layers (zsoil): {zsoil}")
    print(f"  [HRRR] tsoil: {tsoil_raw.shape} "
          f"min={np.nanmin(tsoil_raw):.2f} max={np.nanmax(tsoil_raw):.2f}")
    print(f"  [HRRR] soilw: {soilw_raw.shape} "
          f"min={np.nanmin(soilw_raw):.4f} max={np.nanmax(soilw_raw):.4f}")

    return HRRRSoilData(
        soilt_raw=tsoil_raw,
        soilw_raw=soilw_raw,
        src_lat=src_lat,
        src_lon=src_lon,
        latlon_is_2d=latlon_is_2d,
        grid_kind=grid_kind,
        zsoil_native=zsoil,
        time_info=time_info,
        var_names={'tsoil': tsoil_vn, 'soilw': soilw_vn},
    )


# =============================================================================
# SECTION 16E: Projected-Grid Regridding
# =============================================================================

def regrid_projected_to_target(
    data_2d: np.ndarray,
    src_lat_2d: np.ndarray,
    src_lon_2d: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    method: str = 'linear',
) -> np.ndarray:
    """
    Regrid a field on a projected (2-D lat/lon) grid to a regular target grid.

    Uses scipy.interpolate.griddata for unstructured → structured interpolation.
    Falls back to nearest-neighbor if linear interpolation leaves NaN gaps
    (common at domain edges where the projected grid doesn't fully cover the
    target).

    Args:
        data_2d:     (ny_src, nx_src) source field
        src_lat_2d:  (ny_src, nx_src) latitude of each source cell
        src_lon_2d:  (ny_src, nx_src) longitude of each source cell
        dst_lat:     1-D target latitudes
        dst_lon:     1-D target longitudes
        method:      'linear' (default) or 'nearest'

    Returns:
        (nlat_dst, nlon_dst) interpolated field
    """
    if griddata is None:
        raise ImportError(
            "[HRRR] scipy.interpolate.griddata is required for projected-grid "
            "regridding but scipy is not available.\n"
            "  Install with: pip install scipy\n"
            "  Or:           conda install -c conda-forge scipy"
        )

    # Flatten source coordinates and data.
    src_pts = np.column_stack([src_lat_2d.ravel(), src_lon_2d.ravel()])
    src_vals = data_2d.ravel()

    # Remove NaN source points (griddata cannot handle them).
    valid = np.isfinite(src_vals)
    if valid.sum() == 0:
        warnings.warn("[HRRR regrid] All source values are NaN")
        return np.full((len(dst_lat), len(dst_lon)), np.nan, dtype=np.float32)

    src_pts = src_pts[valid]
    src_vals = src_vals[valid]

    # Build target meshgrid.
    dst_lat_g, dst_lon_g = np.meshgrid(dst_lat, dst_lon, indexing='ij')
    dst_pts = np.column_stack([dst_lat_g.ravel(), dst_lon_g.ravel()])

    # Primary interpolation.
    result = griddata(src_pts, src_vals, dst_pts, method=method)
    result = result.reshape(dst_lat_g.shape)

    # Fill remaining NaNs with nearest-neighbor (edge gaps).
    nan_mask = np.isnan(result)
    n_nan = int(nan_mask.sum())
    if n_nan > 0 and method != 'nearest':
        nn = griddata(src_pts, src_vals, dst_pts, method='nearest')
        nn = nn.reshape(dst_lat_g.shape)
        result[nan_mask] = nn[nan_mask]
        print(f"  [HRRR regrid] {n_nan} edge NaNs filled by nearest-neighbor")

    return result.astype(np.float32)


# =============================================================================
# SECTION 16F: HRRR run_hrrr() — Mode Implementation
# =============================================================================

def run_hrrr(grid: str, date_int: Optional[int], hrrr_file: str,
             indir: str, outdir: str, outdir_nc: str, netcdf_out: bool):
    """
    Mode: hrrr — initialize soil from an HRRR GRIB2 file.

    Two processing paths based on the detected grid geometry:
      regular_ll  → reuse existing pipeline (process_soil_layer, etc.)
      projected   → use regrid_projected_to_target (scipy griddata)
    """
    dst_lat, dst_lon, landmask, nlat, nlon = prepare_common_params(grid, indir)
    print(f"  [HRRR mode]")

    # --- Load and inspect ---
    hdata = _load_hrrr_soil_fields(hrrr_file, date_int)
    zsoil = hdata.zsoil_native

    soilt = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    soilw = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)

    # ------------------------------------------------------------------
    # CASE A: regular lat-lon grid → reuse existing pipeline
    # ------------------------------------------------------------------
    if hdata.grid_kind == "regular_ll":
        print("  [HRRR] Processing as regular lat-lon grid")

        # If 2-D but rectilinear, extract 1-D vectors.
        if hdata.latlon_is_2d:
            src_lat_1d = hdata.src_lat[:, 0]
            src_lon_1d = hdata.src_lon[0, :]
        else:
            src_lat_1d = hdata.src_lat
            src_lon_1d = hdata.src_lon

        for n in range(NSOIL):
            has_nan = bool(np.isnan(hdata.soilt_raw[n]).any())
            soilt[n] = process_soil_layer(
                hdata.soilt_raw[n], src_lat_1d, src_lon_1d,
                dst_lat, dst_lon,
                do_poisson=has_nan,
                label=f"HRRR_T[{n}]",
                cyclic_lon=False,  # HRRR is regional, never cyclic
            )

        for n in range(NSOIL):
            has_nan = bool(np.isnan(hdata.soilw_raw[n]).any())
            soilw[n] = process_soil_layer(
                hdata.soilw_raw[n], src_lat_1d, src_lon_1d,
                dst_lat, dst_lon,
                landmask=landmask, is_moisture=True,
                do_poisson=has_nan,
                label=f"HRRR_W[{n}]",
                cyclic_lon=False,
            )

    # ------------------------------------------------------------------
    # CASE B: projected grid → scipy griddata regridding
    # ------------------------------------------------------------------
    elif hdata.grid_kind == "projected":
        print("  [HRRR] Processing as projected grid (scipy griddata)")

        if griddata is None:
            raise RuntimeError(
                "[HRRR] Projected-grid regridding requires scipy, which is "
                "not available.\n"
                "  Install with: pip install scipy\n"
                "  Or:           conda install -c conda-forge scipy"
            )

        for n in range(NSOIL):
            soilt[n] = regrid_projected_to_target(
                hdata.soilt_raw[n],
                hdata.src_lat, hdata.src_lon,
                dst_lat, dst_lon,
            )
            print(f"  HRRR_T[{n}]: {soilt[n].min():.2f}..{soilt[n].max():.2f}")

        for n in range(NSOIL):
            raw_w = regrid_projected_to_target(
                hdata.soilw_raw[n],
                hdata.src_lat, hdata.src_lon,
                dst_lat, dst_lon,
            )
            soilw[n] = apply_land_mask(raw_w, landmask)
            print(f"  HRRR_W[{n}]: {soilw[n].min():.4f}..{soilw[n].max():.4f}")

    else:
        raise ValueError(f"[HRRR] Unknown grid_kind: {hdata.grid_kind}")

    # --- Sanity checks ---
    tmin, tmax = float(soilt.min()), float(soilt.max())
    wmin, wmax = float(soilw.min()), float(soilw.max())
    print(f"  Final: soilt {tmin:.2f}..{tmax:.2f}  soilw {wmin:.4f}..{wmax:.4f}")

    if tmax < 100 or tmax > 400:
        warnings.warn(
            f"[HRRR] Soil temperature range [{tmin:.1f}, {tmax:.1f}] K "
            f"looks suspicious — verify units are Kelvin."
        )
    if wmax > 1.0:
        warnings.warn(
            f"[HRRR] Soil moisture max={wmax:.4f} exceeds 1.0 m³/m³ — "
            f"verify units."
        )

    # --- Output ---
    date_tag = str(date_int) if date_int else "nodate"
    write_outputs(outdir, outdir_nc, netcdf_out,
                  f"soil_init_{date_tag}_{grid}_HRRR",
                  NSOIL, nlon, nlat, zsoil, soilt, soilw, dst_lat, dst_lon)
    return soilt, soilw


# =============================================================================
# SECTION 17: Mode Implementations
# =============================================================================

def _load_era5_from_directory(filepath: str, date_int: int, label: str = "NCAR"):
    """Load ERA5 data from NCAR-style directory structure."""
    print(f"  Loading ERA5 from directory ({label}): {filepath}")
    stl_paths, swvl_paths = build_ncar_era5_paths(filepath, date_int)
    
    ds0 = xr.open_dataset(stl_paths[0])
    itime = find_time_index(ds0, date_int)
    src_lat = ds0['latitude'].values
    src_lon = ds0['longitude'].values
    ds0.close()
    
    stl_data = []
    for n in range(NSOIL):
        ds = xr.open_dataset(stl_paths[n])
        vn = f"STL{n+1}"
        stl_data.append(ds[vn].isel(time=itime).values)
        ds.close()

    swvl_data = []
    for n in range(NSOIL):
        ds = xr.open_dataset(swvl_paths[n])
        vn = f"SWVL{n+1}"
        swvl_data.append(ds[vn].isel(time=itime).values)
        ds.close()
    
    return stl_data, swvl_data, src_lat, src_lon


def _load_era5_from_single_file(filedata: str, date_int: int):
    """Load ERA5 data from single file (CDS download style)."""
    print(f"  Loading ERA5 from single file: {filedata}")
    ds = xr.open_dataset(filedata)
    
    latn = resolve_coord(ds, ['latitude', 'lat'], "latitude")
    lonn = resolve_coord(ds, ['longitude', 'lon'], "longitude")
    itime = find_time_index(ds, date_int)
    
    stl_vars = [find_var_case_insensitive(ds, f"stl{n+1}") for n in range(NSOIL)]
    swvl_vars = [find_var_case_insensitive(ds, f"swvl{n+1}") for n in range(NSOIL)]
    
    src_lat = ds[latn].values
    src_lon = ds[lonn].values.copy()
    if np.max(src_lon) < 0:
        src_lon += 360.0
        print("  Shifted longitude +360")
    
    stl_data = [ds[stl_vars[n]].values[itime] for n in range(NSOIL)]
    swvl_data = [ds[swvl_vars[n]].values[itime] for n in range(NSOIL)]
    
    ds.close()
    return stl_data, swvl_data, src_lat, src_lon


def run_era5(grid: str, date_int: int, filepath: Optional[str], filedata: Optional[str],
             indir: str, outdir: str, outdir_nc: str, netcdf_out: bool):
    """
    Mode: era5 — standard ERA5 soil initialization.
    
    Supports both NCAR directory structure (--filepath) and single file (--filedata).
    """
    dst_lat, dst_lon, landmask, nlat, nlon = prepare_common_params(grid, indir)
    
    # Load data based on what was provided
    if filepath:
        stl_data, swvl_data, src_lat, src_lon = _load_era5_from_directory(filepath, date_int)
        do_poisson_stl = False
        do_poisson_swvl = True
    else:
        stl_data, swvl_data, src_lat, src_lon = _load_era5_from_single_file(filedata, date_int)
        do_poisson_stl = False
        do_poisson_swvl = False
    
    soilt = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        print(f"  STL{n+1}: min={np.nanmin(stl_data[n]):.2f} max={np.nanmax(stl_data[n]):.2f}")
        soilt[n] = process_soil_layer(stl_data[n], src_lat, src_lon, dst_lat, dst_lon,
                                      do_poisson=do_poisson_stl, label=f"STL{n+1}")

    soilw = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        print(f"  SWVL{n+1}: min={np.nanmin(swvl_data[n]):.4f} max={np.nanmax(swvl_data[n]):.4f}")
        soilw[n] = process_soil_layer(swvl_data[n], src_lat, src_lon, dst_lat, dst_lon,
                                      landmask=landmask, is_moisture=True,
                                      do_poisson=do_poisson_swvl, label=f"SWVL{n+1}",
                                      cyclic_lon=True)

    print(f"  Final: soilt {soilt.min():.2f}..{soilt.max():.2f} "
          f"soilw {soilw.min():.4f}..{soilw.max():.4f}")
    
    write_outputs(outdir, outdir_nc, netcdf_out,
                  f"soil_init_{date_int}_{grid}_era5",
                  NSOIL, nlon, nlat, ZSOIL_ERA5, soilt, soilw, dst_lat, dst_lon)
    return soilt, soilw


def run_era5_lsm(grid: str, date_int: int, filepath: str, era5_lsm_path: str,
                 indir: str, outdir: str, outdir_nc: str, netcdf_out: bool):
    """Mode: era5_lsm — ERA5 with LSM mask and extended Poisson fill."""
    dst_lat, dst_lon, landmask, nlat, nlon = prepare_common_params(grid, indir)
    print(f"  [LSM mode]")

    # Load LSM
    ds_l = xr.open_dataset(era5_lsm_path)
    if 'LSM' not in ds_l.data_vars:
        raise KeyError(f"LSM variable not found in {era5_lsm_path}")
    lsm_full = ds_l['LSM'].values
    while lsm_full.ndim > 2:
        lsm_full = lsm_full[0]
    ds_l.close()

    # Load ERA5 data
    stl_data, swvl_data, src_lat, src_lon = _load_era5_from_directory(filepath, date_int)

    def process_with_lsm(raw, sl, slo, lsm, vn):
        """Process with LSM mask → crop → poisson fill → interp (cyclic=True)."""
        d = raw.astype(np.float64)
        d, lat_s = flip_lat_if_needed(d, sl.copy())
        lw, _ = flip_lat_if_needed(lsm.copy(), sl.copy())
        d = np.where(lw < 0.5, np.nan, d)
        
        dc, lc, loc = crop_to_domain(d, lat_s, slo.copy(), dst_lat, dst_lon, margin=1.0)
        print(f"  [{vn}] crop {d.shape}→{dc.shape}")
        
        n_fill = int(np.isnan(dc).sum())
        if n_fill > 0:
            dc = fill_poisson(dc, lat_deg=lc, lon_deg=loc, is_cyclic=False,
                            nscan=6000, guess=1)
            print(f"  [{vn}] poisson_fill: {n_fill} pts (nscan=6000)")
        
        dc, loc = flip_lon_if_needed(dc, loc, dst_lon)
        return interp_to_grid(dc, lc, loc, dst_lat, dst_lon, cyclic=True)

    soilt = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        soilt[n] = process_with_lsm(stl_data[n], src_lat, src_lon, lsm_full, f"STL{n+1}")
        print(f"  STL{n+1}(LSM): {soilt[n].min():.2f}..{soilt[n].max():.2f}")

    soilw = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        soilw[n] = apply_land_mask(
            process_with_lsm(swvl_data[n], src_lat, src_lon, lsm_full, f"SWVL{n+1}"),
            landmask
        )
        print(f"  SWVL{n+1}(LSM): {soilw[n].min():.4f}..{soilw[n].max():.4f}")

    write_outputs(outdir, outdir_nc, netcdf_out,
                  f"soil_init_{date_int}_{grid}_era5_lsm",
                  NSOIL, nlon, nlat, ZSOIL_ERA5, soilt, soilw, dst_lat, dst_lon)
    return soilt, soilw


def run_era5_delta(grid: str, date_int: int, filepath: str, delta_file: str,
                   dataset_tag: str, indir: str, outdir: str, outdir_nc: str,
                   netcdf_out: bool):
    """Mode: era5_delta — ERA5 + climate delta (PGW)."""
    dst_lat, dst_lon, landmask, nlat, nlon = prepare_common_params(grid, indir)
    print(f"  [DELTA tag={dataset_tag}]")

    # Load delta
    ds_d = xr.open_dataset(delta_file)
    vn = None
    for c in ['skt_delta', 'skt', 'SKT', 'dts', 'DTS']:
        if c in ds_d.data_vars:
            vn = c
            break
    if not vn:
        raise KeyError(f"No SKT delta variable in {delta_file}. Available: {list(ds_d.data_vars)}")
    
    sr = ds_d[vn].values
    while sr.ndim > 2:
        sr = sr[0]
    latn = resolve_coord(ds_d, ['latitude', 'lat'], "latitude")
    lonn = resolve_coord(ds_d, ['longitude', 'lon'], "longitude")
    dla, dlo = ds_d[latn].values, ds_d[lonn].values
    ds_d.close()
    
    sr, dla = flip_lat_if_needed(sr, dla)
    sr, dlo = flip_lon_if_needed(sr, dlo, dst_lon)
    skt_d = interp_to_grid(sr, dla, dlo, dst_lat, dst_lon)
    print(f"  SKT delta: {np.nanmin(skt_d):.3f}..{np.nanmax(skt_d):.3f}")

    # Load ERA5
    stl_data, swvl_data, src_lat, src_lon = _load_era5_from_directory(filepath, date_int)

    soilt = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        soilt[n] = process_soil_layer(stl_data[n], src_lat, src_lon, dst_lat, dst_lon,
                                      do_poisson=False, label=f"STL{n+1}")
        soilt[n] += skt_d
        print(f"  STL{n+1}+Δ: {soilt[n].min():.2f}..{soilt[n].max():.2f}")

    soilw = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        soilw[n] = process_soil_layer(swvl_data[n], src_lat, src_lon, dst_lat, dst_lon,
                                      landmask=landmask, is_moisture=True,
                                      do_poisson=True, label=f"SWVL{n+1}", cyclic_lon=True)

    write_outputs(outdir, outdir_nc, netcdf_out,
                  f"soil_init_{date_int}_{grid}_{dataset_tag}",
                  NSOIL, nlon, nlat, ZSOIL_ERA5, soilt, soilw, dst_lat, dst_lon)
    return soilt, soilw


def run_era5_wetness(grid: str, date_int: int, filepath: str, porosity_file: str,
                     indir: str, outdir: str, outdir_nc: str, netcdf_out: bool):
    """Mode: era5_wetness — SWVL/porosity → negative wetness (gSAM)."""
    dst_lat, dst_lon, landmask, nlat, nlon = prepare_common_params(grid, indir)
    print(f"  [wetness]")

    # Load porosity
    ds_p = xr.open_dataset(porosity_file)
    if 'SLT' not in ds_p.data_vars:
        raise KeyError(f"SLT variable not found in {porosity_file}")
    slt_r = ds_p['SLT'].values
    while slt_r.ndim > 2:
        slt_r = slt_r[0]
    pln = resolve_coord(ds_p, ['latitude', 'lat'], "latitude")
    pon = resolve_coord(ds_p, ['longitude', 'lon'], "longitude")
    plr, plo = ds_p[pln].values, ds_p[pon].values
    ds_p.close()

    slt, pla = flip_lat_if_needed(slt_r, plr.copy())
    poro = np.full_like(slt, POROSITY_DEFAULT, dtype=np.float64)
    for st, pv in POROSITY_TABLE.items():
        poro = np.where(slt == st, pv, poro)
    print(f"  Porosity: {poro.min():.3f}..{poro.max():.3f}")

    # Load ERA5
    stl_data, swvl_data, src_lat, src_lon = _load_era5_from_directory(filepath, date_int)

    e5la_sn = src_lat[::-1] if src_lat[0] > src_lat[-1] else src_lat
    if not grids_aligned(pla, plo, e5la_sn, src_lon):
        print(f"  WARNING: Porosity grid != ERA5 → interpolating")
        poro = interp_to_grid(poro, pla, plo, e5la_sn, src_lon).astype(np.float64)
        poro = np.where(poro < 0.05, POROSITY_DEFAULT, poro)

    soilt = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        soilt[n] = process_soil_layer(stl_data[n], src_lat, src_lon, dst_lat, dst_lon,
                                      do_poisson=False, label=f"STL{n+1}")

    soilw = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        raw_sn, lat_sn = flip_lat_if_needed(swvl_data[n].copy(), src_lat.copy())
        raw_sn = raw_sn / poro
        soilw[n] = process_soil_layer(raw_sn, lat_sn, src_lon, dst_lat, dst_lon,
                                      landmask=landmask, is_moisture=True,
                                      do_poisson=False, label=f"SWVL{n+1}/poro")

    soilw = -soilw
    print(f"  soilw(neg wetness): {soilw.min():.4f}..{soilw.max():.4f}")

    write_outputs(outdir, outdir_nc, netcdf_out,
                  f"soilwetness_init_{date_int}_{grid}_era5",
                  NSOIL, nlon, nlat, ZSOIL_ERA5, soilt, soilw, dst_lat, dst_lon,
                  soilw_attrs={'long_name': 'Soil wetness (neg, gSAM)', 'units': '1'})
    return soilt, soilw


def run_gfs(grid: str, date_int: int, gfs_file: str,
            indir: str, outdir: str, outdir_nc: str, netcdf_out: bool):
    """Mode: gfs — initialize from a GFS GRIB2 file."""
    dst_lat, dst_lon, landmask, nlat, nlon = prepare_common_params(grid, indir)
    print(f"  [GFS]")

    tsoil_raw, soilw_raw, gfs_lat, gfs_lon = None, None, None, None

    try:
        ds_t = xr.open_dataset(gfs_file, engine='cfgrib',
                               backend_kwargs={'filter_by_keys': {
                                   'typeOfLevel': 'depthBelowLandLayer', 'shortName': 'st'}})
        tv = list(ds_t.data_vars)[0]
        tsoil_raw = ds_t[tv].values
        gfs_lat, gfs_lon = ds_t['latitude'].values, ds_t['longitude'].values
        ds_t.close()

        ds_w = xr.open_dataset(gfs_file, engine='cfgrib',
                               backend_kwargs={'filter_by_keys': {
                                   'typeOfLevel': 'depthBelowLandLayer', 'shortName': 'soilw'}})
        soilw_raw = ds_w[list(ds_w.data_vars)[0]].values
        ds_w.close()
        print(f"  cfgrib: loaded")
    except Exception as e:
        print(f"  cfgrib failed ({e}), trying NCL-style names")
        ds = xr.open_dataset(gfs_file)
        for v in ['TSOIL_P0_2L106_GLL0', 'TSOIL_P0_L106_GLL0']:
            if v in ds.data_vars:
                tsoil_raw = ds[v].values
                break
        for v in ['SOILW_P0_2L106_GLL0', 'SOILW_P0_L106_GLL0']:
            if v in ds.data_vars:
                soilw_raw = ds[v].values
                break
        ln = resolve_coord(ds, ['lat_0', 'latitude', 'lat'], "latitude")
        on = resolve_coord(ds, ['lon_0', 'longitude', 'lon'], "longitude")
        gfs_lat, gfs_lon = ds[ln].values, ds[on].values
        ds.close()

    if tsoil_raw is None or soilw_raw is None:
        raise RuntimeError(f"Cannot load soil variables from {gfs_file}")

    if tsoil_raw.ndim == 4:
        if tsoil_raw.shape[0] == 1:
            tsoil_raw, soilw_raw = tsoil_raw[0], soilw_raw[0]
        else:
            raise ValueError(f"GFS has multiple time steps, cannot auto-select")
    elif tsoil_raw.ndim != 3:
        raise ValueError(f"GFS data has unexpected shape {tsoil_raw.shape}")

    if tsoil_raw.shape[0] != NSOIL:
        raise ValueError(f"GFS first dim is {tsoil_raw.shape[0]}, expected {NSOIL}")

    print(f"  GFS: tsoil={tsoil_raw.shape} soilw={soilw_raw.shape}")

    soilt = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        soilt[n] = process_soil_layer(tsoil_raw[n], gfs_lat, gfs_lon, dst_lat, dst_lon,
                                      do_poisson=True, label=f"TSOIL[{n}]", cyclic_lon=True)

    soilw = np.zeros((NSOIL, nlat, nlon), dtype=np.float32)
    for n in range(NSOIL):
        soilw[n] = process_soil_layer(soilw_raw[n], gfs_lat, gfs_lon, dst_lat, dst_lon,
                                      landmask=landmask, is_moisture=True,
                                      do_poisson=True, label=f"SOILW[{n}]", cyclic_lon=True)

    write_outputs(outdir, outdir_nc, netcdf_out,
                  f"soil_init_{date_int}_{grid}_GFS",
                  NSOIL, nlon, nlat, ZSOIL_GFS, soilt, soilw, dst_lat, dst_lon)
    return soilt, soilw


def run_cesm_cc(grid: str, base_bin_file: str, output_bin_file: str,
                cesm_current_file: str, cesm_future_file: str,
                landfrac_file: str, nday: int,
                netcdf_out: bool, outdir_nc: str, indir: str):
    """Mode: cesm_cc — overlay CESM ΔTS onto existing soil initialization."""
    dst_lat, dst_lon, _ = load_landmask(grid, indir)
    gnlat, gnlon = len(dst_lat), len(dst_lon)
    print(f"  Grid: {gnlon}x{gnlat} [CESM CC]")

    ds1 = xr.open_dataset(cesm_current_file)
    ds2 = xr.open_dataset(cesm_future_file)
    
    if 'TS' not in ds1.data_vars:
        raise KeyError(f"TS not in {cesm_current_file}")
    if 'TS' not in ds2.data_vars:
        raise KeyError(f"TS not in {cesm_future_file}")

    ti = nday * 4 - 1
    if ti < 0 or ti >= ds1['TS'].shape[0] or ti >= ds2['TS'].shape[0]:
        raise IndexError(f"nday={nday} → index {ti} out of bounds")

    ts1, ts2 = ds1['TS'].values[ti], ds2['TS'].values[ti]
    cesm_lat, cesm_lon = ds1['lat'].values, ds1['lon'].values
    ds1.close()
    ds2.close()

    # Landfrac path is fully resolved before dispatch.
    if not landfrac_file:
        raise ValueError("landfrac_file must be resolved before run_cesm_cc() is called")

    lf_ds = xr.open_dataset(landfrac_file)
    if 'LANDFRAC' not in lf_ds.data_vars:
        raise KeyError(f"LANDFRAC not in {landfrac_file}")
    lf_vals = lf_ds['LANDFRAC'].values
    while lf_vals.ndim > 2:
        lf_vals = lf_vals[0]
    lf_ds.close()

    if lf_vals.shape != ts1.shape:
        raise ValueError(f"LANDFRAC shape {lf_vals.shape} != TS shape {ts1.shape}")

    cesm_lm = np.where(lf_vals == 0, 0.0, 1.0)

    # Poisson fill with NCL-matching parameters: guess=0, nscan=1500
    ts1 = np.where(cesm_lm > 0, ts1, np.nan)
    ts1 = fill_poisson(ts1, lat_deg=cesm_lat, nscan=1500, guess=0)
    ts2 = np.where(cesm_lm > 0, ts2, np.nan)
    ts2 = fill_poisson(ts2, lat_deg=cesm_lat, nscan=1500, guess=0)

    dts = ts2 - ts1
    dts_interp = interp_to_grid(dts, cesm_lat, cesm_lon, dst_lat, dst_lon)
    print(f"  DTS: {np.nanmin(dts_interp):.2f}..{np.nanmax(dts_interp):.2f}")

    # Read base binary
    with open(base_bin_file, 'rb') as f:
        nsoil_bin = fbinrecread(f, np.int32, 1)[0]
        nlon_bin = fbinrecread(f, np.int32, 1)[0]
        nlat_bin = fbinrecread(f, np.int32, 1)[0]
        zsoil = fbinrecread(f, np.float32, nsoil_bin)
        if nlat_bin != gnlat or nlon_bin != gnlon:
            raise ValueError(f"base_bin grid ({nlat_bin},{nlon_bin}) != landmask ({gnlat},{gnlon})")
        soilt = np.zeros((nsoil_bin, nlat_bin, nlon_bin), dtype=np.float32)
        for i in range(nsoil_bin):
            soilt[i] = fbinrecread(f, np.float32, nlat_bin * nlon_bin).reshape(nlat_bin, nlon_bin)
        soilw = np.zeros((nsoil_bin, nlat_bin, nlon_bin), dtype=np.float32)
        for i in range(nsoil_bin):
            soilw[i] = fbinrecread(f, np.float32, nlat_bin * nlon_bin).reshape(nlat_bin, nlon_bin)

    for i in range(nsoil_bin):
        soilt[i] += dts_interp
        print(f"  tsoil+dts[{i}]: {soilt[i].min():.2f}..{soilt[i].max():.2f}")

    write_binary_output(output_bin_file, nsoil_bin, nlon_bin, nlat_bin, zsoil, soilt, soilw)

    if netcdf_out:
        ensure_output_dir(outdir_nc)
        nc_name = os.path.join(outdir_nc, f"dts_{grid}_era5_CESM2090s.nc")
        ds_out = xr.Dataset({'dts_grid': (['lat', 'lon'], dts_interp)},
                           coords={'lat': dst_lat, 'lon': dst_lon})
        ds_out['dts_grid'].attrs = {'long_name': 'DTS', 'units': 'K'}
        if os.path.exists(nc_name):
            os.remove(nc_name)
        ds_out.to_netcdf(nc_name)
        print(f"  Written {nc_name}")

    return soilt, soilw


# =============================================================================
# SECTION 18: Mode Path Resolution (Unified)
# =============================================================================

@dataclass
class ResolvedModePaths:
    """Container for all resolved paths for a mode."""
    paths: Dict[str, ResolvedPath]
    
    def get(self, key: str) -> Optional[str]:
        """Get a path value, or return None if it is not set."""
        if key in self.paths and self.paths[key]:
            return self.paths[key].path
        return None
    
    def log_all(self):
        """Print all resolved paths."""
        print("  Resolved paths:")
        for name, rp in self.paths.items():
            print(f"    {rp.log()}")


def resolve_mode_paths(mode: Mode, args: argparse.Namespace, on_derecho: bool) -> ResolvedModePaths:
    """
    Resolve all paths for the given mode.
    
    This is the central path resolution function that applies consistent
    priority rules across all modes:
    
    1. User-provided path takes priority
    2. A user path that does not exist → ERROR (no fallback)
    3. No user path + on Derecho → use default
    4. No user path + not on Derecho → ERROR
    """
    paths = {}
    
    # --- ERA5 modes: filepath or filedata ---
    if mode in [Mode.ERA5, Mode.ERA5_LSM, Mode.ERA5_DELTA, Mode.ERA5_WETNESS]:
        user_filepath = getattr(args, 'filepath', '') or ''
        user_filedata = getattr(args, 'filedata', '') or ''
        
        # Priority: filedata > filepath > Derecho default
        if user_filedata.strip():
            # User provided single file
            paths['filedata'] = resolve_input_path(
                user_value=user_filedata,
                default_path=None,
                label='ERA5 single file (--filedata)',
                on_derecho=on_derecho,
                required=True,
                check_file=True,
            )
            paths['filepath'] = ResolvedPath(None, PathSource.NOT_SET, 'ERA5 directory')
        elif user_filepath.strip():
            # User provided directory path
            paths['filepath'] = resolve_input_path(
                user_value=user_filepath,
                default_path=None,
                label='ERA5 directory (--filepath)',
                on_derecho=on_derecho,
                required=True,
                check_file=False,
                check_dir=True,
            )
            paths['filedata'] = ResolvedPath(None, PathSource.NOT_SET, 'ERA5 single file')
        else:
            # No user path → try Derecho default for era5_lsm/delta/wetness (need directory)
            # For basic era5, also try Derecho default
            if mode != Mode.ERA5 or on_derecho:
                paths['filepath'] = resolve_input_path(
                    user_value=None,
                    default_path=_DERECHO_DEFAULT_FILEPATH,
                    label='ERA5 directory',
                    on_derecho=on_derecho,
                    required=True,
                    check_file=False,
                    check_dir=True,
                )
                paths['filedata'] = ResolvedPath(None, PathSource.NOT_SET, 'ERA5 single file')
            else:
                raise ValueError(
                    "No ERA5 data path provided.\n"
                    "  Please provide either:\n"
                    "    --filepath  (NCAR-style directory, e.g., /glade/.../e5.oper.an.sfc)\n"
                    "    --filedata  (single ERA5 file, e.g., ERA5_soil_2017.nc)\n"
                    f"  You are not on Derecho, so default paths are not available."
                )
    
    # --- era5_lsm: also needs LSM path ---
    if mode == Mode.ERA5_LSM:
        paths['era5_lsm_path'] = resolve_input_path(
            user_value=getattr(args, 'era5_lsm_path', ''),
            default_path=_DERECHO_DEFAULT_LSM_PATH,
            label='ERA5 LSM file (--era5_lsm_path)',
            on_derecho=on_derecho,
            required=True,
            check_file=True,
        )
    
    # --- era5_delta: also needs delta file ---
    if mode == Mode.ERA5_DELTA:
        paths['delta_file'] = resolve_input_path(
            user_value=getattr(args, 'delta_file', ''),
            default_path=None,  # No default for delta file
            label='Delta SKT file (--delta_file)',
            on_derecho=on_derecho,
            required=True,
            check_file=True,
        )
    
    # --- era5_wetness: also needs porosity file ---
    if mode == Mode.ERA5_WETNESS:
        paths['porosity_file'] = resolve_input_path(
            user_value=getattr(args, 'porosity_file', ''),
            default_path=_DERECHO_DEFAULT_POROSITY_PATH,
            label='Porosity file (--porosity_file)',
            on_derecho=on_derecho,
            required=True,
            check_file=True,
        )
    
    # --- GFS mode ---
    if mode == Mode.GFS:
        paths['filedata'] = resolve_input_path(
            user_value=getattr(args, 'filedata', ''),
            default_path=None,
            label='GFS GRIB2 file (--filedata)',
            on_derecho=on_derecho,
            required=True,
            check_file=True,
        )
    
    # --- HRRR mode ---
    if mode == Mode.HRRR:
        paths['filedata'] = resolve_input_path(
            user_value=getattr(args, 'filedata', ''),
            default_path=None,
            label='HRRR GRIB2 file (--filedata)',
            on_derecho=on_derecho,
            required=True,
            check_file=True,
        )

    # --- CESM CC mode ---
    if mode == Mode.CESM_CC:
        paths['base_bin'] = resolve_input_path(
            user_value=getattr(args, 'base_bin', ''),
            default_path=None,
            label='Base binary file (--base_bin)',
            on_derecho=on_derecho,
            required=True,
            check_file=True,
        )
        paths['cesm_current'] = resolve_input_path(
            user_value=getattr(args, 'cesm_current', ''),
            default_path=None,
            label='CESM current file (--cesm_current)',
            on_derecho=on_derecho,
            required=True,
            check_file=True,
        )
        paths['cesm_future'] = resolve_input_path(
            user_value=getattr(args, 'cesm_future', ''),
            default_path=None,
            label='CESM future file (--cesm_future)',
            on_derecho=on_derecho,
            required=True,
            check_file=True,
        )
        # Output binary (required, but doesn't need to exist)
        paths['output_bin'] = resolve_output_path(
            user_value=getattr(args, 'output_bin', ''),
            label='Output binary (--output_bin)',
        )
        # Landfrac is resolved here as part of unified path handling.
        landfrac_val = getattr(args, 'landfrac', '')
        if landfrac_val and landfrac_val.strip():
            paths['landfrac'] = resolve_input_path(
                user_value=landfrac_val,
                default_path=None,
                label='Landfrac file (--landfrac)',
                on_derecho=on_derecho,
                required=True,
                check_file=True,
            )
        else:
            paths['landfrac'] = infer_landfrac_path(paths['cesm_current'].path)
    
    return ResolvedModePaths(paths=paths)


# =============================================================================
# SECTION 19: Mode Dispatcher
# =============================================================================

def dispatch_mode(mode: Mode, args: argparse.Namespace, resolved: ResolvedModePaths,
                  netcdf_out: bool):
    """Dispatch to the appropriate mode function."""
    
    if mode == Mode.ERA5:
        return run_era5(
            grid=args.grid,
            date_int=args.date,
            filepath=resolved.get('filepath'),
            filedata=resolved.get('filedata'),
            indir=args.indir,
            outdir=args.outdir,
            outdir_nc=args.outdir_nc,
            netcdf_out=netcdf_out,
        )
    
    elif mode == Mode.ERA5_LSM:
        return run_era5_lsm(
            grid=args.grid,
            date_int=args.date,
            filepath=resolved.get('filepath'),
            era5_lsm_path=resolved.get('era5_lsm_path'),
            indir=args.indir,
            outdir=args.outdir,
            outdir_nc=args.outdir_nc,
            netcdf_out=netcdf_out,
        )
    
    elif mode == Mode.ERA5_DELTA:
        return run_era5_delta(
            grid=args.grid,
            date_int=args.date,
            filepath=resolved.get('filepath'),
            delta_file=resolved.get('delta_file'),
            dataset_tag=args.dataset_tag,
            indir=args.indir,
            outdir=args.outdir,
            outdir_nc=args.outdir_nc,
            netcdf_out=netcdf_out,
        )
    
    elif mode == Mode.ERA5_WETNESS:
        return run_era5_wetness(
            grid=args.grid,
            date_int=args.date,
            filepath=resolved.get('filepath'),
            porosity_file=resolved.get('porosity_file'),
            indir=args.indir,
            outdir=args.outdir,
            outdir_nc=args.outdir_nc,
            netcdf_out=netcdf_out,
        )
    
    elif mode == Mode.GFS:
        return run_gfs(
            grid=args.grid,
            date_int=args.date,
            gfs_file=resolved.get('filedata'),
            indir=args.indir,
            outdir=args.outdir,
            outdir_nc=args.outdir_nc,
            netcdf_out=netcdf_out,
        )
    
    elif mode == Mode.HRRR:
        return run_hrrr(
            grid=args.grid,
            date_int=args.date,
            hrrr_file=resolved.get('filedata'),
            indir=args.indir,
            outdir=args.outdir,
            outdir_nc=args.outdir_nc,
            netcdf_out=netcdf_out,
        )

    elif mode == Mode.CESM_CC:
        return run_cesm_cc(
            grid=args.grid,
            base_bin_file=resolved.get('base_bin'),
            output_bin_file=resolved.get('output_bin'),
            cesm_current_file=resolved.get('cesm_current'),
            cesm_future_file=resolved.get('cesm_future'),
            landfrac_file=resolved.get('landfrac'),
            nday=args.nday,
            netcdf_out=netcdf_out,
            outdir_nc=args.outdir_nc,
            indir=args.indir,
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# SECTION 20: CLI Argument Parser
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    p = argparse.ArgumentParser(
        description="Soil initialization for SAM/gSAM (v7 — simplified modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  era5           Standard ERA5 soil initialization
  era5_lsm       ERA5 with LSM mask and extended fill
  era5_delta     ERA5 + climate delta (PGW)
  era5_wetness   ERA5 → soil wetness (gSAM format)
  gfs            GFS GRIB2 data
  hrrr           HRRR GRIB2 data (projected or regular grid)
  cesm_cc        CESM climate change overlay

Public interface notes:
  - Use only the simplified modes listed above.
  - Legacy names such as era5_ncar are still accepted for compatibility,
    but they are deprecated and intentionally omitted from this help text.
  - Mode 'auto' is no longer supported.

Data Path Priority:
  1. User-provided path (--filepath or --filedata) → MUST exist
  2. If the user path is missing AND you are on Derecho → use the built-in default
  3. If not on Derecho and no user path → ERROR

For ERA5 modes, you can provide data in two ways:
  --filepath  : NCAR RDA directory structure (e.g., /glade/.../e5.oper.an.sfc)
  --filedata  : Single ERA5 file (e.g., ERA5_soil_Sep2017.nc)
  If both are provided, --filedata takes priority over --filepath.

For HRRR mode:
  --filedata  : HRRR GRIB2 file (required, no default path)
  The grid geometry (regular lat-lon vs projected) is detected automatically.
  Projected grids (e.g., Lambert Conformal) require scipy for regridding.

On Derecho:
  --filepath and --era5_lsm_path can be omitted (defaults used)
  --porosity_file still requires an explicit path unless you add a reliable default.

Examples:
  python make_2D_init_soil.py --mode era5 --grid 3456x3456_LI --date 2025090309
  python make_2D_init_soil.py --mode era5 --grid 3456x3456_LI --date 2025090309 \
      --filepath /path/to/era5/data
  python make_2D_init_soil.py --mode era5 --grid 3840x1920_dyvar --date 2017090500 \
      --filedata ./ERA5/ERA5_SOIL_SNOW_Sep5_2017.nc
  python make_2D_init_soil.py --mode hrrr --grid 3456x3456_LI --date 2025090309 \
      --filedata ./hrrr.t09z.wrfsubhf00.grib2
"""
    )

    valid_modes = [m.value for m in Mode]
    p.add_argument('--mode', required=True,
                   help=f'Processing mode: {", ".join(valid_modes)}')
    p.add_argument('--grid', required=True,
                   help='Target grid name (e.g., 3456x3456_LI)')

    p.add_argument('--date', type=int, default=None,
                   help='Date as YYYYMMDDHH (not required for cesm_cc; optional for hrrr single-time files)')

    p.add_argument('--filepath', default='',
                   help='ERA5 data directory (NCAR RDA style)')
    p.add_argument('--filedata', default='',
                   help='Single data file (ERA5, GFS, or HRRR GRIB2)')
    p.add_argument('--era5_lsm_path', default='',
                   help='ERA5 land-sea mask file')
    p.add_argument('--porosity_file', default='',
                   help='Soil porosity file')
    p.add_argument('--delta_file', default='',
                   help='Climate delta file (PGW)')

    p.add_argument('--base_bin', default='',
                   help='Base binary file (cesm_cc)')
    p.add_argument('--output_bin', default='',
                   help='Output binary file (cesm_cc)')
    p.add_argument('--cesm_current', default='',
                   help='CESM current climate file')
    p.add_argument('--cesm_future', default='',
                   help='CESM future climate file')
    p.add_argument('--landfrac', default='',
                   help='Land fraction file (cesm_cc). If omitted, an inferred ../landfrac.nc path is tried during path resolution.')
    p.add_argument('--nday', type=int, default=122,
                   help='Day index for CESM data')

    p.add_argument('--dataset_tag', default='era5_DELTA',
                   help='Dataset tag for delta mode output')

    p.add_argument('--indir', default='NC_D',
                   help='Input directory for landmask')
    p.add_argument('--outdir', default='BIN_D',
                   help='Output directory for binary')
    p.add_argument('--outdir_nc', default='NC_D',
                   help='Output directory for NetCDF')
    p.add_argument('--no_netcdf', action='store_true',
                   help='Skip NetCDF output')

    p.add_argument('--interp_backend', default='python_ncl',
                   choices=['python_ncl', 'python_ncl_fast', 'scipy'],
                   help='Interpolation backend')
    p.add_argument('--fill_backend', default='python_ncl',
                   choices=['python_ncl', 'python_ncl_fast', 'scipy'],
                   help='Poisson fill backend')
    p.add_argument('--fill_interp_nan', action='store_true',
                   help='Fill post-interpolation NaNs with nearest neighbors')
    p.add_argument('--strict_time', action='store_true',
                   help='Require an exact time match')

    return p



# =============================================================================
# SECTION 21: Main Entry Point
# =============================================================================

def main():
    """
    Main entry point.

    Flow:
    1. Check dependencies
    2. Parse CLI arguments
    3. Detect environment
    4. Normalize mode
    5. Resolve all paths
    6. Dispatch to the mode function
    """
    global CONFIG

    print("=" * 60)
    print("make_2D_init_soil.py v7")
    print("=" * 60)

    deps = check_dependencies()
    validate_core_dependencies(deps)
    do_delayed_imports()

    parser = create_argument_parser()
    args = parser.parse_args()

    on_derecho = is_on_derecho()

    CONFIG = RuntimeConfig(
        interp_backend=args.interp_backend,
        fill_backend=args.fill_backend,
        fill_interp_nan=args.fill_interp_nan,
        strict_time=args.strict_time,
        on_derecho=on_derecho,
    )
    CONFIG.log()
    log_dependency_status(deps)

    mode = normalize_mode(args.mode)
    validate_mode_dependencies(mode, deps)

    if mode not in (Mode.CESM_CC, Mode.HRRR) and args.date is None:
        raise ValueError(f"--date is required for mode '{mode.value}'")

    print(f"  Mode: {mode.value}")
    if args.date:
        print(f"  Date: {args.date}")
    print(f"  Grid: {args.grid}")

    try:
        resolved = resolve_mode_paths(mode, args, on_derecho)
        resolved.log_all()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    netcdf_out = not args.no_netcdf
    print("-" * 60)

    try:
        dispatch_mode(mode, args, resolved, netcdf_out)
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        raise

    print("-" * 60)
    print("Done.")



if __name__ == '__main__':
    main()
