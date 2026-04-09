# Soil Init Validation Summary

This note summarizes the issues found while testing whether
`/glade/u/home/songchenl/tools/make_2D_init_soil.py` can replace the legacy
NCL soil-initialization scripts in
`/glade/u/home/songchenl/models/gSAM/gSAM1.8.7/GLOBAL_DATA`.

## Overall status

- `era5` NCAR directory mode: validated
- `era5_wetness`: validated
- `cesm_cc`: validated on synthetic test inputs
- `era5_lsm`: not validated, significant mismatch found
- single-file `era5` (`era5_notncar` path): not validated
- `era5_delta`: not validated
- `gfs`: not validated

## Validated replacements

### 1. `era5` NCAR directory mode

- Compared Python `--mode era5` against legacy
  `make_2D_init_soil_era5_ncar.ncl`
- Test case:
  - `grid=3kmDerechoAug10`
  - `date=2020081000`
  - ERA5 source: NCAR RDA directory layout
- Result:
  - NetCDF outputs matched exactly
  - binary outputs matched exactly

Conclusion:
- The Python `era5` mode is a validated replacement for the tested
  `era5_ncar` workflow.

### 2. `era5_wetness`

- Compared Python `--mode era5_wetness` against a local NCL harness that
  reproduces the legacy wetness workflow using real ERA5 data and the real
  ERA5 `SLT` invariant file.
- Result:
  - `soilt` matched exactly
  - `soilw` differed only at float roundoff level
  - max abs diff in `soilw`: `1.192093e-07`

Conclusion:
- The Python `era5_wetness` mode is a validated replacement for the tested
  workflow.

### 3. `cesm_cc`

- Compared Python `--mode cesm_cc` against a local NCL harness using
  synthetic CESM and land-fraction inputs plus the real base soil-init binary.
- Result:
  - `dts_grid` matched exactly
  - output binary matched exactly

Conclusion:
- The Python `cesm_cc` mode is a validated replacement for the tested code
  path.

## Problems found

### 1. `era5_lsm` does not match the NCL result

This is the main replacement blocker.

Comparison against the local NCL harness showed:

- `soilt` max abs diff: about `5.67` to `8.03 K`
- `soilw` max abs diff: about `0.039` to `0.047`
- NaN masks matched, so this is not just a missing-value layout problem

Observed comparison output:

- `soilt[0]`: max abs diff `5.665039e+00`
- `soilt[1]`: max abs diff `6.212860e+00`
- `soilt[2]`: max abs diff `7.199524e+00`
- `soilt[3]`: max abs diff `8.028229e+00`
- `soilw[0]`: max abs diff `4.129834e-02`
- `soilw[1]`: max abs diff `4.108648e-02`
- `soilw[2]`: max abs diff `3.949273e-02`
- `soilw[3]`: max abs diff `4.733375e-02`

Likely cause areas to inspect in the Python implementation:

- crop window logic versus NCL coordinate-subsetting semantics
- Poisson-fill setup in LSM mode
  - cyclic versus non-cyclic handling
  - fill-value handling
  - exact domain passed into the fill
- longitude handling before or after crop/fill/interpolation
- interpolation inputs after LSM masking and crop

Conclusion:
- The Python `era5_lsm` mode should not yet be treated as a replacement for
  the NCL workflow.

### 2. single-file `era5` (`era5_notncar` path) is not validated

I attempted to compare the Python single-file path with a synthetic ERA5
single-file dataset. That comparison did not match.

Observed binary summary:

- `soilt` max abs diff: `31.393799`
- `soilw` max abs diff: `0.5100602`
- NaN masks differed between Python and NCL outputs

Important limitation:

- This test used synthetic input data, not the original real ERA5 single-file
  dataset referenced by the legacy NCL script.
- Because of that, this result is evidence that the path is not yet proven,
  but it is not strong enough to conclude the Python implementation is wrong.

Conclusion:
- Do not claim replacement for the single-file ERA5 workflow yet.
- It needs a comparison against a real single-file ERA5 input.

### 3. `era5_delta` is still unvalidated

I was able to run the Python delta mode with a synthetic delta file, but the
local NCL harness used for synthetic testing hit a dimension-mismatch issue.

Implication:

- There is not yet a trustworthy Python-vs-NCL comparison result for
  `era5_delta`.

Conclusion:
- Do not claim replacement for `era5_delta` yet.

### 4. `gfs` is still unvalidated

The Python `gfs` mode first attempts `cfgrib` loading. My synthetic test input
was NetCDF, not GRIB2, so it is not a faithful replacement test for the real
legacy workflow.

Implication:

- The current environment did not provide the real GFS GRIB2 input referenced
  by the legacy script.
- The synthetic stand-in is not sufficient to validate replacement.

Conclusion:
- Do not claim replacement for `gfs` yet.

## Practical conclusion

At this point, the Python script can replace only the modes that were actually
validated:

- `era5`
- `era5_wetness`
- `cesm_cc` for the tested code path

It should not yet be presented as a full replacement for all legacy soil-init
NCL scripts, because:

- `era5_lsm` has a real mismatch
- single-file `era5` is not yet proven
- `era5_delta` is not yet proven
- `gfs` is not yet proven

## Recommended next step

The highest-value next step is to debug `era5_lsm`, because that is the only
mode that clearly produced a substantive Python-vs-NCL regression during
testing.