# HRRR Mode Test Result

Date: 2026-04-09

## Summary

Tested the updated `hrrr` mode in `/glade/u/home/songchenl/tools/make_2D_init_soil.py` using:

```text
/glade/derecho/scratch/songchenl/condaEnvs/default/bin/python
```

The new HRRR loader behaves better than the previous version. It no longer fails immediately on an empty filtered dataset. Instead, it falls back to a broader scan and exits with a clear diagnostic when the HRRR file does not contain layered soil fields.

## Command Used

```bash
/glade/derecho/scratch/songchenl/condaEnvs/default/bin/python -u \
  /glade/u/home/songchenl/tools/make_2D_init_soil.py \
  --mode hrrr \
  --grid hrrr_test \
  --date 2020081001 \
  --filedata /glade/u/home/songchenl/tools/hrrr_test_run/hrrr_2020081001_01.grib2 \
  --indir /glade/u/home/songchenl/tools/hrrr_test_run/NC_D \
  --outdir /glade/u/home/songchenl/tools/hrrr_test_run/BIN_D \
  --outdir_nc /glade/u/home/songchenl/tools/hrrr_test_run/NC_D \
  --no_netcdf
```

## Test Inputs

- Script: `/glade/u/home/songchenl/tools/make_2D_init_soil.py`
- Source HRRR file directory: `/glade/u/home/songchenl/PythonScripts/derecho/data/HRRR/T00`
- Test GRIB2 file used for the clean rerun:
  `/glade/u/home/songchenl/tools/hrrr_test_run/hrrr_2020081001_01.grib2`
- Temporary landmask file:
  `/glade/u/home/songchenl/tools/hrrr_test_run/NC_D/landmask_hrrr_test.nc`

## Result

The script started normally, entered HRRR mode, attempted targeted filters, then fell back to scanning all GRIB datasets. It failed with the following runtime error:

```text
[HRRR] Cannot find layered soil temperature or soil moisture fields in:
  /glade/u/home/songchenl/tools/hrrr_test_run/hrrr_2020081001_01.grib2

File diagnostic:
  Total datasets in file: 40
  Soil-related datasets found (but missing required layered fields):
    dataset 6: vars=['mstav'], level=['depthBelowLand']
```

## Interpretation

The new HRRR mode correctly identifies that this file is not a valid layered-soil HRRR product for soil initialization.

Observed behavior:

- Targeted search for layered HRRR soil variables failed
- Broad cfgrib scan found only `mstav`
- The file does not contain the expected 4-layer soil fields such as:
  - `st`
  - `soilw`
  - `typeOfLevel='depthBelowLandLayer'`

This means the `T00` sample file tested here is still not suitable for the success path of HRRR soil initialization.

## Relevant Code Paths

- `/glade/u/home/songchenl/tools/make_2D_init_soil.py:1337`
- `/glade/u/home/songchenl/tools/make_2D_init_soil.py:1426`
- `/glade/u/home/songchenl/tools/make_2D_init_soil.py:1465`
- `/glade/u/home/songchenl/tools/make_2D_init_soil.py:1662`

## Output Files

No soil initialization output was produced.

Verified:

- `/glade/u/home/songchenl/tools/hrrr_test_run/BIN_D` is empty

## Conclusion

The updated HRRR mode now fails for the right reason and with a useful diagnostic. The remaining blocker is the input data product, not the Python environment and not the earlier empty-dataset bug.
