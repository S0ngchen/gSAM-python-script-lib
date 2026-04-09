# HRRR Mode Test Result

Date: 2026-04-09

## Summary

Tested `hrrr` mode in `/glade/u/home/songchenl/tools/make_2D_init_soil.py` using the conda environment:

```text
/glade/derecho/scratch/songchenl/condaEnvs/default/bin/python
```

The test failed before producing output files. The immediate failure is that the sample HRRR GRIB2 file does not expose the layered soil fields the script expects.

## Command Used

```bash
/glade/derecho/scratch/songchenl/condaEnvs/default/bin/python \
  /glade/u/home/songchenl/tools/make_2D_init_soil.py \
  --mode hrrr \
  --grid hrrr_test \
  --date 2020081001 \
  --filedata /glade/u/home/songchenl/PythonScripts/derecho/data/HRRR/T00/hrrr_2020081001_01.grib2 \
  --indir /glade/u/home/songchenl/tools/hrrr_test_run/NC_D \
  --outdir /glade/u/home/songchenl/tools/hrrr_test_run/BIN_D \
  --outdir_nc /glade/u/home/songchenl/tools/hrrr_test_run/NC_D \
  --no_netcdf
```

## Test Inputs

- Script: `/glade/u/home/songchenl/tools/make_2D_init_soil.py`
- HRRR sample file: `/glade/u/home/songchenl/PythonScripts/derecho/data/HRRR/T00/hrrr_2020081001_01.grib2`
- Temporary landmask file created for the test:
  `/glade/u/home/songchenl/tools/hrrr_test_run/NC_D/landmask_hrrr_test.nc`

## Result

The script reached HRRR mode, loaded the target grid, and then failed in the HRRR loader. Reported error:

```text
KeyError: "[HRRR] No soil temperature variable found.
  Tried: ['st', 'ST', 'tsoil', 'TSOIL', 'soilt', 'SOILT', 't']
  Available: []"
```

This comes from the variable discovery step in the HRRR loader after `cfgrib` opens an empty dataset.

Relevant code paths:

- `/glade/u/home/songchenl/tools/make_2D_init_soil.py:1517`
- `/glade/u/home/songchenl/tools/make_2D_init_soil.py:1530`
- `/glade/u/home/songchenl/tools/make_2D_init_soil.py:1291`
- `/glade/u/home/songchenl/tools/make_2D_init_soil.py:1342`

## Additional Verification

Checked GRIB metadata in the `T00` directory with `grib_ls`.

Observed pattern:

- Files contain `mstav` at `depthBelowLand`
- Files do not show the expected 4-layer `depthBelowLandLayer` soil fields
- The script currently expects layered HRRR soil variables such as `st` and `soilw`

This indicates the failure is caused by the sample HRRR data content, not by the Python environment.

## Notes

- No binary output was written to `/glade/u/home/songchenl/tools/hrrr_test_run/BIN_D`
- One script usability issue remains: it prints that a soil dataset was "opened" even when the resulting dataset is empty
