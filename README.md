# GNATSS

[![image](https://img.shields.io/pypi/v/gnatss.svg)](https://pypi.python.org/pypi/gnatss)
[![BSD License](https://badgen.net/badge/license/BSD-3-Clause/blue)](LICENSE)
[![ssec](https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic)](https://escience.washington.edu/offshore-geodesy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11663059.svg)](https://doi.org/10.5281/zenodo.11663059)

[![CI](https://github.com/seafloor-geodesy/gnatss/actions/workflows/ci.yaml/badge.svg)](https://github.com/seafloor-geodesy/gnatss/actions/workflows/ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/gnatss/badge/?version=latest)](https://gnatss.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/seafloor-geodesy/gnatss/graph/badge.svg?token=XB7S8FYOG7)](https://codecov.io/gh/seafloor-geodesy/gnatss)
<br>
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Nox](https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg)](https://github.com/wntrblm/nox)
[![CodeFactor](https://www.codefactor.io/repository/github/seafloor-geodesy/gnatss/badge)](https://www.codefactor.io/repository/github/seafloor-geodesy/gnatss)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/seafloor-geodesy/gnatss/main.svg)](https://results.pre-commit.ci/latest/github/seafloor-geodesy/gnatss/main)

GNATSS is an open-source software for processing Global Navigation Satellite
Systems - Acoustic (GNSS-A) data for seafloor horizontal positioning. The
software is a redevelopment of existing FORTRAN codes and shell scripts
developed by C. David Chadwell for processing data including measurements made
with Wave Gliders. Existing code, which includes proprietary routines, is
developed and maintained by [John DeSanto](https://github.com/johnbdesanto).

## Using the software

**This software is available via [PyPI](https://pypi.org/), the Python Package
Index**

You can install the software with pip directly by running the following command:

```bash
pip install gnatss
```

Once the software is installed, you should be able to get to the GNATSS Command
Line Interface (CLI) using the command `gnatss`. For example: `gnatss --help`,
will get you to the main GNSS-A Processing in Python help page.

```console

 Usage: gnatss [OPTIONS] COMMAND [ARGS]...

 GNSS-A Processing in Python

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --version                     Show version and exit.                                                                                                            │
│ --install-completion          Install completion for the current shell.                                                                                         │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                  │
│ --help                        Show this message and exit.                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ run   Runs the full pre-processing routine for GNSS-A                                                                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## Pre-processing routines

Currently there's a single command available in the CLI, `run`, which will run
the full pre-processing routine for GNSS-A. You can retrieve the helper text for
this command by running `gnatss run --help`.

```console

 Usage: gnatss run [OPTIONS] CONFIG_YAML

 Runs the full pre-processing routine for GNSS-A
 Note: Currently only supports 3 transponders

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    config_yaml      TEXT  Custom path to configuration yaml file. **Currently only support local files!** [default: None] [required]                          │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --extract-dist-center        --no-extract-dist-center               Flag to extract distance from center from run. [default: extract-dist-center]               │
│ --extract-process-dataset    --no-extract-process-dataset           Flag to extract process results. [default: extract-process-dataset]                         │
│ --outlier-threshold                                          FLOAT  Threshold for allowable percentage of outliers before raising a runtime error.              │
│                                                                     [default: None]                                                                             │
│ --distance-limit                                             FLOAT  Distance in meters from center beyond which points will be excluded from solution. Note     │
│                                                                     that this will override the value set as configuration.                                     │
│                                                                     [default: None]                                                                             │
│ --residual-limit                                             FLOAT  Maximum residual in centimeters beyond which data points will be excluded from solution.    │
│                                                                     Note that this will override the value set as configuration.                                │
│                                                                     [default: None]                                                                             │
│ --qc                         --no-qc                                Flag to plot residuals from run and store in output folder. [default: qc]                   │
│ --from-cache                 --no-from-cache                        Flag to load the GNSS-A Level-2 Data from cache. [default: no-from-cache]                   │
│ --remove-outliers            --no-remove-outliers                   Flag to execute removing outliers from the GNSS-A Level-2 Data before running the solver    │
│                                                                     process.                                                                                    │
│                                                                     [default: no-remove-outliers]                                                               │
│ --run-all                    --no-run-all                           Flag to run the full end-to-end GNSS-A processing routine. [default: run-all]               │
│ --solver                     --no-solver                            Flag to run the solver process only. Requires GNSS-A Level-2 Data. [default: no-solver]     │
│ --posfilter                  --no-posfilter                         Flag to run the posfilter process only. Requires GNSS-A Level-1 Data Inputs.                │
│                                                                     [default: no-posfilter]                                                                     │
│ --help                                                              Show this message and exit.                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

_Currently the pre-processing routine have been tested to only supports 3
transponders, but this will be expanded in the future._

### Configuration yaml file

The run command takes in a configuration yaml file, which is used to configure
the pre-processing routine. **You will need to provide a config yaml file when
calling `gnatss run`.**

Here's a sample configuration yaml file:

```yaml
site_id: SITE #Site Identifier
campaign: Region #Geographical region/Subduction Zone
time_origin: YYYY-MM-DD 00:00:00 #Time of survey
array_center:
  lat: xx.yyyy #decimal latitude
  lon: xxx.yyyy #decimal longitude
transponders: # list out all transponder and info, each entry is a different transponder (default: 3 transponders)
  - lat: xx.yyyyyyyyyy #decimal latitude
    lon: xx.yyyyyyyyyy #decimal longitude
    height: -zzzz.zz #transponder depth (m, positive up)
    internal_delay: t.tttt #Transponder Turn-Around Time (s)
    sv_mean: vvvv.vvv #Estimate of mean sound velocity (m/s)
  - lat: xx.yyyyyyyyyy #decimal latitude
    lon: xx.yyyyyyyyyy #decimal longitude
    height: -zzzz.zz #transponder depth (m, positive up)
    internal_delay: t.tttt #Transponder Turn-Around Time (s)
    sv_mean: vvvv.vvv #Estimate of mean sound velocity (m/s)
  - lat: xx.yyyyyyyyyy #decimal latitude
    lon: xx.yyyyyyyyyy #decimal longitude
    height: -zzzz.zz #transponder depth (m, positive up)
    internal_delay: t.tttt #Transponder Turn-Around Time (s)
    sv_mean: vvvv.vvv #Estimate of mean sound velocity (m/s)
travel_times_variance: 1e-10 #Default value
travel_times_correction: 0.0 #Default value
transducer_delay_time: 0.0 #Default value

# Main input files
input_files:
  travel_times: #Assume Chadwell format, (Time at Ping send [DD-MON-YY HH:MM:SS.ss], TWTT1 (microseconds), TWTT2, TWTT3, TWTT4), TWTT=0 if no reply
    path: /path/to/pxp_tt

# Posfilter configuration
posfilter:
  export:
    full: false #false for only required fields, true to include optional RPH value and uncertainties
  atd_offsets:
    forward: 0.0053 #Value for SV3 Wave Glider
    rightward: 0 #Value for SV3 Wave Glider
    downward: 0.92813 #Value for SV3 Wave Glider
  input_files:
    novatel:
      path: /path/to/file #File with INSPVAA strings
    novatel_std:
      path: /path/to/file #File with INSSTDEVA strings
    gps_positions: #Assume Chadwell format, (j2000 seconds, "GPSPOS" string, ECEF XYZ coordinates (m), XYZ Standard Deviations)
      path: /path/to/GPS_POS_FREED #File path to antenna positions, use wildcards ** for day-separated data

# Solver configuration
solver:
  reference_ellipsoid: #These values should be constant unless the Earth changes
    semi_major_axis: 6378137.000
    reverse_flattening: 298.257222101
  gps_sigma_limit: 0.05 #Uncertainty threshold for transducer positions, data with larger uncertainties ignored
  std_dev: true #true=standard deviation, false=covariance, probably deprecated
  geoid_undulation: xx.yy #Geoid height in m
  bisection_tolerance: 1e-10 #Do not change
  harmonic_mean_start_depth: -4.0 #Shallowest water depth for calculating mean soundvelocity from CTD data
  input_files:
    sound_speed: #Assume 2-column text file with depth (m), sound velocity (m/s)
      path: /path/to/file
    # deletions: # Path to deletns.dat deletions file used by Chadwell code as well
    #   path: ../tests/data/2022/NCL1/deletns.dat
    #gps_solution: #Path to pre-processed input data in standard GNSS-A data format, this skips the Posfilter step
    #  path: ../gps_solution.csv
    #quality_control:
    #  path: /Users/lsetiawan/Repos/SSEC/offshore-geodesy/tests/data/2022/NCL1/quality_control.csv

# Output configuration
output: # Directory path to output directory
  path: /path/to/output/
```

## Contributing

Please refer to our [Contributing Guide](.github/CONTRIBUTING.md) on how to
setup your environment to contribute to this project.

Thanks to our contributors so far!

[![Contributors](https://contrib.rocks/image?repo=seafloor-geodesy/gnatss)](https://github.com/seafloor-geodesy/gnatss/graphs/contributors)

## Open source licensing

This has a **BSD-3-Clause License**, which can be found [here](LICENSE).
