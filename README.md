# GNATSS

[![ssec](https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic)](https://escience.washington.edu/offshore-geodesy/)
[![BSD License](https://badgen.net/badge/license/BSD-3-Clause/blue)](LICENSE)

[![CI](https://github.com/seafloor-geodesy/gnatss/actions/workflows/ci.yaml/badge.svg)](https://github.com/seafloor-geodesy/gnatss/actions/workflows/ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/gnatss/badge/?version=latest)](https://gnatss.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/seafloor-geodesy/gnatss/graph/badge.svg?token=XB7S8FYOG7)](https://codecov.io/gh/seafloor-geodesy/gnatss)
<br>
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CodeFactor](https://www.codefactor.io/repository/github/seafloor-geodesy/gnatss/badge)](https://www.codefactor.io/repository/github/seafloor-geodesy/gnatss)

GNATSS is an open-source software for processing Global Navigation Satellite Systems - Acoustic (GNSS-A) data for seafloor horizontal positioning.
The software is a redevelopment of existing FORTRAN codes and shell scripts developed by C. David Chadwell for processing data including measurements made with Wave Gliders.
Existing code, which includes proprietary routines, is developed and maintained by [John DeSanto](https://github.com/johnbdesanto).

## Using the software

**This software is currently under heavy development and is not available via [PyPI](https://pypi.org/), the Python Package Index**

You can install the software with pip directly by running the following command:

```bash
pip install gnatss
```

Once the software is installed, you should be able to get to the GNATSS Command Line Interface (CLI)
using the command `gnatss`. For example: `gnatss --help`, will get you to the main GNSS-A Processing in Python help page.

```console
 Usage: gnatss [OPTIONS] COMMAND [ARGS]...

 GNSS-A Processing in Python

╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion        [bash|zsh|fish|powershell|pwsh]  Install completion for the specified │
│                                                              shell.                               │
│                                                              [default: None]                      │
│ --show-completion           [bash|zsh|fish|powershell|pwsh]  Show completion for the specified    │
│                                                              shell, to copy it or customize the   │
│                                                              installation.                        │
│                                                              [default: None]                      │
│ --help                                                       Show this message and exit.          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────╮
│ run      Runs the full pre-processing routine for GNSS-A                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Pre-processing solve routine

Currently there's a single command available in the CLI, `run`, which will run the full pre-processing routine for GNSS-A.
You can retrieve the helper text for this command by running `gnatss run --help`.

```console
 Usage: gnatss run [OPTIONS]

 Runs the full pre-processing routine for GNSS-A
 Note: Currently only supports 3 transponders

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────╮
│ --config-yaml                                              TEXT   Custom path to configuration │
│                                                                   yaml file. **Currently only  │
│                                                                   support local files!**       │
│                                                                   [default: None]              │
│ --extract-dist-center         --no-extract-dist-center            Flag to extract distance     │
│                                                                   from center from run.        │
│                                                                   [default:                    │
│                                                                   extract-dist-center]         │
│ --extract-process-dataset     --no-extract-process-data           Flag to extract process      │
│                                                                   results.                     │
│                                                                   [default:                    │
│                                                                   extract-process-dataset]     │
│ --qc                          --no-qc                             Flag to plot residuals from  │
│                                                                   run and store in             │
│                                                                   output folder.               │
│                                                                   [default: qc]                │
│ --distance-limit                                           FLOAT  Distance in meters from      │
│                                                                   center beyond which points   │
│                                                                   will be excluded from        │
│                                                                   solution. Note that this     │
│                                                                   will override the value set  │
│                                                                   as configuration.            │
│                                                                   [default: None]              │
│ --residual-limit                                           FLOAT  Maximum residual in          │
│                                                                   centimeters beyond which     │
│                                                                   data points will be excluded │
│                                                                   from solution. Note that     │
│                                                                   this will override the value │
│                                                                   set as configuration.        │
│                                                                   [default: None]              │
│ --help                                                            Show this message and exit.  │
╰────────────────────────────────────────────────────────────────────────────────────────────────╯
```

*Currently the pre-processing routine have been tested to only supports 3 transponders, but this will be expanded in the future.*

### Configuration yaml file

The run command takes in a configuration yaml file, which is used to configure the pre-processing routine.
By default, the program will look for a configuration file in the current working directory called `config.yaml`.
If this file is found somewhere else, you can pass in the path to the file using the `--config-yaml` flag.

Here's a sample configuration yaml file:

```yaml
site_id: SITE1

solver:
  # note that the coordinates are not real and are just for example
  transponders:
    - lat: 47.302064471
      lon: -126.978181346
      height: -1176.5866
      internal_delay: 0.200000
      sv_mean: 1481.551
    - lat: 47.295207747
      lon: -126.958752845
      height: -1146.5881
      internal_delay: 0.320000
      sv_mean: 1481.521
    - lat: 47.309643593
      lon: -126.959348875
      height: -1133.7305
      internal_delay: 0.440000
      sv_mean: 1481.509
  reference_ellipsoid:
    semi_major_axis: 6378137.000
    reverse_flattening: 298.257222101
  gps_sigma_limit: 0.05
  std_dev: true
  geoid_undulation: -26.59
  bisection_tolerance: 1e-10
  array_center:
    lat: 47.3023
    lon: -126.9656
  travel_times_variance: 1e-10
  travel_times_correction: 0.0
  transducer_delay_time: 0.0
  harmonic_mean_start_depth: -4.0
  input_files:
    sound_speed:
      path: /path/to/CTD_NCL1_Ch_Mi
    travel_times:
      path: /path/to/**/WG_*/pxp_tt # this option will take in glob patterns
    gps_solution:
      path: /path/to/**/posfilter/POS_FREED_TRANS_TWTT # this option will take in glob patterns

output:
  path: /my/output/dir/
```

### Deletions file

This will output the final resulting deletions file to the output directory specified in the configuration yaml file.
This file will be in Comma Separated Value (CSV) format called `deletions.csv`.

### Residual file

This will output the final resulting residual file to the output directory specified in the configuration yaml file.
This file will be in Comma Separated Value (CSV) format called `residuals.csv`.

## Contributing

Please refer to our [Contributing Guide](.github/CONTRIBUTING.md) on how to setup your environment to contribute to this project.

Thanks to our contributors so far!

[![Contributors](https://contrib.rocks/image?repo=seafloor-geodesy/gnatss)](https://github.com/seafloor-geodesy/gnatss/graphs/contributors)

## Open source licensing

This has a **BSD-3-Clause License**, which can be found [here](LICENSE).
