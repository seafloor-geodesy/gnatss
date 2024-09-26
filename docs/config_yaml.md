# GNATSS Config File

The GNSS-Acoustic positioning performed by GNATSS is governed by a configuration
file. This is a text file with a YAML format that defines the _a priori_
geometry of the GNSS-Acoustic array.

The configuration file is divided into multiple sections, but not all of the
sections are required depending on your processing needs. In general, GNATSS
operates in two modes, a posfilter mode which performs pre-processing on wave
glider data to generate input required for array positioning, and a solver mode
that perms the array positioning using the data computed by the posfilter. These
modes are defined separately in the config file. It is possible to run only one
of these two modes at a time, or to run both in sequence, so it is optional to
include their information in the config file.

At a high level, the sections of the configuration file are:

- Metadata _(required)_: Information about the array
- Main Input files _(required for posfilter)_: Travel time files
- Posfilter configuration _(optional)_: Pre-processing configuration
- Solver _(optional)_: Array positioning configuration
- Output configuration _(required)_: Destination for output files

Further information on the input data formats may be found on the
[Required Input Data](./input.md) page.

A template of the configuration file is:

```
# Metadata
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
  harmonic_mean_start_depth: -4.0 #Shallowest water depth for calculating mean sound velocity from CTD data
  input_files:
    sound_speed: #Assume 2-column text file with depth (m), sound velocity (m/s)
      path: /path/to/file
    # deletions: # Path to deletns.dat deletions file used by Chadwell code as well
    #   path: /path/to/deletns.dat
    #gps_solution: /path/to/gps_solution.csv #Path to pre-processed input data in standard GNSS-A data format, this skips the Posfilter step
    #  path: /path/to/gps_solution.csv
    #quality_control:
    #  path: /path/to/quality_control.csv

# Output configuration
output: # Directory path to output directory
  path: /path/to/output/
```

The information that should be defined in the config.yaml file is as follows:

### Metadata

- **Site ID** This is the name of the array, generally denoted by a four-letter
  code
- **Campaign** The geographical region where the array is located, generally a
  subduction zone
- **Time Origin** The approximate date when the survey was conducted. Note that
  surveys may last multiple days, but only one date is required here since the
  data will be averaged into one final position.
- **Array Center** The approximate center of the array.
- **Transponders** A priori information for each transponder in the array.
  GNATSS is currently configured to operate on arrays of 3 transponders. GNATSS
  will assign names to the transponders based upon the order they are included
  in the configuration file. The first entry will be named "SITE-1", the second
  entry "SITE-2", etc. The following information must be provided:
  - lat: The latitude of the transponder in decimal degrees
  - lon: The longitude of the transponder in decimal degrees
  - height: The ellipsoidal height of the transponder in meters (negative down)
  - internal_delay: The user-defined transponder wait time in seconds. When the
    transponder receives an interrogation ping, it will wait for this amount of
    time before sending a reply ping in order to avoid interference between the
    replies from the array.
  - sv_mean: An initial estimate of the average sound velocity throughout the
    water column above the transponder in m/s
- **Travel Time Variance** The assumed uncertainty of the acoustic two-way
  travel times, given as a variance. This value is treated as a constant and
  should not need to be modified.
- **Travel Times Correction** This value is used to adjust the travel times and
  should not need to be changed.
- **Transducer Delay Time** This is a time delay that is introduced by software
  on the wave glider separating when an interrogation ping is registered and
  when the acoustic pulse is sent by the transducer. By default, it is assumed
  that any transducer delay times are accounted for before or during the
  posfilter. Some surface platforms, notably the Sonardyne GNSS-A payload, also
  include this delay in the TWTT measurement despite the acoustic pulse not
  being in the water during the delay. In this case, the user must remove the
  delay from the TWTT measurements before running GNATSS or else the TWTT
  measurements will be systematically inflated by the delay.

### Main input files

- **Travel Times** The TWTT input file is required for the posfilter mode.
  GNATSS assumes this file is in the legacy Chadwell format (See
  [_Required Input Data_](./input.md)).
  - The user may input a file path to a single input file or use the UNIX "\*\*"
    wildcard to point to multiple input files, such as day-separated data.

### Posfilter configuration

- **Export** The _full_ parameter determines whether to include roll, pitch, and
  heading values not required for the solver but useful for recalculating
  transducer positions. The default value of _false_ provides only the minimum
  data fields required for the solver, as defined by the
  [GNSS-Acoustic Standard Data Format](https://hal.science/hal-04319233/).
- **ATD Offsets** The ATD offsets (also called the _lever arms_) are the static
  offsets between the GNSS antenna and acoustic transducer in the static body
  frame coordinates of the surface platform. The given values in the above
  template are the ATD offsets for the SV3 Wave Glider.
- **Input Files** These are file paths to [input data](./input.md) required
  specifically for the posfilter mode, assumed to be collected by an SV3 wave
  glider using a Sonardyne GNSS-A payload.
  - The user may input a file path to a single input file or use the UNIX "\*\*"
    wildcard to point to multiple input files, such as day-separated data.
  - The novatel files contain raw NMEA strings describing the velocity,
    orientation, and standard deviations of the surface platform.
  - The GPS positions are assumed to be computed by the user with a GNSS
    processing software of their choice, such as PRIDE PPP-AR, GAMIT, or GipsyX.
    Regardless of the GNSS software used, GNATSS assumes that the solution has
    been converted into a legacy Chadwell format (See
    [_Required Input Data_](./input.md)).

### Solver configuration

- **Reference Ellipsoid** Ellipsoidal parameters of the Earth. These should not
  have to change.
- **GPS Sigma Limit** This is an uncertainty threshold in meters. If the
  uncertainty of the surface platform position crosses this threshold, the data
  will not be considered in the GNSS-A positioning due to poor positioning.
  - **Std_dev** Determines the uncertainty parameters of the surface platform
    positions. If _std_dev=true_, the position uncertainties are assumed to be
    standard deviations. If _std_dev=false_, the uncertainties are assumed to be
    variances.
  - **Geoid Undulation** The local geoid height at the center of the
    GNSS-Acoustic array in meters.
- **Bisection Tolerance** Tolerance parameter for acoustic raytracing
  calculations. Should not need to be changed.
- **Harmonic Mean Start Depth** GNATSS will automatically recalculate the
  harmonic mean sound velocity for each transponder in the array between the
  transponder depth and the starting depth defined here in meters.
- **Input files** File paths to the input files required for the solver mode.
  These files include:
  - Sound speed file, as described in [_Required Input Data_](./input.md).
  - Deletions file, in legacy Chadwell format. A separate deletions file in
    GNATSS format is automatically generated and updated in the output file
    path.
  - GPS Solution file, containing the transducer positions and TWTTs in the
    [GNSS-Acoustic Standard Data Format](https://hal.science/hal-04319233/).
    Only required if not running the posfilter mode. If running end-to-end
    processing by calling the posfilter and solver module, the gps*solution.csv
    file will be generated in the output path and can be automatically loaded by
    calling the *--from-cache\_ flag when running GNATSS, in which case a file
    path does not need to be designated.
  - Quality Control file

### Output configuration

- **Output Path** File path in which GNATSS will store posfilter and solver
  results.
