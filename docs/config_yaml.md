# GNATSS Config File

The GNSS-Acoustic positioning performed by GNATSS is governed by a configuration
file named "config.yaml". This is a text file with a YAML format that defines
the a priori geometry of the GNSS-Acoustic array.

The config file is divided into multiple sections, but not all of the sections
are required depending on your processing needs. In general, GNATSS operates in
two modes, a posfilter mode and a solver mode, and these modes are defined
separately in the config file. It is possible to run only one of these two modes
at a time, or to run both in sequence, so it is optional to include their
information in the config file.

At a high level, the sections of the config file are:

- Metadata _(required)_: Information about the array
- Main Input files _(required for posfilter)_: Travel time files
- Posfilter configuration _(optional)_: Pre-processing configuration
- Solver _(optional)_: Array positioning configuration
- Output configuration _(required)_: Destination for output files

A template of the config.yaml file is:

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
  harmonic_mean_start_depth: -4.0 #Shallowest water depth for calculating mean soundvelocity from CTD data
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

- **Transponders** A priori information for each transponder in the array. The
  following information must be provided:
  - lat: The latitude of the transponder in decimal degrees
  - lon: The longitude of the transponder in decimal degrees
  - height: The ellipsoidal height of the transponder in meters (negative down)
  - internal_delay: The user-defined transponder wait time in seconds. When the
    transponder receives an interrogation ping, it will wait for this amount of
    time before sending a reply ping in order to avoid interference between the
    replies from the array.
  - sv_mean: An initial estimate of the average sound velocity throughout the
    water column above the transponder in m/s
- **Reference Ellipsoid** Ellipsoidal parameters of the Earth. These should not
  have to change.
- **GPS Sigma Limit** This is an uncertainty threshold in meters. If the
  uncertainty of the surface platform position crosses this threshold, the data
  will not be considered in the GNSS-A positioning due to poor positioning
- **Std_dev** Determines the uncertainty parameters of the surface platform
  positions. If _std_dev=true_, the position uncertainties are assumed to be
  standard deviations. If _std_dev=false_, the uncertainties are assumed to be
  variances.
- **Geoid Undulation** The local geoid height at the center of the GNSS-Acoustic
  array in meters.
- **Bisection Tolerance** Tolerance parameter for acoustic raytracing
  calculations. Should not need to be changed.
- **Array Center** Latitude and longitude of the array center in decimal
  degrees.
- **Travel Times Variance** The instrument uncertainty of the acoustic
  transponder mounted to the surface platform in s^2.
- **Travel Times Correction** The correction to tabulated acoustic travel times
  in seconds. Should not need to be changed.
- **Transducer Delay Time** The delay time of the surface platform between an
  interrogate command being sent to the transducer and the interrogation ping
  being emitted, in seconds. Different surface platforms will have different
  delays. In general, GNATSS assumes that:
  - The time variable in the _pxp_tt_ file is when the interrogate command is
    sent (before the start of the transducer delay time)
  - The two-way travel times logged in the _pxp_tt_ file do not include the
    transducer delay time since the interrogation ping was not in the water
    during this delay.
  - The internal delay of the seafloor transponders _is_ included in the two-way
    travel times by convention since the interrogation ping is in the water
    during this delay.
  - The transducer positions in the _POS_FREED_TRANS_TWTT_ file are at times
    when the interrogation ping is sent and the replies received, and thus after
    the transducer delay.
  - You can set the transducer delay to zero if you remove the delay time prior
    to running GNATSS. GNATSS will run a cross-check before processing data and
    return an error if the ping timings of the _pxp_tt_ and
    _POS_FREED_TRANS_TWTT_ files do not align.
- **Harmonic Mean Start Depth** GNATSS will automatically recalculate the
  harmonic mean sound velocity for each transponder in the array between the
  transponder depth and the starting depth defined here in meters.
- **Input files** File paths to the CTD, _pxp_tt_, and _POS_FREED_TRANS_TWTT_
  files. You may choose to provide UNIX wildcard characters in the filepaths if
  you chose to prepare the data in daily batches, in which case GNATSS will
  automatically compile them. Alternatively, you may choose to provide single
  _pxp_tt_ and _POS_FREED_TRANS_TWTT_ files.
- **Output Path** File path in which GNATSS will store positioning results.
