# GNATSS Config File

The GNSS-Acoustic positioning performed by GNATSS is governed by a configuration file named "config.yaml". This is a text file with a YAML format that defines the a priori geometry of the GNSS-Acoustic array. An example of the config.yaml file for the GNSS-A array NCL1 offshore Cascadia is:

```
site_id: NCL1

solver:
  transponders: # list out all transponder and info
    - lat: 45.302064471
      lon: -124.978181346
      height: -1176.5866
      internal_delay: 0.200000
      sv_mean: 1481.551
    - lat: 45.295207747
      lon: -124.958752845
      height: -1146.5881
      internal_delay: 0.320000
      sv_mean: 1481.521
    - lat: 45.309643593
      lon: -124.959348875
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
    lat: 45.3023
    lon: -124.9656
  travel_times_variance: 1e-10
  travel_times_correction: 0.0
  transducer_delay_time: 0.0
  harmonic_mean_start_depth: -4.0
  input_files:
    sound_speed:
      path: ./tests/data/2022/NCL1/ctd/CTD_NCL1_Ch_Mi
    travel_times: # Global path to pxp_tt
      path: ./tests/data/2022/NCL1/**/WG_*/pxp_tt
    gps_solution:  # Global path to POS_FREED_TRANS_TWTT
      path: ./tests/data/2022/NCL1/**/posfilter/POS_FREED_TRANS_TWTT


output: # Directory path to output directory
  path: ./tests/data/output/
```

The information that should be defined in the config.yaml file is as follows:

- **Transponders** A priori information for each transponder in the array. The following information must be provided:
  - lat: The latitude of the transponder in decimal degrees
  - lon: The longitude of the transponder in decimal degrees
  - height: The ellipsoidal height of the transponder in meters (negative down)
  - internal_delay: The user-defined transponder wait time in seconds. When the transponder receives an interrogation ping, it will wait for this amount of time before sending a reply ping in order to avoid interference between the replies from the array.
  - sv_mean: An initial estimate of the average sound velocity throughout the water column above the transponder in m/s
- **Reference Ellipsoid** Ellipsoidal parameters of the Earth. These should not have to change.
- **GPS Sigma Limit** This is an uncertainty threshold in meters. If the uncertainty of the surface platform position crosses this threshold, the data will not be considered in the GNSS-A positioning due to poor positioning
- **Std_dev** Determines the uncertainty parameters of the surface platform positions. If *std_dev=true*, the position uncertainties are assumed to be standard deviations. If *std_dev=false*, the uncertainties are assumed to be variances.
- **Geoid Undulation** The local geoid height at the center of the GNSS-Acoustic array in meters.
- **Bisection Tolerance** Tolerance parameter for acoustic raytracing calculations. Should not need to be changed.
- **Array Center** Latitude and longitude of the array center in decimal degrees.
- **Travel Times Variance** The instrument uncertainty of the acoustic transponder mounted to the surface platform in s^2.
- **Travel Times Correction** The correction to tabulated acoustic travel times in seconds. Should not need to be changed.
- **Transducer Delay Time** The delay time of the surface platform between an interrogate command being sent to the transducer and the interrogation ping being emitted, in seconds. Different surface platforms will have different delays. In general, GNATSS assumes that:
  - The time variable in the *pxp_tt* file is when the interrogate command is sent (before the start of the transducer delay time)
  - The two-way travel times logged in the *pxp_tt* file do not include the transducer delay time since the interrogation ping was not in the water during this delay.
  - The internal delay of the seafloor transponders *is* included in the two-way travel times by convention since the interrogation ping is in the water during this delay.
  - The transducer positions in the *POS_FREED_TRANS_TWTT* file are at times when the interrogation ping is sent and the replies received, and thus after the transducer delay.
  - You can set the transducer delay to zero if you remove the delay time prior to running GNATSS. GNATSS will run a cross-check before processing data and return an error if the ping timings of the *pxp_tt* and *POS_FREED_TRANS_TWTT* files do not align.
- **Harmonic Mean Start Depth** GNATSS will automatically recalculate the harmonic mean sound velocity for each transponder in the array between the transponder depth and the starting depth defined here in meters.
- **Input files** File paths to the CTD, *pxp_tt*, and *POS_FREED_TRANS_TWTT* files. You may choose to provide UNIX wildcard characters in the filepaths if you chose to prepare the data in daily batches, in which case GNATSS will automatically compile them. Alternatively, you may choose to provide single *pxp_tt* and *POS_FREED_TRANS_TWTT* files.
- **Output Path** File path in which GNATSS will store positioning results.
