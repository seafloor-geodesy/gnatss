# Running GNATSS

## Overview

GNATSS has two processing modes, a "posfilter" mode and a "solver" mode, which
are used at different phases of the processing chain. It is possible to run both
processing modes in sequence or individually. The posfilter module outputs
GNSS-Acoustic travel times and transducer positions in the community standard
GNSS-Acoustic data format, so if you have obtained data already in that format
you may skip the posfilter mode and immediately run the solver.

The basic step-by-step guide to position a GNSS-Acoustic array using GNATSS is:

Step 1: Gather the [required input files](#required-input-data)

Step 2: Create working directory and [configuration file](./config_yaml.md)

Step 3 _(optional)_: Run the posfilter module to generate transducer positions
at the interrogation/reply times of the GNSS-Acoustic survey

```bash
gnatss run --posfilter config.yaml
```

Step 4: Run the solver to calculate an array position

```bash
gnatss run --solver --distance-limit 150 --residual-limit 10000 config.yaml
```

Step 5: Run the solver again to with the _--remove-outliers_ option to remove
flagged residuals.

```bash
gnatss run --solver --distance-limit 150 --remove-outliers config.yaml
```

Step 6: Repeat Steps 4-5, reducing the residual limit as desired in each
successive iteration in order to remove erroneous residuals.

Step 7: Array positions, offsets, and statistics are stored in the
process_dataset.nc file.

_Alternative_: GNATSS supports end-to-end processing, so you may execute Steps 3
and 4 simultaneously. This will generate an array position as well as input to
run the solver again without having to repeat the posfilter analysis. For
repeated calls to the solver module during Step 5, add the _--from-cache_ flag
to skip the posfilter and immediately load gps solutions from your output
directory.

```bash
gnatss run --distance-limit 150 config.yaml
gnatss run --from-cache --distance-limit 150 --residual-limit 10000 config.yaml
gnatss run --from-cache --distance-limit 150 --remove-outliers config.yaml
```

## Required Input Data

GNATSS requires the following data in order to generate an array position:

- Acoustic two-way travel times
- Surface platform positions
- A sound velocity profile
- _(Optional)_ Velocities and Roll, Pitch, Heading values for the surface
  platform

These data may ingested in a Level-1 (L-1) or Level-2 (L-2) data format, with
L-1 data required for the posfilter mode and L-2 data required for the solver.

### L-1 Data

**Acoustic Two-Way Travel Times**

Acoustic two-way travel times are defined as the duration between when the
transducer on the surface platform sends an acoustic interrogation and when it
receives a reply from a seafloor transponder. Multiple transponders may respond
to the same interrogation if they each receive the ping. Two-way travel times
should be stored in a file named “pxp_tt”, with a text column format in which
each row corresponds to a ping, containing UTC timestamps and two-way travel
times in microseconds. Example:

```console
26-JUN-23 00:00:07.00   2241985   2337422   2469996
26-JUN-23 00:00:22.00   2240506   2346387   2463020
26-JUN-23 00:00:37.00   2240627   2353454   2454958
26-JUN-23 00:00:52.00   2243723   2359242   2445547
```

Each of the two-way travel time columns corresponds to the replies from a
specific transponder as defined in the config.yaml file. In order to prevent
interference between the replies, each transponder may be assigned an internal
delay time that will elapse before sending a reply to an interrogation ping.
These delay times must be defined in the config.yaml file and will be removed
during positioning.

There may also be a delay time associated with the surface platform
corresponding to the time between when the computer sends an interrogation
command and when the transducer emits an acoustic pulse. For instance, the
Sonardyne GNSS-A payload has a 0.13 second delay time that is recorded in its
raw two-way travel times. This delay must be removed prior to GNATSS positioning
because there is not an acoustic pulse physically traveling through the water
column during this time.

**Surface Platform Positions**

There are two positions on a GNSS-Acoustic surface platform relevant for array
positioning, the positioning of the GNSS antenna and the position of the
transducer. While GNATSS requires positions for the transducer for array
positioning, raw positioning data in the form of RINEX files track the GNSS
antenna. Thus, the user must provide GNATSS with a time series of antenna
positions from which GNATSS will derive the transducer positions during the
posfilter operation.

The antenna positions may be calculated with a GNSS processing software of the
user's preference. However, in order for GNATSS to ingest the antenna positions
they must be converted into a custom format. This format is a text column file
of the form:

```console
   712497600.0000000    GPSPOS       -2605855.7990    -3707733.0210     4472994.7430    0.0100    0.0100    0.0100
   712497601.0000000    GPSPOS       -2605855.2850    -3707733.4330     4472995.0110    0.0100    0.0100    0.0100
   712497602.0000000    GPSPOS       -2605855.1770    -3707734.0660     4472995.1420    0.0100    0.0100    0.0100
   712497603.0000000    GPSPOS       -2605854.7550    -3707734.4340     4472994.8080    0.0100    0.0100    0.0100
```

The columns of this file are as follows:

- The first column is time in J2000 seconds, relative to 2000-01-01 12:00:00.
- The second column is a text flag.
- Columns 3-5 are the antenna positions in ECEF XYZ coordinates with units of
  meters.
- Columns 6-8 are the standard deviations of the antenna positions, also in
  meters.

**Velocities and Roll, Pitch, Heading values**

GNATSS is currently equipped to ingest raw orientation data collected by the
Sonardyne GNSS-A payload used by the Liquid Robotics model SV-3 Wave Gliders
currently deployed in the
[Near Trench Community Geodetic Experiment](https://www.seafloorgeodesy.org/commexp).
These payloads include a Novatel dual-antenna navigation system that reports the
orientation in NMEA text strings:

- [_#INSPVA_](https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSPVA.htm)
  includes the platform positions, velocities, and orientations
- [_#INSSTDEV_](https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSSTDEV.htm)
  includes the uncertainties for the velocities and orientations

### L-2 Data

L-2 data is generated by the posfilter module or may be acquired independently
of GNATSS. These data should be in the
[Community Standard GNSS-Acoustics Data Format](https://hal.science/hal-04319233/).
These data are logged in a csv format and include the following data and header:

```console
T_receive,MT_ID,TravelTime,T_transmit,ant_cov_XX1,ant_cov_XY1,ant_cov_XZ1,ant_cov_YX1,ant_cov_YY1,ant_cov_YZ1,ant_cov_ZX1,ant_cov_ZY1,ant_cov_ZZ1,X_receive,Y_receive,Z_receive,ant_cov_XX0,ant_cov_XY0,ant_cov_XZ0,ant_cov_YX0,ant_cov_YY0,ant_cov_YZ0,ant_cov_ZX0,ant_cov_ZY0,ant_cov_ZZ0,X_transmit,Y_transmit,Z_transmit
712483771.569176,NDP1-2,3.569176,712483768.0,0.0002271,9.8e-07,-2.2e-07,9.8e-07,0.0002278,-3.1e-07,-2.2e-07,-3.1e-07,0.00022849,-2606685.75482955,-3705817.00172181,4474091.26651063,7.841e-05,1.1e-07,-4e-08,1.1e-07,7.849e-05,-6e-08,-4e-08,-6e-08,7.857e-05,-2606687.12007168,-3705817.05582391,4474089.56415817
```

By default, L-2 Data generated by GNATSS will report time in J2000 seconds
relative to 12:00:00 on Jan 1, 2000. GNATSS automatically corrects for leap
seconds.

### Sound Velocity Profile

The sound velocity profile is a simple two-column text file with the first
column being the depth (m, negative down) and the second column being the sound
velocity in m/s. This may be derived from a CTD or XBT during instrument
deployment. For example:

```
-1.9840000000000000 1502.5000000000000
-2.9760000000000000 1502.5400000000000
-3.9670000000000001 1502.5699999999999
-4.9589999999999996 1502.5100000000000
-5.9509999999999996 1502.4900000000000
-6.9429999999999996 1502.3599999999999
-7.9349999999999996 1502.3000000000000
```
