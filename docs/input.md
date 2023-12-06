# Running GNATSS

## Overview

Step 1: Generate the required input files (acoustic two-way travel times, surface platform positions, sound velocity profile)

Step 2: Create working directory and config.yaml file

Step 3: From working directory, execute

```bash
gnatss run --extract-dist-center --extract-process-dataset --qc --distance-limit 150 --residual-limit 10000
```

Step 4: Repeat Step 3, reducing the residual limit as desired in each successive iteration in order to remove erroneous residuals

Step 5: Array positions, offsets, and statistics are stored in the process_dataset.nc file.

## Required Input Data

GNATSS requires the following data in order to generate an array position:

- Acoustic two-way travel times
- Surface platform positions
- A sound velocity profile

### Acoustic Two-Way Travel Times

Acoustic two-way travel times are defined as the duration between when the transducer on the surface platform sends an acoustic interrogation and when it receives a reply from a seafloor transponder. Multiple transponders may respond to the same interrogation if they each receive the ping. Two-way travel times should be stored in a file named “pxp_tt”, with a text column format in which each row corresponds to a ping, containing UTC timestamps and two-way travel times in microseconds. Example:

```console
26-JUN-23 00:00:07.00   2241985   2337422   2469996
26-JUN-23 00:00:22.00   2240506   2346387   2463020
26-JUN-23 00:00:37.00   2240627   2353454   2454958
26-JUN-23 00:00:52.00   2243723   2359242   2445547
```

Each of the two-way travel time columns corresponds to the replies from a specific transponder as defined in the config.yaml file. In order to prevent interference between the replies, each transponder may be assigned an internal delay time that will elapse before sending a reply to an interrogation ping. These delay times must be defined in the config.yaml file and will be removed during positioning.

There may also be a delay time associated with the surface platform corresponding to the time between when the computer sends an interrogation command and when the transducer emits an acoustic pulse. For instance, the Sonardyne GNSS-A payload has a 0.13 second delay time that is recorded in its raw two-way travel times. This delay must be removed prior to GNATSS positioning because there is not an acoustic pulse physically traveling through the water column during this time.

### Surface Platform Positions

The surface platform positions refer to the positions of the transducer on the surface platform and must be known at every time epoch when the transducer emits an interrogation pulse and receives a reply. These positions must be stored in a file named “POS_FREED_TRANS_TWTT”, with a text column file with the following format:

```
Time (j2000 seconds), Geodetic XYZ coordinates (m, 3 entries), XYZ covariance matrix (m^2, 9 entries)
```

These positions should correspond to the times and two-way travel times recorded in the “pxp_tt” file. For instance, the positions corresponding to the ping interrogation and replies of the “26-JUN-23 00:00:07.00” in the pxp_tt excerpt above are recorded as

```
   741009607.0000000  -2575338.456      -3682605.618       4511008.716      0.7592545444E-03  0.3218813907E-06  -.3030846830E-05  0.3218813907E-06  0.7594897193E-03  -.4333961609E-05  -.3030846830E-05  -.4333961609E-05  0.7598263681E-03
   741009609.2419850  -2575338.614      -3682604.946       4511008.820      0.2796384902E-03  0.1142867658E-06  -.8293527721E-06  0.1142867658E-06  0.2797219910E-03  -.1185933310E-05  -.8293527721E-06  -.1185933310E-05  0.2798324391E-03
   741009609.3374220  -2575338.610      -3682604.861       4511008.866      0.3749910877E-03  0.1630394849E-06  -.1153364964E-05  0.1630394849E-06  0.3751102085E-03  -.1649254651E-05  -.1153364964E-05  -.1649254651E-05  0.3752666760E-03
   741009609.4699960  -2575338.593      -3682604.744       4511008.958      0.5023142547E-03  0.2296528483E-06  -.1602120826E-05  0.2296528483E-06  0.5024820449E-03  -.2290953239E-05  -.1602120826E-05  -.2290953239E-05  0.5027016135E-03
```

Notice how the time difference between the 2nd-4th entries and the 1st entry correspond to the three two-way travel times logged in the pxp_tt file.

There is some extra processing required in order to generate the transducer positions since GNSS processing software will only solve for positions of the GNSS antenna at regular time epochs. You will need to interpolate the antenna positions to the interrogation and reply epochs and rotate the positions from the antenna to the transducer. The interpolation may be accomplished with multiple strategies such as a spline fitting algorithm or Kalman filter. The rotation requires the instantaneous roll, pitch, and heading values of the surface platform (also interpolated) and the body frame offsets of the surface platform. The roll, pitch, and heading values should be recorded by an inertial navigation system deployed on the surface platform. The body frame offsets are unique to the surface platform and must be carefully surveyed prior to the experiment.

### Sound Velocity Profile

The sound velocity profile is a simple two-column text file with the first column being the depth (m, negative down) and the second column being the sound velocity in m/s. This may be derived from a CTD or XBT during instrument deployment. For example:

```
-1.9840000000000000 1502.5000000000000
-2.9760000000000000 1502.5400000000000
-3.9670000000000001 1502.5699999999999
-4.9589999999999996 1502.5100000000000
-5.9509999999999996 1502.4900000000000
-6.9429999999999996 1502.3599999999999
-7.9349999999999996 1502.3000000000000
```
