# GNATSS Example

## Setting up the initial run

The following example shows how to calculate an array position for a GNSS-Acoustic survey using GNATSS. The data set we will demonstrate is the 2022 survey of the array NDP1 offshore Depoe Bay, OR, which was collected as part of the Seafloor Geodesy Community experiment.

The first step is to prepare all of the input files required to caluclate a position, including the *pxp_tt* file with acoustic two-way travel times, the *POS_FREED_TRANS_TWTT* file with transducer positions, and the sound velocity profile. Create a working directory and place a configuration file named *config.yaml* inside of it. Edit *config.yaml* to reflect the a priori site information and locations of the input data files. Once you have done this, run GNATSS in the working directory with the following command on the command line interface:

```
gnatss run --extract-dist-center --extract-process-dataset --qc --distance-limit 150
```

After a successful run, GNATSS will generate an *output* folder with the following files:

- *process_dataset.nc*: NetCDF file with array offsets.
- *residuals.csv*: File with the GNSS-Acoustic residuals, the difference between the measured and modelled two-way travel times for each ping. Residual values are converted to centimeters from seconds with the mean sound velocity.
- *deletions.csv*: File with a list of residuals with poor data quality, to be removed the next time gnatss is executed.
- *dist_center.csv*: File with the distances between the transducer and the center of the array for each ping.
- *outliers.csv*: File with residuals that fall outside a user-defined threshold. Concatenated to *deletions.csv* the next time gnatss is executed.
- *residuals.png*: A plot of the acoustic residuals. There is one time series for each transponder in the array.
- *residuals_enu_components.png*: A plot of the acoustic residuals averaged together over space to estimate the apparent offset of the array center during each ping.

A good way to assess the quality of the solution is to evaluate the *residuals.png* file. After running gnatss the first time with the above command, this plot shows the following: 
