# GNATSS Example

## Setting up the initial run

The following example shows how to calculate an array position for a
GNSS-Acoustic survey using GNATSS. The data set we will demonstrate is the 2022
survey of the array NDP1 offshore Depoe Bay, OR, which was collected as part of
the Seafloor Geodesy Community experiment.

The first step is to prepare all of the input files required to calculate a
position, including the _pxp_tt_ file with acoustic two-way travel times, the
_gps_positions_ file with antenna positions, the Novatel files with #INSPVA and
#INSSTDEVA strings, and the sound velocity profile. Create a working directory
and place a configuration file inside of it. This configuration file may be
edited from the template available on the [GNATSS Config File](./config_yaml.md)
page or the [GNATSS GitHub](https://github.com/seafloor-geodesy/gnatss). You may
name this file whatever you like, but for the purposes of this example, we will
name the configuration file _NDP1_2022.yaml_. Edit _NDP1_2022.yaml_ to reflect
the a priori site information and locations of the input data files. Once you
have done this, run GNATSS in the working directory with the following command
on the command line interface:

```
gnatss run --distance-limit 150 NDP1_2022.yaml
```

After a successful run, GNATSS will generate an _output_ directory defined in
the configuration file, by default _./output_. The output directory will contain
the following files:

- `gps_solution.csv`: Input data file in the Community Standard GNSS-A format.
  This file is only generated when the user runs the _posfilter_ mode of GNATSS.
- `process_dataset.nc`: NetCDF file with array offsets.
- `residuals.csv`: File with the GNSS-Acoustic residuals, the difference between
  the measured and modelled two-way travel times for each ping. Residual values
  are converted to centimeters from seconds with the mean sound velocity.
- `deletions.csv`: File with a list of residuals with poor data quality, to be
  removed the next time gnatss is executed.
- `dist_center.csv`: File with the distances between the transducer and the
  center of the array for each ping.
- `outliers.csv`: File with residuals that fall outside a user-defined
  threshold. Concatenated to _deletions.csv_ the next time gnatss is executed.
- `residuals.png`: A plot of the acoustic residuals. There is one time series
  for each transponder in the array.
- `residuals_enu_components.png`: A plot of the acoustic residuals averaged
  together over space to estimate the apparent offset of the array center during
  each ping.

The above run executed both the posfilter and solver modes of GNATSS by default
since we did not specify an individual mode. Because of this, a
_gps_solution.csv_ file was generated in addition to the array position. If we
want to repeat the array positioning executed by the solver, we can skip the
posfilter mode by using this _gps_solution.csv_ file as input data.

A good way to assess the quality of the solution is to evaluate the
_residuals.png_ file. After running GNATSS the first time with the above
command, this plot shows the following:

![residuals_1](./residuals_1.png)

In this example, we directed GNATSS to automatically flag and remove any pings
from the inversion during which the transducer was more than 150 m from the
center of the array with the "--distance-limit" option. However, the residuals
still appear to have some large outliers relative to the mean. Notably, the
large vertical spikes are likely errors due to GNSS cycle slips and are
significantly biasing the solution.

## Refining the solution

One of the strengths of GNATSS is that it can identify and flag outlier
residuals according to a user-defined threshold. Since the first run showed that
the vast majority of residuals were <500 cm from the mean, we will run GNATSS
again while setting the outlier threshold to 500. In this run, we no longer need
to run the posfilter mode since the _gps_solution.csv_ file was generated
previously. To take advantage of this, we can use the "--from-cache" flag, which
will direct GNATSS to skip the posfilter mode and look for the
_gps_solution.csv_ file in the output directory. We don't even need to update
the configuration file!

```
gnatss run --from-cache --distance-limit 150 --residual-limit 500 NDP1_2022.yaml
```

After doing this, you will notice that the solution itself has not changed.
However, if we open look a the residuals again, we can see that the vertical
spkies due to the GNSS cycle slips have been flagged.

![residuals_2](./residuals_2.png)

All of the flagged residuals have been stored in the file _outliers.csv_. If we
run GNATSS with the "--remove-outliers" flag, this file will be concatenated to
the _deletions.csv_ file and the residuals removed from the inversion. If for
some reason you feel that too many residuals have been flagged, you can delete
_outliers.csv_ in order to unflag those residuals, or simply run GNATSS again
with a looser outlier threshold. Note that no data has been actually removed
from the input files, so if you feel that too much data has been concatenated to
the _deletions.csv_ file, you can delete it to start a fresh inversion.

In any case, now that we have some outliers flagged, we can now run GNATSS with
the "--remove-outliers" flag, which will direct it to automatically remove any
flagged residuals:

```
gnatss run --from-cache --distance-limit 150 --remove-outliers NDP1_2022.yaml
```

GNATSS has now removed the residuals that had been flagged during the previous
run.

![residuals_3](./residuals_3.png)

There are still some outlier residuals, so let's try tightening the outlier
threshold. The strategy used to flag and remove residuals is up to the user's
discretion. In this case, there is a fairly consistent band of residuals near 0
cm that I interpret as the primary geodetic signal within the time series. I
choose a strict outlier threshold of 30 cm in order to isolate that signal.

```
gnatss run --from-cache --distance-limit 150 --residual-limit 30 NDP1_2022.yaml
```

This flags the following residuals:

![residuals_4](./residuals_4.png)

Since I am satisfied that only outliers from the main time series are being
flagged, I lock in the deletions by running GNATSS again. In this case I'll call
both the "--remove-outliers" flag and the "--residual-limit" flag, although I
won't change the residual limit.

```
gnatss run --from-cache --distance-limit 150 --residual-limit 30 --remove-outliers NDP1_2022.yaml
```

This generates the following residual plot:

![residuals_5](./residuals_5.png)

Notice how new residuals have been flagged! Since the final position is
generated by averaging all of the residuals in the time series, outliers can
bias the result. Because of this, when we removed the previous outliers, the
mean position shifted enough that residuals just below the threshold were pushed
beyond it. However, although it is possible to direct GNATSS to remove outliers
and flag new residuals in a single run, we do not recommend doing so since it
increases the risk of the user accidentally flagging good data. In the case that
you flag residuals you do not want to discard, you may either manually delete
the _outliers.csv_ file or run gnatss again with a new "--residual-limit flag"
to update _outliers.csv_. As long as you do not execute GNATSS with the
"--remove-outliers" flag, any flagged residuals will not be removed from the
positioning.

I choose to run GNATSS a few more times in order to remove all of the residuals
right at the 30 cm threshold until no new residuals are flagged. The final
residual plot looks like this:

![residuals_6](./residuals_6.png)

There is a definitive cyclical signal within these residuals due to
oceanographic effects. However, the important thing to note is that all three
residual time series for the transponders are capturing similar signals from the
ocean and plotting on top of each other. Because of this, if we average the
residuals over space to estimate the offstes of the array center, we will see
that this cyclical signal mostly maps in to the vertical component and that the
horizontal offsets are pretty stable. This can be verified in the
_residuals_enu_components.png_ plot:

![residuals_enu_components_6](./residuals_enu_components_6.png)
