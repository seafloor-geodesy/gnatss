"""The main command line interface for gnatss"""
from pathlib import Path
from typing import Optional

import typer

from . import constants, package_name
from .configs.io import CSVOutput
from .configs.solver import Solver
from .loaders import (
    load_configuration,
    load_gps_solutions,
    load_travel_times,
    read_novatel_L1_data_files,
)
from .main import gather_files_all_procs, main
from .ops.data import clean_tt, preprocess_tt, standardize_data
from .ops.posfilter import kalman_filtering, rotation, spline_interpolate

# Global variables
OVERRIDE_MESSAGE = "Note that this will override the value set as configuration."

app = typer.Typer(name=package_name)


@app.callback()
def callback():
    """
    GNSS-A Processing in Python
    """


@app.command()
def run_e2e(
    config_yaml: Optional[str] = typer.Option(
        None,
        help="Custom path to configuration yaml file. **Currently only support local files!**",
    ),
    extract_dist_center: Optional[bool] = typer.Option(
        True, help="Flag to extract distance from center from run."
    ),
    extract_process_dataset: Optional[bool] = typer.Option(
        True, help="Flag to extract process results."
    ),
    outlier_threshold: Optional[float] = typer.Option(
        constants.DATA_OUTLIER_THRESHOLD,
        help=(
            "Threshold for allowable percentage of outliers "
            "before raising a runtime error."
        ),
    ),
    distance_limit: Optional[float] = typer.Option(
        None,
        help=(
            f"{Solver.model_fields.get('distance_limit').description}"
            f". {OVERRIDE_MESSAGE}"
        ),
    ),
    residual_limit: Optional[float] = typer.Option(
        None,
        help=(
            f"{Solver.model_fields.get('residual_limit').description}"
            f". {OVERRIDE_MESSAGE}"
        ),
    ),
    qc: Optional[bool] = typer.Option(
        True, help="Flag to plot residuals from run and store in output folder."
    ),
) -> None:
    """Runs the full end to end routine for GNSS-A

    Note: Currently only supports 3 transponders and still under heavy development
    """
    # TODO: Clean up e2e function
    typer.echo("Loading configuration ...")
    config = load_configuration(config_yaml)

    # Override the distance and residual limits if provided
    # this short-circuits pydantic model
    if distance_limit is not None:
        config.solver.distance_limit = distance_limit

    if residual_limit is not None:
        config.solver.residual_limit = residual_limit

    all_files_dict = gather_files_all_procs(config)
    novatel_data = [
        {
            "data_files": all_files_dict["novatel"],
            "data_format": "INSPVAA",
        },
        {
            "data_files": all_files_dict["novatel_std"],
            "data_format": "INSSTDEVA",
        },
    ]
    typer.echo("Loading GNSS Data ...")
    gps_df = load_gps_solutions(all_files_dict["gps_positions"], is_gnss_data=True)
    transponder_ids = [tp.pxp_id for tp in config.transponders]
    typer.echo("Loading and cleaning Travel Times Data ...")
    pxp_df = load_travel_times(
        all_files_dict["travel_times"], transponder_ids=transponder_ids
    )
    pxp_df = clean_tt(
        pxp_df,
        transponder_ids=transponder_ids,
        travel_times_correction=config.travel_times_correction,
        transducer_delay_time=config.transducer_delay_time,
    )
    twtt_df = preprocess_tt(pxp_df)
    typer.echo("Loading NOVATEL L1 Data ...")
    novatel_dfs = [read_novatel_L1_data_files(**data_dct) for data_dct in novatel_data]
    inspvaa_df, insstdeva_df = novatel_dfs

    typer.echo("Performing Kalman filtering ...")
    # These are antenna positions and covariances
    pos_twtt = kalman_filtering(inspvaa_df, insstdeva_df, gps_df, twtt_df)
    typer.echo("Performing Spline Interpolation ...")
    cov_rph_twtt = spline_interpolate(inspvaa_df, insstdeva_df, twtt_df)
    typer.echo("Performing Rotation ...")
    # Values are transducer positions and covariances
    pos_freed_trans_twtt = rotation(
        pos_twtt, cov_rph_twtt, config.posfilter.atd_offsets, config.array_center
    )

    typer.echo(
        f"Standardizing data to specification version {constants.DATA_SPEC.version} ..."
    )
    all_observations = standardize_data(pos_freed_trans_twtt)

    # Run the main function
    # TODO: Clean up so that we aren't throwing data away
    typer.echo("Performing Solve for transponder locations ...")
    _, _, resdf, dist_center_df, process_ds, outliers_df = main(
        config,
        all_files_dict,
        all_observations=all_observations,
        extract_process_dataset=extract_process_dataset,
        outlier_threshold=outlier_threshold,
    )

    # TODO: Switch to fsspec so we can save anywhere
    output_path = Path(config.output.path)
    if extract_dist_center:
        dist_center_csv = output_path / CSVOutput.dist_center.value
        typer.echo(
            f"Saving the distance from center file to {str(dist_center_csv.absolute())}"
        )
        dist_center_df.to_csv(dist_center_csv, index=False)

    # Write out to residuals.csv file
    res_csv = output_path / CSVOutput.residuals.value
    typer.echo(f"Saving the latest residuals to {str(res_csv.absolute())}")
    resdf.to_csv(res_csv, index=False)

    # Write out to outliers.csv file
    if len(outliers_df) > 0:
        outliers_csv = output_path / CSVOutput.outliers.value
        typer.echo(
            f"Saving the latest residual outliers to {str(outliers_csv.absolute())}"
        )
        outliers_df.to_csv(outliers_csv, index=False)

    if extract_process_dataset:
        # Write out to process_dataset.nc file
        process_dataset_nc = output_path / "process_dataset.nc"
        typer.echo(
            "Saving the process results "
            f"dataset to {str(process_dataset_nc.absolute())}"
        )
        process_ds.to_netcdf(process_dataset_nc)

    if qc:
        from .ops.qc import plot_enu_comps, plot_residuals

        res_png = output_path / "residuals.png"
        enu_comp_png = output_path / "residuals_enu_components.png"

        # Plot the figures
        res_figure = plot_residuals(resdf, outliers_df)
        enu_figure = plot_enu_comps(resdf, config)

        # Save the figures
        res_figure.savefig(res_png)
        enu_figure.savefig(enu_comp_png)


@app.command()
def run(
    config_yaml: Optional[str] = typer.Option(
        None,
        help="Custom path to configuration yaml file. **Currently only support local files!**",
    ),
    extract_dist_center: Optional[bool] = typer.Option(
        True, help="Flag to extract distance from center from run."
    ),
    extract_process_dataset: Optional[bool] = typer.Option(
        True, help="Flag to extract process results."
    ),
    outlier_threshold: Optional[float] = typer.Option(
        constants.DATA_OUTLIER_THRESHOLD,
        help=(
            "Threshold for allowable percentage of outliers "
            "before raising a runtime error."
        ),
    ),
    distance_limit: Optional[float] = typer.Option(
        None,
        help=(
            f"{Solver.model_fields.get('distance_limit').description}"
            f". {OVERRIDE_MESSAGE}"
        ),
    ),
    residual_limit: Optional[float] = typer.Option(
        None,
        help=(
            f"{Solver.model_fields.get('residual_limit').description}"
            f". {OVERRIDE_MESSAGE}"
        ),
    ),
    qc: Optional[bool] = typer.Option(
        True, help="Flag to plot residuals from run and store in output folder."
    ),
) -> None:
    """Runs the full pre-processing routine for GNSS-A

    Note: Currently only supports 3 transponders
    """
    typer.echo("Loading configuration ...")
    config = load_configuration(config_yaml)

    # Override the distance and residual limits if provided
    # this short-circuits pydantic model
    if distance_limit is not None:
        config.solver.distance_limit = distance_limit

    if residual_limit is not None:
        config.solver.residual_limit = residual_limit

    typer.echo("Configuration loaded.")
    all_files_dict = gather_files_all_procs(config)

    # Run the main function
    # TODO: Clean up so that we aren't throwing data away
    _, _, resdf, dist_center_df, process_ds, outliers_df = main(
        config,
        all_files_dict,
        extract_process_dataset=extract_process_dataset,
        outlier_threshold=outlier_threshold,
    )

    # TODO: Switch to fsspec so we can save anywhere
    output_path = Path(config.output.path)
    if extract_dist_center:
        dist_center_csv = output_path / CSVOutput.dist_center.value
        typer.echo(
            f"Saving the distance from center file to {str(dist_center_csv.absolute())}"
        )
        dist_center_df.to_csv(dist_center_csv, index=False)

    # Write out to residuals.csv file
    res_csv = output_path / CSVOutput.residuals.value
    typer.echo(f"Saving the latest residuals to {str(res_csv.absolute())}")
    resdf.to_csv(res_csv, index=False)

    # Write out to outliers.csv file
    if len(outliers_df) > 0:
        outliers_csv = output_path / CSVOutput.outliers.value
        typer.echo(
            f"Saving the latest residual outliers to {str(outliers_csv.absolute())}"
        )
        outliers_df.to_csv(outliers_csv, index=False)

    if extract_process_dataset:
        # Write out to process_dataset.nc file
        process_dataset_nc = output_path / "process_dataset.nc"
        typer.echo(
            "Saving the process results "
            f"dataset to {str(process_dataset_nc.absolute())}"
        )
        process_ds.to_netcdf(process_dataset_nc)

    if qc:
        from .ops.qc import plot_enu_comps, plot_residuals

        res_png = output_path / "residuals.png"
        enu_comp_png = output_path / "residuals_enu_components.png"

        # Plot the figures
        res_figure = plot_residuals(resdf, outliers_df)
        enu_figure = plot_enu_comps(resdf, config)

        # Save the figures
        res_figure.savefig(res_png)
        enu_figure.savefig(enu_comp_png)
