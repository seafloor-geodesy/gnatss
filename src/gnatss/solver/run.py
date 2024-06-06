from __future__ import annotations

from ..ops.data import ensure_monotonic_increasing
from ..ops.validate import check_sig3d
from .utilities import (
    _print_detected_outliers,
    _print_final_stats,
    extract_latest_residuals,
    filter_by_distance_limit,
    filter_deletions_and_qc,
    generate_process_xr_dataset,
    get_all_epochs,
    get_residual_outliers,
    prepare_and_solve,
)


def run_solver(config, data_dict, return_raw: bool = False):
    all_observations = data_dict.get("gps_solution")
    if all_observations is None:
        msg = "No GNSS-A Level-2 data found. Unable to perform solver."
        raise ValueError(msg)

    # Ensure the data is sorted properly
    all_observations = ensure_monotonic_increasing(all_observations)

    all_observations = filter_deletions_and_qc(all_observations, data_dict)
    all_observations = check_sig3d(all_observations, config.solver.gps_sigma_limit)
    all_observations, dist_center_df = filter_by_distance_limit(all_observations, config)
    all_epochs = get_all_epochs(all_observations)

    twtt_model = config.solver.twtt_model
    process_data, is_converged = prepare_and_solve(all_observations, config, twtt_model=twtt_model)

    if is_converged:
        _print_final_stats(config.solver.transponders, process_data)

    # Extract the latest run residuals
    resdf = extract_latest_residuals(config, all_epochs, process_data)

    # Get the outliers
    outliers_df = get_residual_outliers(config, resdf)
    outlier_threshold = config.solver.residual_outliers_threshold
    # Print the outliers stats
    _print_detected_outliers(
        outliers_df, outlier_threshold, all_epochs, config.solver.residual_limit
    )

    # Capture the process results as an xarray dataset
    process_ds = generate_process_xr_dataset(process_data, resdf, config)
    data_dict.update(
        {
            "residuals": resdf,
            "distance_from_center": dist_center_df,
            "process_dataset": process_ds,
            "outliers": outliers_df,
        }
    )
    if return_raw:
        # Add the raw processing data to the result dict
        data_dict.update(
            {
                "all_epochs": all_epochs,
                "process_data": process_data,
            }
        )
    return data_dict
