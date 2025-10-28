"""
Tests for quality control validation functions in gnatss.ops.validate

This module tests the non-numba quality control functions that ensure
scientific data integrity in GNSS-A processing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnatss.ops.validate import calc_std_and_verify


class TestCalcStdAndVerify:
    """Test the calc_std_and_verify function for GPS accuracy validation."""

    def test_calc_std_and_verify_good_gps_passes(self):
        """Test that good GPS data (< 5cm accuracy) passes validation."""
        # Create GPS data with 1cm accuracy in each direction
        good_gps = pd.Series({
            'ant_cov_XX1': 0.01,  # 1cm std dev
            'ant_cov_YY1': 0.01,  # 1cm std dev
            'ant_cov_ZZ1': 0.01   # 1cm std dev
        })

        result = calc_std_and_verify(
            good_gps,
            std_dev=True,
            sigma_limit=0.05,  # 5cm limit
            verify=True
        )

        # Should return 3D std dev ≈ sqrt(0.01² + 0.01² + 0.01²) ≈ 0.0173m
        expected_3d_std = np.sqrt(0.01**2 + 0.01**2 + 0.01**2)
        assert abs(result - expected_3d_std) < 1e-6
        assert result < 0.05  # Should be under 5cm limit

    def test_calc_std_and_verify_bad_gps_fails(self):
        """Test that bad GPS data (> 5cm accuracy) fails validation."""
        # Create GPS data with 10cm accuracy in each direction (too inaccurate)
        bad_gps = pd.Series({
            'ant_cov_XX1': 0.1,   # 10cm std dev
            'ant_cov_YY1': 0.1,   # 10cm std dev
            'ant_cov_ZZ1': 0.1    # 10cm std dev
        })

        # Should raise ValueError when GPS accuracy exceeds limit
        with pytest.raises(ValueError, match="3D Standard Deviation.*exceeds GPS Sigma Limit"):
            calc_std_and_verify(
                bad_gps,
                std_dev=True,
                sigma_limit=0.05,  # 5cm limit
                verify=True
            )


if __name__ == "__main__":
    pytest.main([__file__])
