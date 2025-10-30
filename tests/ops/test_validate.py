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
        good_gps = pd.Series(
            {
                "ant_cov_XX1": 0.01,  # 1cm std dev
                "ant_cov_YY1": 0.01,  # 1cm std dev
                "ant_cov_ZZ1": 0.01,  # 1cm std dev
            }
        )

        result = calc_std_and_verify(
            good_gps,
            std_dev=True,
            sigma_limit=0.05,  # 5cm limit
            verify=True,
        )

        # Should return 3D std dev ≈ sqrt(0.01² + 0.01² + 0.01²) ≈ 0.0173m
        expected_3d_std = np.sqrt(0.01**2 + 0.01**2 + 0.01**2)
        assert abs(result - expected_3d_std) < 1e-6
        assert result < 0.05  # Should be under 5cm limit

    def test_calc_std_and_verify_bad_gps_fails(self):
        """Test that bad GPS data (> 5cm accuracy) fails validation."""
        # Create GPS data with 10cm accuracy in each direction (too inaccurate)
        bad_gps = pd.Series(
            {
                "ant_cov_XX1": 0.1,  # 10cm std dev
                "ant_cov_YY1": 0.1,  # 10cm std dev
                "ant_cov_ZZ1": 0.1,  # 10cm std dev
            }
        )

        # Should raise ValueError when GPS accuracy exceeds limit
        with pytest.raises(ValueError, match=r"3D Standard Deviation.*exceeds GPS Sigma Limit"):
            calc_std_and_verify(
                bad_gps,
                std_dev=True,
                sigma_limit=0.05,  # 5cm limit
                verify=True,
            )

    def test_calc_std_and_verify_good_gps_fails_with_lower_sigma_limit(self):
        """Test that good GPS data fails with a stricter sigma limit."""
        # Use the same good GPS data (1cm accuracy)
        good_gps = pd.Series(
            {
                "ant_cov_XX1": 0.01,  # 1cm std dev
                "ant_cov_YY1": 0.01,  # 1cm std dev
                "ant_cov_ZZ1": 0.01,  # 1cm std dev
            }
        )

        # Should fail with stricter limit (0.01m) since 3D std ≈ 0.0173m > 0.01m
        with pytest.raises(ValueError, match=r"3D Standard Deviation.*exceeds GPS Sigma Limit"):
            calc_std_and_verify(
                good_gps,
                std_dev=True,
                sigma_limit=0.01,  # 1cm limit (stricter)
                verify=True,
            )

    def test_calc_std_and_verify_bad_gps_passes_with_higher_sigma_limit(self):
        """Test that bad GPS data passes with a more lenient sigma limit."""
        # Use the same bad GPS data (10cm accuracy)
        bad_gps = pd.Series(
            {
                "ant_cov_XX1": 0.1,  # 10cm std dev
                "ant_cov_YY1": 0.1,  # 10cm std dev
                "ant_cov_ZZ1": 0.1,  # 10cm std dev
            }
        )

        result = calc_std_and_verify(
            bad_gps,
            std_dev=True,
            sigma_limit=0.2,  # 20cm limit (more lenient)
            verify=True,
        )

        # Should return 3D std dev ≈ sqrt(0.1² + 0.1² + 0.1²) ≈ 0.173m
        expected_3d_std = np.sqrt(0.1**2 + 0.1**2 + 0.1**2)
        assert abs(result - expected_3d_std) < 1e-6
        assert result < 0.2  # Should be under 20cm limit

    def test_calc_std_and_verify_with_verify_false(self):
        """Test that verify=False returns 3D std without raising errors."""
        # Use bad GPS data that would normally fail
        bad_gps = pd.Series(
            {
                "ant_cov_XX1": 0.1,  # 10cm std dev
                "ant_cov_YY1": 0.1,  # 10cm std dev
                "ant_cov_ZZ1": 0.1,  # 10cm std dev
            }
        )

        result = calc_std_and_verify(
            bad_gps,
            std_dev=True,
            sigma_limit=0.05,  # 5cm limit (would normally fail)
            verify=False,  # Don't verify, just calculate
        )

        # Should return 3D std dev without raising error
        expected_3d_std = np.sqrt(0.1**2 + 0.1**2 + 0.1**2)
        assert abs(result - expected_3d_std) < 1e-6

    def test_calc_std_and_verify_with_variances(self):
        """Test that std_dev=False correctly handles variance inputs."""
        # Provide variance values instead of standard deviations
        gps_variances = pd.Series(
            {
                "ant_cov_XX1": 0.0001,  # 0.01² = 0.0001 (1cm std dev as variance)
                "ant_cov_YY1": 0.0001,  # 0.01² = 0.0001 (1cm std dev as variance)
                "ant_cov_ZZ1": 0.0001,  # 0.01² = 0.0001 (1cm std dev as variance)
            }
        )

        result = calc_std_and_verify(
            gps_variances,
            std_dev=False,  # Input values are variances, not std devs
            sigma_limit=0.05,  # 5cm limit
            verify=True,
        )

        # Should return 3D std dev ≈ sqrt(0.01² + 0.01² + 0.01²) ≈ 0.0173m
        # Same as the good GPS test since we're using equivalent variance values
        expected_3d_std = np.sqrt(0.01**2 + 0.01**2 + 0.01**2)
        assert abs(result - expected_3d_std) < 1e-6
        assert result < 0.05  # Should be under 5cm limit


if __name__ == "__main__":
    pytest.main([__file__])
