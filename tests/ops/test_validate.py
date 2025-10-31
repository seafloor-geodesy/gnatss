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
    """Comprehensive test suite for the calc_std_and_verify function."""

    # Basic functionality tests
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

    # Edge case tests

    def test_zero_values(self):
        """Test handling of zero covariance values."""
        zero_gps = pd.Series({
            "ant_cov_XX1": 0.0,
            "ant_cov_YY1": 0.0,
            "ant_cov_ZZ1": 0.0,
        })

        result = calc_std_and_verify(zero_gps, std_dev=True, sigma_limit=0.05, verify=True)
        assert result == 0.0

        # Zero values should always pass verification
        result_strict = calc_std_and_verify(zero_gps, std_dev=True, sigma_limit=0.001, verify=True)
        assert result_strict == 0.0

    def test_negative_values_variance_mode(self):
        """Test handling of negative values in variance mode."""
        negative_variance = pd.Series({
            "ant_cov_XX1": -0.01,  # Negative variance (mathematically invalid)
            "ant_cov_YY1": 0.01,
            "ant_cov_ZZ1": 0.01,
        })

        # Function should handle negative values (sum becomes 0.01 - 0.01 + 0.01 = 0.01)
        result = calc_std_and_verify(negative_variance, std_dev=False, sigma_limit=0.05, verify=False)
        expected = np.sqrt(0.01)  # sqrt(0.01) ≈ 0.1
        assert abs(result - expected) < 1e-10

        # Test case where negative values make sum negative (should produce NaN)
        all_negative_variance = pd.Series({
            "ant_cov_XX1": -0.01,
            "ant_cov_YY1": -0.01,
            "ant_cov_ZZ1": -0.01,
        })

        result_negative = calc_std_and_verify(all_negative_variance, std_dev=False, sigma_limit=0.05, verify=False)
        # sqrt of negative number should produce NaN
        assert np.isnan(result_negative)

    def test_nan_values(self):
        """Test handling of NaN values in input data."""
        nan_gps = pd.Series({
            "ant_cov_XX1": np.nan,
            "ant_cov_YY1": 0.01,
            "ant_cov_ZZ1": 0.01,
        })

        # pandas sum() skips NaN by default, so result should be sqrt(0.01² + 0.01²)
        result = calc_std_and_verify(nan_gps, std_dev=True, sigma_limit=0.05, verify=False)
        expected = np.sqrt(0.01**2 + 0.01**2)
        assert abs(result - expected) < 1e-10

        # Test all NaN case
        all_nan_gps = pd.Series({
            "ant_cov_XX1": np.nan,
            "ant_cov_YY1": np.nan,
            "ant_cov_ZZ1": np.nan,
        })

        result_all_nan = calc_std_and_verify(all_nan_gps, std_dev=True, sigma_limit=0.05, verify=False)
        # Sum of all NaN should be 0 (pandas behavior), so sqrt(0) = 0
        assert result_all_nan == 0.0

    def test_inf_values(self):
        """Test handling of infinite values in input data."""
        inf_gps = pd.Series({
            "ant_cov_XX1": np.inf,
            "ant_cov_YY1": 0.01,
            "ant_cov_ZZ1": 0.01,
        })

        result = calc_std_and_verify(inf_gps, std_dev=True, sigma_limit=0.05, verify=False)
        assert np.isinf(result)

        # Infinite values should fail verification
        with pytest.raises(ValueError, match=r"3D Standard Deviation.*exceeds GPS Sigma Limit"):
            calc_std_and_verify(inf_gps, std_dev=True, sigma_limit=0.05, verify=True)

    def test_exact_sigma_limit_matches(self):
        """Test boundary conditions where 3D std exactly equals sigma_limit."""
        # Calculate values that will give exactly 0.05 3D std
        target_3d_std = 0.05
        individual_std = target_3d_std / np.sqrt(3)  # Equal contribution from each axis

        exact_limit_gps = pd.Series({
            "ant_cov_XX1": individual_std,
            "ant_cov_YY1": individual_std,
            "ant_cov_ZZ1": individual_std,
        })

        # Should pass when exactly at limit (condition is > not >=)
        result = calc_std_and_verify(exact_limit_gps, std_dev=True, sigma_limit=0.05, verify=True)
        assert abs(result - 0.05) < 1e-10

        # Should fail when slightly over limit
        over_limit_gps = pd.Series({
            "ant_cov_XX1": individual_std * 1.001,
            "ant_cov_YY1": individual_std * 1.001,
            "ant_cov_ZZ1": individual_std * 1.001,
        })

        with pytest.raises(ValueError, match=r"3D Standard Deviation.*exceeds GPS Sigma Limit"):
            calc_std_and_verify(over_limit_gps, std_dev=True, sigma_limit=0.05, verify=True)

        # Should pass when slightly under limit
        under_limit_gps = pd.Series({
            "ant_cov_XX1": individual_std * 0.999,
            "ant_cov_YY1": individual_std * 0.999,
            "ant_cov_ZZ1": individual_std * 0.999,
        })

        result = calc_std_and_verify(under_limit_gps, std_dev=True, sigma_limit=0.05, verify=True)
        assert result < 0.05

    def test_extreme_limits(self):
        """Test with extremely small and large sigma limits."""
        normal_gps = pd.Series({
            "ant_cov_XX1": 0.01,
            "ant_cov_YY1": 0.01,
            "ant_cov_ZZ1": 0.01,
        })

        # Extremely small limit should fail
        with pytest.raises(ValueError):
            calc_std_and_verify(normal_gps, std_dev=True, sigma_limit=1e-10, verify=True)

        # Extremely large limit should pass
        result = calc_std_and_verify(normal_gps, std_dev=True, sigma_limit=1e10, verify=True)
        assert result < 1e10

    def test_different_column_counts(self):
        """Test behavior with different numbers of columns."""
        # Test with fewer columns
        incomplete_gps = pd.Series({
            "ant_cov_XX1": 0.01,
            "ant_cov_YY1": 0.01,
            # Missing ant_cov_ZZ1
        })

        # Function should work with any number of columns
        result = calc_std_and_verify(incomplete_gps, std_dev=True, sigma_limit=0.05, verify=True)
        expected = np.sqrt(0.01**2 + 0.01**2)
        assert abs(result - expected) < 1e-10

        # Test with more columns
        extended_gps = pd.Series({
            "ant_cov_XX1": 0.01,
            "ant_cov_YY1": 0.01,
            "ant_cov_ZZ1": 0.01,
            "ant_cov_WW1": 0.01,  # Extra column
        })

        result_extended = calc_std_and_verify(extended_gps, std_dev=True, sigma_limit=0.1, verify=True)
        expected_extended = np.sqrt(4 * 0.01**2)
        assert abs(result_extended - expected_extended) < 1e-10

    def test_invalid_data_types(self):
        """Test handling of invalid data types in Series."""
        # Test with numeric strings - should raise TypeError during ** operation
        string_gps = pd.Series({
            "ant_cov_XX1": "0.01",  # String instead of float
            "ant_cov_YY1": 0.01,
            "ant_cov_ZZ1": 0.01,
        })

        # Mixed string/numeric types should raise TypeError during ** operation
        with pytest.raises(TypeError):
            calc_std_and_verify(string_gps, std_dev=True, sigma_limit=0.05, verify=True)

        # Test with all numeric data (should work)
        numeric_gps = pd.Series({
            "ant_cov_XX1": 0.01,
            "ant_cov_YY1": 0.01,
            "ant_cov_ZZ1": 0.01,
        })

        result = calc_std_and_verify(numeric_gps, std_dev=True, sigma_limit=0.05, verify=True)
        expected = np.sqrt(0.01**2 + 0.01**2 + 0.01**2)
        assert abs(result - expected) < 1e-10

    def test_empty_series(self):
        """Test handling of empty Series input."""
        empty_gps = pd.Series(dtype=float)

        # Empty series sum should be 0, so sqrt(0) = 0
        result = calc_std_and_verify(empty_gps, std_dev=True, sigma_limit=0.05, verify=True)
        assert result == 0.0

    def test_parameter_matrix_combinations(self):
        """Test all combinations of std_dev × verify × sigma_limit parameters."""
        test_gps = pd.Series({
            "ant_cov_XX1": 0.01,
            "ant_cov_YY1": 0.01,
            "ant_cov_ZZ1": 0.01,
        })

        variance_gps = pd.Series({
            "ant_cov_XX1": 0.0001,  # 0.01²
            "ant_cov_YY1": 0.0001,
            "ant_cov_ZZ1": 0.0001,
        })

        # Test matrix: std_dev=[True, False] × verify=[True, False] × sigma_limit=[strict, lenient]
        combinations = [
            (True, True, 0.02),    # std_dev=True, verify=True, strict limit (should pass)
            (True, True, 0.01),    # std_dev=True, verify=True, very strict (should fail)
            (True, False, 0.01),   # std_dev=True, verify=False, strict (should return value)
            (False, True, 0.02),   # std_dev=False, verify=True, strict (should pass)
            (False, True, 0.01),   # std_dev=False, verify=True, very strict (should fail)
            (False, False, 0.01),  # std_dev=False, verify=False, strict (should return value)
        ]

        for std_dev, verify, sigma_limit in combinations:
            data = test_gps if std_dev else variance_gps

            if verify and sigma_limit < 0.015:  # Will fail verification
                with pytest.raises(ValueError):
                    calc_std_and_verify(data, std_dev=std_dev, sigma_limit=sigma_limit, verify=verify)
            else:  # Should succeed
                result = calc_std_and_verify(data, std_dev=std_dev, sigma_limit=sigma_limit, verify=verify)
                assert isinstance(result, float)
                assert result > 0

    def test_numerical_precision_small_values(self):
        """Test numerical precision with very small values."""
        tiny_gps = pd.Series({
            "ant_cov_XX1": 1e-10,
            "ant_cov_YY1": 1e-10,
            "ant_cov_ZZ1": 1e-10,
        })

        result = calc_std_and_verify(tiny_gps, std_dev=True, sigma_limit=1e-8, verify=True)
        expected = np.sqrt(3 * (1e-10)**2)
        assert abs(result - expected) < 1e-15

    def test_numerical_precision_large_values(self):
        """Test numerical precision with very large values."""
        large_gps = pd.Series({
            "ant_cov_XX1": 1e6,
            "ant_cov_YY1": 1e6,
            "ant_cov_ZZ1": 1e6,
        })

        result = calc_std_and_verify(large_gps, std_dev=True, sigma_limit=2e6, verify=True)
        expected = np.sqrt(3 * (1e6)**2)
        assert abs(result - expected) < 1e3  # Allow for some floating point error

    def test_floating_point_edge_cases(self):
        """Test floating-point arithmetic edge cases."""
        # Test values that might cause floating-point precision issues
        precision_gps = pd.Series({
            "ant_cov_XX1": 0.1 + 0.2 - 0.3,  # Should be 0, but floating point...
            "ant_cov_YY1": 1.0 / 3.0 * 3.0 - 1.0,  # Should be 0
            "ant_cov_ZZ1": 0.01,
        })

        result = calc_std_and_verify(precision_gps, std_dev=True, sigma_limit=0.05, verify=False)
        # Should handle floating-point imprecision gracefully
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_single_axis_dominance(self):
        """Test cases where one axis dominates the 3D calculation."""
        dominant_axis_gps = pd.Series({
            "ant_cov_XX1": 0.05,  # Large value
            "ant_cov_YY1": 1e-10,  # Tiny value
            "ant_cov_ZZ1": 1e-10,  # Tiny value
        })

        result = calc_std_and_verify(dominant_axis_gps, std_dev=True, sigma_limit=0.1, verify=True)
        # Result should be dominated by the X component
        assert abs(result - 0.05) < 1e-6

    def test_variance_vs_std_dev_consistency(self):
        """Test that variance and std_dev modes give consistent results."""
        std_values = pd.Series({
            "ant_cov_XX1": 0.02,
            "ant_cov_YY1": 0.03,
            "ant_cov_ZZ1": 0.04,
        })

        var_values = pd.Series({
            "ant_cov_XX1": 0.02**2,
            "ant_cov_YY1": 0.03**2,
            "ant_cov_ZZ1": 0.04**2,
        })

        result_std = calc_std_and_verify(std_values, std_dev=True, sigma_limit=0.1, verify=False)
        result_var = calc_std_and_verify(var_values, std_dev=False, sigma_limit=0.1, verify=False)

        # Results should be identical within floating-point precision
        assert abs(result_std - result_var) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
