"""
Unit Tests for Sensitivity Kernel Loader and Catalog

Tests for loading, parsing, and managing sensitivity kernel data.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from pygeoinf.interval import (
    SensitivityKernelData,
    SensitivityKernelCatalog,
    parse_mode_id,
    format_mode_id,
    load_kernel_file,
    parse_header
)


class TestModeIdParsing:
    """Test mode identifier parsing and formatting."""

    def test_parse_mode_id_valid(self):
        """Test parsing valid mode IDs."""
        assert parse_mode_id("00s03") == (0, 3)
        assert parse_mode_id("12s15") == (12, 15)
        assert parse_mode_id("19s30") == (19, 30)

    def test_parse_mode_id_invalid(self):
        """Test parsing invalid mode IDs."""
        with pytest.raises(ValueError):
            parse_mode_id("0s3")  # Wrong format
        with pytest.raises(ValueError):
            parse_mode_id("00S03")  # Wrong case
        with pytest.raises(ValueError):
            parse_mode_id("invalid")

    def test_format_mode_id_valid(self):
        """Test formatting valid (n, l) pairs."""
        assert format_mode_id(0, 3) == "00s03"
        assert format_mode_id(12, 15) == "12s15"
        assert format_mode_id(99, 99) == "99s99"

    def test_format_mode_id_invalid(self):
        """Test formatting invalid (n, l) pairs."""
        with pytest.raises(ValueError):
            format_mode_id(-1, 3)  # Negative n
        with pytest.raises(ValueError):
            format_mode_id(0, 100)  # l too large
        with pytest.raises(ValueError):
            format_mode_id(100, 0)  # n too large

    def test_round_trip(self):
        """Test that parse and format are inverse operations."""
        mode_id = "12s15"
        n, l = parse_mode_id(mode_id)
        assert format_mode_id(n, l) == mode_id


class TestKernelFileLoading:
    """Test loading kernel data files."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    def test_load_simple_file(self, temp_dir):
        """Test loading a simple 2-column file."""
        # Create test file
        test_file = temp_dir / "test_kernel.dat"
        with open(test_file, 'w') as f:
            f.write("0.0 0.1\n")
            f.write("100.0 0.5\n")
            f.write("500.0 0.8\n")
            f.write("1000.0 0.3\n")

        depths, values = load_kernel_file(test_file)

        assert len(depths) == 4
        assert len(values) == 4
        np.testing.assert_array_equal(depths, [0.0, 100.0, 500.0, 1000.0])
        np.testing.assert_array_equal(values, [0.1, 0.5, 0.8, 0.3])

    def test_load_file_with_comments(self, temp_dir):
        """Test loading file with comment lines."""
        test_file = temp_dir / "test_kernel.dat"
        with open(test_file, 'w') as f:
            f.write("# Comment line\n")
            f.write("0.0 0.1\n")
            f.write("# Another comment\n")
            f.write("100.0 0.5\n")

        depths, values = load_kernel_file(test_file)

        assert len(depths) == 2
        assert len(values) == 2

    def test_load_file_scientific_notation(self, temp_dir):
        """Test loading file with scientific notation."""
        test_file = temp_dir / "test_kernel.dat"
        with open(test_file, 'w') as f:
            f.write("0.0 1.23E-03\n")
            f.write("100.0 4.56E+02\n")

        depths, values = load_kernel_file(test_file)

        np.testing.assert_allclose(values, [1.23e-3, 4.56e2])

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_kernel_file(Path("/nonexistent/file.dat"))

    def test_load_empty_file(self, temp_dir):
        """Test loading empty file raises error."""
        test_file = temp_dir / "empty.dat"
        test_file.touch()

        with pytest.raises(ValueError, match="No data found"):
            load_kernel_file(test_file)

    def test_load_malformed_file(self, temp_dir):
        """Test loading malformed file raises error."""
        test_file = temp_dir / "malformed.dat"
        with open(test_file, 'w') as f:
            f.write("0.0 0.1 0.2\n")  # 3 columns instead of 2

        with pytest.raises(ValueError, match="expected 2 columns"):
            load_kernel_file(test_file)


class TestHeaderParsing:
    """Test parsing combined file headers."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    def test_parse_valid_header(self, temp_dir):
        """Test parsing valid header."""
        test_file = temp_dir / "test_header.dat"
        with open(test_file, 'w') as f:
            f.write("#2134.18042   6.70531   5.35906\n")
            f.write("#   0.46856 417.48904\n")
            f.write("0.0 0.1 0.2 0.3\n")

        header = parse_header(test_file)

        assert 'period' in header
        assert 'vp_ref' in header
        assert 'vs_ref' in header
        assert 'group_velocity' in header

        np.testing.assert_allclose(header['period'], 2134.18042)
        np.testing.assert_allclose(header['vp_ref'], 6.70531)
        np.testing.assert_allclose(header['vs_ref'], 5.35906)
        np.testing.assert_allclose(header['group_velocity'], 417.48904)

    def test_parse_header_no_hash(self, temp_dir):
        """Test parsing header without # fails."""
        test_file = temp_dir / "test_header.dat"
        with open(test_file, 'w') as f:
            f.write("2134.18042   6.70531   5.35906\n")
            f.write("#   0.46856 417.48904\n")

        with pytest.raises(ValueError, match="Expected header line starting with #"):
            parse_header(test_file)

    def test_parse_header_too_short(self, temp_dir):
        """Test parsing file with incomplete header."""
        test_file = temp_dir / "test_header.dat"
        with open(test_file, 'w') as f:
            f.write("#2134.18042   6.70531\n")  # Missing vs_ref

        with pytest.raises(ValueError):
            parse_header(test_file)


@pytest.mark.skipif(
    not Path("pygeoinf/interval/demos/kernels_modeplotaat_Adrian").exists(),
    reason="Real kernel data not available"
)
class TestSensitivityKernelData:
    """Test SensitivityKernelData class with real data."""

    @pytest.fixture
    def data_dir(self):
        """Path to real kernel data."""
        return Path("pygeoinf/interval/demos/kernels_modeplotaat_Adrian")

    def test_load_kernel_data(self, data_dir):
        """Test loading kernel data for a specific mode."""
        kernel = SensitivityKernelData("00s03", data_dir)

        assert kernel.mode_id == "00s03"
        assert kernel.n == 0
        assert kernel.l == 3
        assert kernel.period is not None
        assert len(kernel.vp_depths) > 0
        assert len(kernel.vs_depths) > 0
        assert len(kernel.rho_depths) > 0

    def test_kernel_data_shapes_consistent(self, data_dir):
        """Test that kernel data arrays have consistent shapes."""
        kernel = SensitivityKernelData("00s03", data_dir)

        # Volumetric kernels should have same length
        assert len(kernel.vp_depths) == len(kernel.vp_values)
        assert len(kernel.vs_depths) == len(kernel.vs_values)
        assert len(kernel.rho_depths) == len(kernel.rho_values)

        # Topography kernel
        assert len(kernel.topo_depths) == len(kernel.topo_values)

    def test_kernel_data_depths_valid(self, data_dir):
        """Test that depth values are valid."""
        kernel = SensitivityKernelData("00s03", data_dir)

        # Depths should be in [0, 6371] km
        assert np.all(kernel.vp_depths >= 0)
        assert np.all(kernel.vp_depths <= 6371)
        assert np.all(kernel.vs_depths >= 0)
        assert np.all(kernel.vs_depths <= 6371)

    def test_get_kernel_summary(self, data_dir):
        """Test getting kernel summary statistics."""
        kernel = SensitivityKernelData("00s03", data_dir)
        summary = kernel.get_kernel_summary()

        assert 'mode_id' in summary
        assert 'vp_kernel' in summary
        assert 'vs_kernel' in summary
        assert 'rho_kernel' in summary
        assert summary['vp_kernel']['n_points'] > 0


@pytest.mark.skipif(
    not Path("pygeoinf/interval/demos/kernels_modeplotaat_Adrian").exists(),
    reason="Real kernel data not available"
)
class TestSensitivityKernelCatalog:
    """Test SensitivityKernelCatalog class with real data."""

    @pytest.fixture
    def data_dir(self):
        """Path to real kernel data."""
        return Path("pygeoinf/interval/demos/kernels_modeplotaat_Adrian")

    @pytest.fixture
    def catalog(self, data_dir):
        """Create catalog instance."""
        return SensitivityKernelCatalog(data_dir)

    def test_catalog_initialization(self, catalog):
        """Test catalog initialization."""
        assert len(catalog.available_modes) > 0
        assert len(catalog) == len(catalog.available_modes)

    def test_get_mode(self, catalog):
        """Test retrieving a specific mode."""
        mode = catalog.get_mode("00s03")
        assert mode.mode_id == "00s03"

    def test_get_mode_by_nl(self, catalog):
        """Test retrieving mode by (n, l) notation."""
        mode = catalog.get_mode_by_nl(0, 3)
        assert mode.mode_id == "00s03"

    def test_get_nonexistent_mode(self, catalog):
        """Test retrieving non-existent mode raises error."""
        with pytest.raises(ValueError):
            catalog.get_mode("99s99")

    def test_list_modes_no_filter(self, catalog):
        """Test listing all modes."""
        modes = catalog.list_modes()
        assert len(modes) == len(catalog.available_modes)

    def test_list_modes_with_filter(self, catalog):
        """Test listing modes with filters."""
        # Get fundamental modes (n=0)
        fundamental = catalog.list_modes(n_min=0, n_max=0)
        assert len(fundamental) > 0

        for mode_id in fundamental:
            n, l = parse_mode_id(mode_id)
            assert n == 0

    def test_caching(self, catalog):
        """Test that modes are cached."""
        # First access
        mode1 = catalog.get_mode("00s03")
        assert len(catalog._cache) == 1

        # Second access should use cache
        mode2 = catalog.get_mode("00s03")
        assert mode1 is mode2  # Same object

    def test_clear_cache(self, catalog):
        """Test cache clearing."""
        catalog.get_mode("00s03")
        assert len(catalog._cache) > 0

        catalog.clear_cache()
        assert len(catalog._cache) == 0

    def test_get_catalog_summary(self, catalog):
        """Test getting catalog summary."""
        summary = catalog.get_catalog_summary()

        assert 'n_modes' in summary
        assert 'n_range' in summary
        assert 'l_range' in summary
        assert summary['n_modes'] == len(catalog.available_modes)

    def test_find_modes_by_period(self, catalog):
        """Test finding modes by period range."""
        modes = catalog.find_modes_by_period(1000, 2000)

        # Check that returned modes are in period range
        for mode_id in modes:
            mode = catalog.get_mode(mode_id)
            assert mode.period >= 1000
            assert mode.period <= 2000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
