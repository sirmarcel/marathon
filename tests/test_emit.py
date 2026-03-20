import tempfile
from pathlib import Path

import pytest

from marathon.emit.log import Txt
from marathon.emit.properties import (
    DEFAULT_PROPERTIES,
    get_base_unit,
    get_full_unit,
    get_scale,
    get_symbol,
)
from marathon.evaluate.properties import DEFAULT_NORMALIZATION

# -- Properties tests --


def test_get_scale():
    assert get_scale("energy") == 1000
    assert get_scale("forces") == 1000
    assert get_scale("stress") == 1000
    # Unknown property defaults to 1
    assert get_scale("unknown") == 1


def test_get_base_unit():
    assert get_base_unit("energy") == "meV"
    assert get_base_unit("forces") == "meV/Å"
    assert get_base_unit("stress") == "meV"
    assert get_base_unit("unknown") == ""


def test_get_full_unit():
    # With default normalization
    assert get_full_unit("energy") == "meV/atom"
    assert get_full_unit("forces") == "meV/Å"
    assert get_full_unit("stress") == "meV/atom"

    # Without normalization
    assert get_full_unit("energy", normalization={}) == "meV"


def test_get_symbol():
    assert get_symbol("energy") == "E"
    assert get_symbol("forces") == "F"
    assert get_symbol("stress") == "σ"
    # Unknown defaults to key name
    assert get_symbol("unknown") == "unknown"


def test_custom_properties():
    custom = {
        **DEFAULT_PROPERTIES,
        "dipole": {"report_unit": (1, "Debye"), "symbol": "μ"},
    }
    assert get_scale("dipole", custom) == 1
    assert get_base_unit("dipole", custom) == "Debye"
    assert get_full_unit("dipole", custom, {}) == "Debye"
    assert get_full_unit("dipole", custom, {"dipole": "atom"}) == "Debye/atom"
    assert get_symbol("dipole", custom) == "μ"


# -- Txt logger tests --


@pytest.fixture
def sample_metrics():
    return {
        "energy": {"r2": 99.5, "mae": 0.001, "rmse": 0.002},
        "forces": {"r2": 98.0, "mae": 0.01, "rmse": 0.02},
    }


def test_txt_logger_creates_files(sample_metrics):
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        logger = Txt(keys=["energy", "forces"], workdir=workdir)
        logger(0, 1e-3, sample_metrics, 1e-4, sample_metrics)

        assert (workdir / "logs" / "train.txt").exists()
        assert (workdir / "logs" / "valid.txt").exists()


def test_txt_logger_scaling():
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        logger = Txt(keys=["energy"], workdir=workdir)

        # MAE of 0.001 should become 1.0 after scaling by 1000
        train_metrics = {"energy": {"r2": 99.5, "mae": 0.001, "rmse": 0.002}}

        logger(0, 1e-3, train_metrics, 1e-4, train_metrics)

        content = (workdir / "logs" / "train.txt").read_text()
        # Scaled MAE (1.0) appears in scientific notation
        assert "1.00e+00" in content


def test_txt_logger_header_units(sample_metrics):
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        logger = Txt(keys=["energy", "forces"], workdir=workdir)
        logger(0, 1e-3, sample_metrics, 1e-4, sample_metrics)

        content = (workdir / "logs" / "train.txt").read_text()
        # Units appear in header
        assert "meV/atom (E)" in content
        assert "meV/Å (F)" in content


def test_txt_logger_custom_properties():
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        custom_properties = {
            **DEFAULT_PROPERTIES,
            "dipole": {"report_unit": (1, "Debye"), "symbol": "μ"},
        }

        logger = Txt(
            keys=["energy", "dipole"],
            workdir=workdir,
            properties=custom_properties,
            normalization=DEFAULT_NORMALIZATION,
        )

        train_metrics = {
            "energy": {"r2": 99.5, "mae": 0.001, "rmse": 0.002},
            "dipole": {"r2": 95.0, "mae": 0.5, "rmse": 0.6},
        }

        logger(0, 1e-3, train_metrics, 1e-4, train_metrics)

        content = (workdir / "logs" / "train.txt").read_text()
        # Custom property appears with correct unit
        assert "Debye (μ)" in content
        # Dipole MAE 0.5 * 1 (scale) = 0.5
        assert "5.00e-01" in content


def test_txt_logger_r2_not_scaled():
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        logger = Txt(
            keys=["energy"],
            metrics={"energy": ["r2"]},
            workdir=workdir,
        )

        # R² should NOT be scaled (99.5 stays 99.5, not 99500)
        train_metrics = {"energy": {"r2": 99.5}}

        logger(0, 1e-3, train_metrics, 1e-4, train_metrics)

        content = (workdir / "logs" / "train.txt").read_text()
        assert "99.500" in content
