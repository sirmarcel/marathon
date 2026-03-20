from marathon.emit.properties import (
    DEFAULT_PROPERTIES,
    get_full_unit,
    get_scale,
    get_symbol,
)
from marathon.evaluate.properties import DEFAULT_NORMALIZATION


def format_metrics(
    metrics,
    keys=None,
    properties=DEFAULT_PROPERTIES,
    normalization=DEFAULT_NORMALIZATION,
    include_per_structure=False,
):
    # returns list of strings suitable for comms.state()
    # applies scaling (eV -> meV) to MAE/RMSE but not R2
    if keys is None:
        keys = list(metrics.keys())

    # Optionally include _per_structure variants
    if include_per_structure:
        actual_keys = list(keys)
        for key in keys:
            ps_key = f"{key}_per_structure"
            if ps_key in metrics and ps_key not in actual_keys:
                actual_keys.append(ps_key)
        keys = actual_keys

    msg = []
    for key in keys:
        if key not in metrics:
            continue
        m = metrics[key]
        symbol = get_symbol(key, properties)
        unit = get_full_unit(key, properties, normalization)
        scale = get_scale(key, properties)

        msg.append(f". {symbol}")
        msg.append(f".. R2  : {m['r2']:.3f} %")
        msg.append(f".. MAE : {m['mae'] * scale:.3e} {unit}")
        msg.append(f".. RMSE: {m['rmse'] * scale:.3e} {unit}")

    return msg


# -- test --


def test_format_metrics():
    # test with default properties
    metrics = {
        "energy": {"r2": 99.5, "mae": 0.001, "rmse": 0.002},
        "forces": {"r2": 98.0, "mae": 0.01, "rmse": 0.02},
    }

    result = format_metrics(metrics, keys=["energy", "forces"])

    assert ". E" in result
    assert ". F" in result
    assert ".. R2  : 99.500 %" in result
    # MAE 0.001 * 1000 = 1.0
    assert ".. MAE : 1.000e+00 meV/atom" in result
    # forces MAE 0.01 * 1000 = 10.0
    assert ".. MAE : 1.000e+01 meV/Å" in result

    # test with custom properties
    custom_properties = {
        **DEFAULT_PROPERTIES,
        "dipole": {"report_unit": (1, "Debye"), "symbol": "μ"},
    }

    metrics_with_dipole = {
        "energy": {"r2": 99.0, "mae": 0.001, "rmse": 0.002},
        "dipole": {"r2": 95.0, "mae": 0.5, "rmse": 0.6},
    }

    result = format_metrics(
        metrics_with_dipole,
        keys=["energy", "dipole"],
        properties=custom_properties,
        normalization=DEFAULT_NORMALIZATION,
    )

    assert ". μ" in result
    # dipole MAE 0.5 * 1 = 0.5
    assert ".. MAE : 5.000e-01 Debye" in result

    # test include_per_structure flag
    metrics_with_ps = {
        "energy": {"r2": 99.5, "mae": 0.001, "rmse": 0.002},
        "energy_per_structure": {"r2": 98.0, "mae": 0.005, "rmse": 0.01},
        "forces": {"r2": 98.0, "mae": 0.01, "rmse": 0.02},
    }

    # Default: exclude _per_structure
    result = format_metrics(metrics_with_ps, keys=["energy", "forces"])
    assert result.count(". E") == 1  # only one E entry

    # With flag: include _per_structure
    result = format_metrics(
        metrics_with_ps, keys=["energy", "forces"], include_per_structure=True
    )
    assert result.count(". E") == 2  # E appears twice (energy and energy_per_structure)
    # energy_per_structure: MAE 0.005 * 1000 = 5.0, unit = meV (no /atom)
    assert ".. MAE : 5.000e+00 meV" in result


test_format_metrics()
