import numpy as np

from marathon.data import Sample
from marathon.evaluate.metrics import get_metrics_fn, get_stats

N = 100
Na = 125
batch_size = 2


def rmse(true, pred):
    """Root mean squared error."""
    return np.sqrt(np.mean((true - pred) ** 2))


def mae(true, pred):
    """Mean absolute error."""
    return np.mean(np.fabs(true - pred))


def cod(true, pred):
    """Coefficient of determination (R²). Can be negative, but <= 1.0."""
    mean = np.mean(true)
    sum_of_squares = np.sum((true - mean) ** 2)
    sum_of_residuals = np.sum((true - pred) ** 2)
    return 1.0 - (sum_of_residuals / sum_of_squares)


def test_standard_properties():
    keys = ["energy", "forces", "stress"]

    dummy_energy = np.random.random(N)
    dummy_forces = np.random.random((N, Na, 3))
    dummy_stress = np.random.random((N, 3, 3))

    samples = [
        Sample(
            {"positions": np.zeros((Na, 3))},
            {
                "energy": dummy_energy[i],
                "forces": dummy_forces[i],
                "stress": dummy_stress[i],
                "num_atoms": Na,
            },
        )
        for i in range(N)
    ]

    metrics_fn = get_metrics_fn(samples=samples, keys=keys)

    dummy_preds = {
        "energy": np.random.random(N),
        "forces": np.random.random((N, Na, 3)),
        "stress": np.random.random((N, 3, 3)),
    }

    tmp = {}
    for key in keys:
        tmp[f"{key}_abs"] = []
        tmp[f"{key}_sq"] = []
        tmp[f"{key}_n"] = []

    for batch in range(N // batch_size):
        start = batch * batch_size
        stop = (batch + 1) * batch_size

        energy = (dummy_preds["energy"][start:stop] - dummy_energy[start:stop]) / Na
        forces = dummy_preds["forces"][start:stop] - dummy_forces[start:stop]
        stress = (dummy_preds["stress"][start:stop] - dummy_stress[start:stop]) / Na

        tmp["energy_abs"].append(np.abs(energy).sum())
        tmp["forces_abs"].append(np.abs(forces).sum(axis=(0, 1)))
        tmp["stress_abs"].append(np.abs(stress).sum(axis=0))

        tmp["energy_sq"].append((energy**2).sum())
        tmp["forces_sq"].append((forces**2).sum(axis=(0, 1)))
        tmp["stress_sq"].append((stress**2).sum(axis=0))

        tmp["energy_n"].append(2)
        tmp["forces_n"].append(2 * Na)
        tmp["stress_n"].append(2)

    auxs = {key: np.stack(val) for key, val in tmp.items()}

    metrics = metrics_fn(auxs)

    np.testing.assert_allclose(
        metrics["energy"]["r2"],
        100 * cod(dummy_energy / Na, dummy_preds["energy"] / Na),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["energy"]["mae"],
        mae(dummy_energy / Na, dummy_preds["energy"] / Na),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["energy"]["rmse"],
        rmse(dummy_energy / Na, dummy_preds["energy"] / Na),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["forces"]["r2"],
        100 * cod(dummy_forces, dummy_preds["forces"].reshape(N, Na, 3)),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["forces"]["mae"],
        mae(dummy_forces, dummy_preds["forces"].reshape(N, Na, 3)),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["forces"]["rmse"],
        rmse(dummy_forces, dummy_preds["forces"].reshape(N, Na, 3)),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["stress"]["r2"],
        100 * cod(dummy_stress / Na, dummy_preds["stress"] / Na),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["stress"]["mae"],
        mae(dummy_stress / Na, dummy_preds["stress"] / Na),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["stress"]["rmse"],
        rmse(dummy_stress / Na, dummy_preds["stress"] / Na),
        rtol=1e-6,
    )

    for i in range(3):
        np.testing.assert_allclose(
            metrics["forces"]["r2_per_component"][i],
            100
            * cod(dummy_forces[..., i], dummy_preds["forces"].reshape(N, Na, 3)[..., i]),
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            metrics["forces"]["mae_per_component"][i],
            mae(dummy_forces[..., i], dummy_preds["forces"].reshape(N, Na, 3)[..., i]),
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            metrics["forces"]["rmse_per_component"][i],
            rmse(dummy_forces[..., i], dummy_preds["forces"].reshape(N, Na, 3)[..., i]),
            rtol=1e-6,
        )

    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(
                metrics["stress"]["r2_per_component"][i, j],
                100
                * cod(dummy_stress[..., i, j] / Na, dummy_preds["stress"][..., i, j] / Na),
                rtol=1e-6,
            )

            np.testing.assert_allclose(
                metrics["stress"]["mae_per_component"][i, j],
                mae(dummy_stress[..., i, j] / Na, dummy_preds["stress"][..., i, j] / Na),
                rtol=1e-6,
            )

            np.testing.assert_allclose(
                metrics["stress"]["rmse_per_component"][i, j],
                rmse(dummy_stress[..., i, j] / Na, dummy_preds["stress"][..., i, j] / Na),
                rtol=1e-6,
            )

    stats = get_stats(samples, keys=keys)
    np.testing.assert_allclose(
        dummy_energy.mean() / Na,
        stats["energy"]["mean"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        dummy_forces.mean(),
        stats["forces"]["mean"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        dummy_stress.mean() / Na,
        stats["stress"]["mean"],
        rtol=1e-6,
    )


def test_custom_properties():
    # Custom scalar (per-structure, no components, no atom-normalization)
    dummy_custom_scalar = np.random.random(N)

    # Custom per-structure with components, shape (4,) - e.g., quaternion
    dummy_quaternion = np.random.random((N, 4))

    # Custom per-atom with components, shape ("atom", 2)
    dummy_peratom = np.random.random((N, Na, 2))

    samples_custom = [
        Sample(
            {"positions": np.zeros((Na, 3))},
            {
                "custom_scalar": dummy_custom_scalar[i],
                "custom_quat": dummy_quaternion[i],
                "custom_peratom": dummy_peratom[i],
                "num_atoms": Na,
            },
        )
        for i in range(N)
    ]

    keys_custom = ["custom_scalar", "custom_quat", "custom_peratom"]

    custom_properties = {
        "custom_scalar": {"shape": (1,), "storage": "atoms.info"},
        "custom_quat": {"shape": (4,), "storage": "atoms.info"},
        "custom_peratom": {"shape": ("atom", 2), "storage": "atoms.arrays"},
    }

    # Custom properties have no default normalization (not atom-normalized)
    custom_normalization = {}

    metrics_fn_custom = get_metrics_fn(
        samples=samples_custom,
        keys=keys_custom,
        normalization=custom_normalization,
        properties=custom_properties,
    )

    dummy_preds_custom = {
        "custom_scalar": np.random.random(N),
        "custom_quat": np.random.random((N, 4)),
        "custom_peratom": np.random.random((N, Na, 2)),
    }

    tmp_custom = {}
    for key in keys_custom:
        tmp_custom[f"{key}_abs"] = []
        tmp_custom[f"{key}_sq"] = []
        tmp_custom[f"{key}_n"] = []

    for batch in range(N // batch_size):
        start = batch * batch_size
        stop = (batch + 1) * batch_size

        # custom_scalar: no normalization
        scalar_res = (
            dummy_preds_custom["custom_scalar"][start:stop]
            - dummy_custom_scalar[start:stop]
        )
        tmp_custom["custom_scalar_abs"].append(np.abs(scalar_res).sum())
        tmp_custom["custom_scalar_sq"].append((scalar_res**2).sum())
        tmp_custom["custom_scalar_n"].append(batch_size)

        # custom_quat: per-structure with 4 components, no normalization
        quat_res = (
            dummy_preds_custom["custom_quat"][start:stop] - dummy_quaternion[start:stop]
        )
        tmp_custom["custom_quat_abs"].append(np.abs(quat_res).sum(axis=0))  # shape (4,)
        tmp_custom["custom_quat_sq"].append((quat_res**2).sum(axis=0))
        tmp_custom["custom_quat_n"].append(batch_size)

        # custom_peratom: per-atom with 2 components
        peratom_res = (
            dummy_preds_custom["custom_peratom"][start:stop] - dummy_peratom[start:stop]
        )
        tmp_custom["custom_peratom_abs"].append(
            np.abs(peratom_res).sum(axis=(0, 1))
        )  # shape (2,)
        tmp_custom["custom_peratom_sq"].append((peratom_res**2).sum(axis=(0, 1)))
        tmp_custom["custom_peratom_n"].append(batch_size * Na)

    auxs_custom = {key: np.stack(val) for key, val in tmp_custom.items()}

    metrics_custom = metrics_fn_custom(auxs_custom)

    # custom_scalar (no per_component)
    np.testing.assert_allclose(
        metrics_custom["custom_scalar"]["mae"],
        mae(dummy_custom_scalar, dummy_preds_custom["custom_scalar"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_scalar"]["rmse"],
        rmse(dummy_custom_scalar, dummy_preds_custom["custom_scalar"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_scalar"]["r2"],
        100 * cod(dummy_custom_scalar, dummy_preds_custom["custom_scalar"]),
        rtol=1e-6,
    )
    assert "mae_per_component" not in metrics_custom["custom_scalar"]

    # custom_quat (4 components, per-structure)
    np.testing.assert_allclose(
        metrics_custom["custom_quat"]["mae"],
        mae(dummy_quaternion, dummy_preds_custom["custom_quat"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_quat"]["rmse"],
        rmse(dummy_quaternion, dummy_preds_custom["custom_quat"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_quat"]["r2"],
        100 * cod(dummy_quaternion, dummy_preds_custom["custom_quat"]),
        rtol=1e-6,
    )

    for i in range(4):
        np.testing.assert_allclose(
            metrics_custom["custom_quat"]["mae_per_component"][i],
            mae(dummy_quaternion[..., i], dummy_preds_custom["custom_quat"][..., i]),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            metrics_custom["custom_quat"]["rmse_per_component"][i],
            rmse(dummy_quaternion[..., i], dummy_preds_custom["custom_quat"][..., i]),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            metrics_custom["custom_quat"]["r2_per_component"][i],
            100 * cod(dummy_quaternion[..., i], dummy_preds_custom["custom_quat"][..., i]),
            rtol=1e-6,
        )

    # custom_peratom (2 components, per-atom)
    np.testing.assert_allclose(
        metrics_custom["custom_peratom"]["mae"],
        mae(dummy_peratom, dummy_preds_custom["custom_peratom"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_peratom"]["rmse"],
        rmse(dummy_peratom, dummy_preds_custom["custom_peratom"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_peratom"]["r2"],
        100 * cod(dummy_peratom, dummy_preds_custom["custom_peratom"]),
        rtol=1e-6,
    )

    for i in range(2):
        np.testing.assert_allclose(
            metrics_custom["custom_peratom"]["mae_per_component"][i],
            mae(dummy_peratom[..., i], dummy_preds_custom["custom_peratom"][..., i]),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            metrics_custom["custom_peratom"]["rmse_per_component"][i],
            rmse(dummy_peratom[..., i], dummy_preds_custom["custom_peratom"][..., i]),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            metrics_custom["custom_peratom"]["r2_per_component"][i],
            100 * cod(dummy_peratom[..., i], dummy_preds_custom["custom_peratom"][..., i]),
            rtol=1e-6,
        )

    stats_custom = get_stats(
        samples_custom,
        keys=keys_custom,
        normalization=custom_normalization,
        properties=custom_properties,
    )
    np.testing.assert_allclose(
        dummy_custom_scalar.mean(),
        stats_custom["custom_scalar"]["mean"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        dummy_quaternion.mean(),
        stats_custom["custom_quat"]["mean"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        dummy_peratom.mean(),
        stats_custom["custom_peratom"]["mean"],
        rtol=1e-6,
    )
