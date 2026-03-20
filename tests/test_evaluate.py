import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util

from lj import LennardJones
from lj_data import epsilon, rc, ro, sigma, steps

from marathon.data import batch_samples, determine_max_sizes, to_sample
from marathon.evaluate import get_loss_fn, get_metrics_fn, get_predict_fn


def test():
    lj = LennardJones(cutoff=rc, onset=ro, initial_sigma=sigma, initial_epsilon=epsilon)
    params = lj.init(jax.random.key(0), *lj.dummy_inputs())

    keys = ["energy", "forces", "stress"]
    samples = [to_sample(atoms, rc, stress=True) for atoms in steps[:10]]

    num_nodes, num_edges = determine_max_sizes(samples, 5)

    batches = [
        batch_samples(samples[:5], num_nodes, num_edges, keys),
        batch_samples(samples[5:10], num_nodes, num_edges, keys),
    ]

    pred_fn = get_predict_fn(lj.apply, stress=True)

    predictions = [pred_fn(params, batch) for batch in batches]

    Na = len(steps[0])

    for i in range(5):
        np.testing.assert_allclose(
            predictions[0]["energy"][i], steps[i].get_potential_energy(), atol=5e-4
        )
        np.testing.assert_allclose(
            predictions[0]["forces"][i * Na : (i + 1) * Na],
            steps[i].get_forces(),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            predictions[0]["stress"][i],
            steps[i].get_stress(voigt=False) * steps[i].get_volume(),
            atol=5e-3,
        )

    for i in range(5):
        np.testing.assert_allclose(
            predictions[1]["energy"][i], steps[i + 5].get_potential_energy(), atol=5e-4
        )
        np.testing.assert_allclose(
            predictions[1]["forces"][i * Na : (i + 1) * Na],
            steps[i + 5].get_forces(),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            predictions[1]["stress"][i],
            steps[i + 5].get_stress(voigt=False) * steps[i + 5].get_volume(),
            atol=5e-3,
        )

    loss_fn = get_loss_fn(pred_fn, weights={"energy": 1.0, "forces": 1.0, "stress": 1.0})

    ready_batches = tree_util.tree_map(lambda *x: jnp.stack(x), *batches)

    loss, aux = jax.vmap(lambda x: loss_fn(params, x))(ready_batches)

    metrics_fn = get_metrics_fn(samples=samples, keys=keys)

    metrics = metrics_fn(aux)

    for v in metrics.values():
        assert v["r2"] > 99.999
