import jax
import jax.numpy as jnp


def get_predict_fn(apply_fn=None, stress=False, energy_fn=None):
    if apply_fn is not None and energy_fn is None:

        def energy_fn(params, batch):
            energies = apply_fn(
                params,
                batch.edges,
                batch.centers,
                batch.others,
                batch.nodes,
                batch.edge_mask,
                batch.node_mask,
            )
            energies *= batch.node_mask

            return jnp.sum(energies), energies

    energy_and_derivatives_fn = jax.value_and_grad(
        energy_fn, allow_int=True, has_aux=True, argnums=1
    )

    def predict(params, batch):
        batch_energy_and_atom_energies, grads = energy_and_derivatives_fn(params, batch)
        _, energies = batch_energy_and_atom_energies

        energy = jax.ops.segment_sum(
            energies, batch.node_to_graph, batch.graph_mask.shape[0]
        )

        R_ij = batch.edges * batch.edge_mask[..., None]
        dR_ij = grads.edges * batch.edge_mask[..., None]

        forces_1 = jax.ops.segment_sum(
            dR_ij, batch.centers, batch.nodes.shape[0], indices_are_sorted=False
        )
        forces_2 = jax.ops.segment_sum(
            dR_ij, batch.others, batch.nodes.shape[0], indices_are_sorted=False
        )

        forces = (forces_1 - forces_2) * batch.node_mask[..., None]

        results = {"energy": energy, "forces": forces}

        if stress:
            pre_stress = jnp.einsum("pa,pb->pab", R_ij, dR_ij)

            results["stress"] = (
                jax.ops.segment_sum(
                    pre_stress,
                    batch.edge_to_graph,
                    batch.graph_mask.shape[0],
                    indices_are_sorted=False,
                )
                * batch.graph_mask[..., None, None]
            )

        return results

    return predict
