import jax
import jax.numpy as jnp


def get_predict_fn(
    apply_fn,
    stress=False,
    derivative_variance=False,
    derivative_variance_config={"scan": {"unroll": 2, "vmap": 4}},
):
    # we assume that the apply_fn returns [node, ens] outputs, which
    # correspond to per-atom energy contributions per ensemble member

    # jax doesn't like using the batch namedtuple here, so we have
    # to make this horribly long list of arguments
    def energy_fn(
        params,
        edges,
        centers,
        others,
        nodes,
        edge_mask,
        node_mask,
    ):
        energies = apply_fn(
            params,
            edges,
            centers,
            others,
            nodes,
            edge_mask,
            node_mask,
        )  # -> [node, ens]
        energies *= node_mask[..., None]

        if not derivative_variance:
            # we don't need to take the derivative wrt each ensemble prediction,
            # so we take the mean already here
            mean = jnp.mean(energies, axis=-1)
            return jnp.sum(mean), energies
        else:
            # we can only sum over the nodes in the batch
            # we need the ensemble dimension
            return jnp.sum(energies, axis=0), energies

    if not derivative_variance:
        grad_and_energies_fn = jax.grad(energy_fn, has_aux=True, argnums=1)
    else:
        grad_and_energies_fn = get_grad_and_energies_fn_with_derivatives(
            energy_fn, derivative_variance_config
        )

    def predict(params, batch):
        grads, energies = grad_and_energies_fn(
            params,
            batch.edges,
            batch.centers,
            batch.others,
            batch.nodes,
            batch.edge_mask,
            batch.node_mask,
        )

        energy_ens = jax.ops.segment_sum(
            energies, batch.node_to_graph, batch.graph_mask.shape[0]
        )  # -> [graph, ens]
        energy_mean = jnp.mean(energy_ens, axis=-1)  # -> [graph]
        energy_var = jnp.var(energy_ens, axis=-1, ddof=1)  # -> [graph]

        results = {
            "energy": energy_mean,
            "energy_var": energy_var,
            "energy_ens": energy_ens,
        }

        R_ij = batch.edges * batch.edge_mask[..., None]

        if not derivative_variance:
            dR_ij = grads * batch.edge_mask[..., None]
            forces_1 = jax.ops.segment_sum(
                dR_ij, batch.centers, batch.nodes.shape[0], indices_are_sorted=False
            )
            forces_2 = jax.ops.segment_sum(
                dR_ij, batch.others, batch.nodes.shape[0], indices_are_sorted=False
            )

            forces = (forces_1 - forces_2) * batch.node_mask[..., None]

            results["forces"] = forces

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

        else:
            # grads is [ens, pair, 3], so we move the ens axis to the back, so
            # we can do segment_sum more easily
            grads = jnp.moveaxis(grads, 0, -1)
            dR_ij = grads * batch.edge_mask[..., None, None]  # -> [pair, 3, ens]

            forces_1 = jax.ops.segment_sum(
                dR_ij, batch.centers, batch.nodes.shape[0], indices_are_sorted=False
            )
            forces_2 = jax.ops.segment_sum(
                dR_ij, batch.others, batch.nodes.shape[0], indices_are_sorted=False
            )

            forces_ens = (forces_1 - forces_2) * batch.node_mask[
                ..., None, None
            ]  # -> [node, 3, ens]
            forces_mean = jnp.mean(forces_ens, axis=-1)
            forces_var = jnp.var(forces_ens, axis=-1, ddof=1)

            results["forces"] = forces_mean
            results["forces_var"] = forces_var
            results["forces_ens"] = forces_ens

            if stress:
                pre_stress = jnp.einsum("pa,pbe->pabe", R_ij, dR_ij)

                stress_ens = (
                    jax.ops.segment_sum(
                        pre_stress,
                        batch.edge_to_graph,
                        batch.graph_mask.shape[0],
                        indices_are_sorted=False,
                    )
                    * batch.graph_mask[..., None, None, None]
                )

                results["stress"] = jnp.mean(stress_ens, axis=-1)
                results["stress_var"] = jnp.var(stress_ens, axis=-1, ddof=1)
                results["stress_ens"] = stress_ens

            return results

    return predict


def get_grad_and_energies_fn_with_derivatives(energy_fn, config):
    if "jacrev" in config:
        return jax.jacrev(energy_fn, has_aux=True, argnums=1)

    elif "scan" in config:
        unroll = config["scan"]["unroll"]
        vmap = config["scan"]["vmap"]

        def grad_and_energies_fn(
            params,
            edges,
            centers,
            others,
            nodes,
            edge_mask,
            node_mask,
        ):
            wrapped = lambda x: energy_fn(
                params, x, centers, others, nodes, edge_mask, node_mask
            )

            total, vjpfun, energies = jax.vjp(
                wrapped,
                edges,
                has_aux=True,
            )
            ens = total.shape[0]

            def to_scan(ignored, v):
                return None, jax.vmap(vjpfun)(v)

            def scanned_vjp(vs):
                return jax.lax.scan(to_scan, None, xs=vs, unroll=unroll)[1][0]

            basis = jnp.eye(ens, dtype=edges.dtype)

            basis = basis.reshape(ens // vmap, vmap, -1)

            jacobian = scanned_vjp(basis)

            jacobian = jacobian.reshape(ens, edges.shape[0], 3)

            return jacobian, energies

        return grad_and_energies_fn

    else:
        raise NotImplementedError
