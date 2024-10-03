import jax
import jax.numpy as jnp

from flax.linen import Module, compact

from jaxtyping import Array, Bool, Float, Int


class LennardJones(Module):
    cutoff: float
    onset: float
    initial_sigma: float
    initial_epsilon: float
    ensemble: bool = False

    @compact
    def __call__(
        self,
        R_ij: Float[Array, "pairs 3"],
        i: Int[Array, " pairs"],
        j: Int[Array, " pairs"],
        Z_i: Int[Array, " nodes"],
        pair_mask: Bool[Array, " pairs"],
        node_mask: Bool[Array, " nodes"],
    ):
        sigma = self.param("sigma", constant(self.initial_sigma))
        epsilon = self.param("epsilon", constant(self.initial_epsilon))

        energy = (
            lennard_jones(
                R_ij,
                i,
                Z_i,
                pair_mask,
                sigma=sigma,
                epsilon=epsilon,
                cutoff=self.cutoff,
                onset=self.onset,
            )
            * node_mask
        )

        if not self.ensemble:
            return energy
        else:
            return jnp.stack([energy - 1.0, energy, energy + 1.0]).T

    def dummy_inputs(self, dtype=jnp.float32):
        return (
            jnp.array([[0, 0, 0], [1, 1, 1]], dtype=dtype),
            jnp.array([0, 1]),
            jnp.array([1, 0]),
            jnp.array([0, 0]),
            jnp.array([True, True]),
            jnp.array([True, True]),
        )


def constant(value):
    def _constant(*no, **thanks):
        return jnp.array(value, dtype=jnp.float32)

    return _constant


def lennard_jones(R_ij, i, Z_i, pair_mask, sigma=2.0, epsilon=1.5, cutoff=10.0, onset=6.0):
    # we assume double counting, so 4*epsilon/2 is the prefactor
    factor = 2 * epsilon
    sigma = sigma
    cutoff2 = cutoff**2
    onset2 = onset**2
    zero = 0.0
    one = 1.0

    def pairwise_energy_fn(dr):
        mask = dr > 0.0

        inverse_r = jnp.where(mask, 1.0 / dr, 0.0)
        inverse_r6 = inverse_r**6
        inverse_r12 = inverse_r6 * inverse_r6

        return factor * (sigma**12 * inverse_r12 - sigma**6 * inverse_r6)

    def cutoff_fn(dr):
        # inspired by jax-md, which in turns uses HOOMD-BLUE

        distance2 = dr**2

        # in between onset and infinity:
        # either our mollifier or zero
        after_onset = jnp.where(
            distance2 < cutoff2,
            (cutoff2 - distance2) ** 2
            * (cutoff2 + 2.0 * distance2 - 3.0 * onset2)
            / (cutoff2 - onset2) ** 3,
            zero,
        )

        # do nothing before onset, then mollify
        return jnp.where(
            distance2 < onset2,
            one,
            after_onset,
        )

    def pair_lj(dr):
        return cutoff_fn(dr) * pairwise_energy_fn(dr)

    distances = distance(R_ij)
    contributions = jax.vmap(pair_lj)(distances)
    out = contributions * pair_mask

    out = jax.ops.segment_sum(out, i, Z_i.shape[0], indices_are_sorted=False)

    return out


def squared_distance(R):
    return jnp.sum(R**2, axis=-1)


def distance(R):
    # todo: can this be made more elegant?
    # see: https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    r2 = squared_distance(R)
    mask = r2 > 0
    safe_r2 = jnp.where(mask, r2, 0)
    return jnp.where(mask, jnp.sqrt(safe_r2), 0)
