import numpy as np
import jax
import jax.numpy as jnp

from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import PropertyNotImplementedError, compare_atoms
from glp import atoms_to_system
from glp.graph import system_to_graph
from glp.neighborlist import neighbor_list

from marathon import comms
from marathon.data import Batch


class Calculator(GetPropertiesMixin):
    # ase/vibes compatibility. not used!
    name = "marathon"
    parameters = {}

    def todict(self):
        return self.parameters

    implemented_properties = [
        "energy",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        apply_fn,
        species_weights,
        params,
        cutoff,
        skin=0.1,
        atoms=None,
        stress=False,
        uq=False,
    ):
        self.params = params
        self.cutoff = cutoff
        self.skin = 0.1

        if not stress:
            self.implemented_properties = ["energy", "forces"]

        if uq:
            from marathon.ensemble import get_predict_fn

            predict_fn = get_predict_fn(
                apply_fn,
                stress=stress,
                derivative_variance=True,
                derivative_variance_config={"scan": {"unroll": 1, "vmap": 1}},
            )
        else:
            from marathon.evaluate import get_predict_fn

            predict_fn = get_predict_fn(apply_fn, stress=stress)

        self.predict_fn = predict_fn
        self.species_weights = species_weights

        self.atoms = None
        self.graph = None
        self.results = {}
        if atoms is not None:
            self.setup(atoms)

    @classmethod
    def from_checkpoint(
        cls,
        folder,
        **kwargs,
    ):
        from pathlib import Path

        from myrto.engine import from_dict, read_yaml

        folder = Path(folder)

        model = from_dict(read_yaml(folder / "model/model.yaml"))

        _ = model.init(jax.random.key(1), *model.dummy_inputs())

        baseline = read_yaml(folder / "model/baseline.yaml")
        species_to_weight = baseline["elemental"]

        from marathon.emit.checkpoint import read_msgpack

        params = read_msgpack(folder / "model/model.msgpack")

        return cls(model.apply, species_to_weight, params, model.cutoff, **kwargs)

    def update(self, atoms):
        changes = compare_atoms(self.atoms, atoms)

        if len(changes) > 0:
            self.results = {}
            self.atoms = atoms.copy()

            if self.need_setup(changes):
                self.setup(atoms)

    def need_setup(self, changes):
        return "pbc" in changes or "numbers" in changes

    def setup(self, atoms):
        system = atoms_to_system(atoms)
        state, update_fn = neighbor_list(
            system, self.cutoff, self.skin, capacity_multiplier=1.1
        )

        def calculate_fn(system, neighbors):
            neighbors = update_fn(system, neighbors)
            graph = system_to_graph(system, neighbors)

            batch = Batch(
                graph.nodes,
                graph.edges,
                graph.centers,
                graph.others,
                jnp.zeros_like(graph.nodes, dtype=int),
                jnp.zeros_like(graph.centers, dtype=int),
                jnp.array([True]),
                jnp.ones_like(graph.nodes, dtype=bool),
                graph.mask,
                {},
            )

            return self.predict_fn(self.params, batch), neighbors, graph

        self.calculate_fn = jax.jit(calculate_fn)
        self.state = state
        self.atoms = atoms.copy()

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=None,
        **kwargs,
    ):
        self.update(atoms)

        system = atoms_to_system(atoms)
        results, state, graph = self.calculate_fn(system, self.state)

        if state.overflow:
            comms.talk("overflow. redoing calculation...")
            self.setup(atoms)
            results, state, graph = self.calculate_fn(system, self.state)
            if state.overflow:
                msg = []
                msg.append("Encountered an overflow twice in a row. This means")
                msg.append("that the neighborlist cannot be computed correctly,")
                msg.append("probably because the cutoff is too large to fit into")
                msg.append("the simulation cell. You can try decreasing `skin`.")
                comms.state(msg, title="‼️ Multiple overflow ‼️")
                raise RuntimeError

        actual_results = {k: np.array(v.squeeze()) for k, v in results.items()}

        for key in self.implemented_properties:
            if key + "_var" in actual_results:
                actual_results[key + "_std"] = np.sqrt(actual_results[key + "_var"])

        energy_offset = np.sum(
            [self.species_weights[Z] for Z in atoms.get_atomic_numbers()]
        )
        actual_results["energy"] += energy_offset

        self.results = actual_results
        return actual_results

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f"{name} property not implemented")

        self.update(atoms)

        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms=atoms)

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError(f"{name} property not present in results!")

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_potential_energy(self, atoms=None):
        return self.get_property(name="energy", atoms=atoms)
