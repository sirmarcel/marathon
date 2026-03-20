"""
ASE calculator wrapping a marathon model.

Uses marathon's own data pipeline (vesin neighborlists) instead of glp.
"""

import numpy as np
import jax
import jax.numpy as jnp

from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import PropertyNotImplementedError, compare_atoms

from marathon.data import Batch
from marathon.data.sample import to_structure


class Calculator(GetPropertiesMixin):
    name = "marathon"
    parameters = {}

    def todict(self):
        return self.parameters

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        apply_fn,
        species_weights,
        params,
        cutoff,
        atoms=None,
        stress=False,
    ):
        self.params = params
        self.cutoff = cutoff

        if not stress:
            self.implemented_properties = ["energy", "forces"]

        from marathon.evaluate import get_predict_fn

        self.predict_fn = get_predict_fn(apply_fn, stress=stress)
        self.species_weights = species_weights

        self.atoms = None
        self.results = {}
        if atoms is not None:
            self.atoms = atoms.copy()

    @classmethod
    def from_checkpoint(cls, folder, **kwargs):
        from pathlib import Path

        from marathon.emit.checkpoint import read_msgpack
        from marathon.io import from_dict, read_yaml

        folder = Path(folder)

        model = from_dict(read_yaml(folder / "model/model.yaml"))
        _ = model.init(jax.random.key(1), *model.dummy_inputs())

        baseline = read_yaml(folder / "model/baseline.yaml")
        species_to_weight = baseline["elemental"]

        params = read_msgpack(folder / "model/model.msgpack")

        return cls(model.apply, species_to_weight, params, model.cutoff, **kwargs)

    def update(self, atoms):
        changes = compare_atoms(self.atoms, atoms)
        if len(changes) > 0:
            self.results = {}
            self.atoms = atoms.copy()

    def _atoms_to_batch(self, atoms):
        """Convert ASE Atoms to a marathon Batch using vesin neighborlists."""
        structure = to_structure(atoms, self.cutoff)

        # Build a single-structure batch
        batch = Batch(
            atomic_numbers=jnp.array(structure["atomic_numbers"]),
            displacements=jnp.array(structure["displacements"]),
            centers=jnp.array(structure["centers"]),
            others=jnp.array(structure["others"]),
            atom_to_structure=jnp.zeros_like(
                jnp.array(structure["atomic_numbers"]), dtype=int
            ),
            pair_to_structure=jnp.zeros_like(jnp.array(structure["centers"]), dtype=int),
            structure_mask=jnp.array([True]),
            atom_mask=jnp.ones(len(structure["atomic_numbers"]), dtype=bool),
            pair_mask=jnp.array(structure["others"] >= 0),
            labels={},
        )
        return batch

    def calculate(self, atoms=None, properties=None, system_changes=None, **kwargs):
        self.update(atoms)

        batch = self._atoms_to_batch(atoms)
        results = jax.jit(self.predict_fn)(self.params, batch)

        actual_results = {k: np.array(v.squeeze()) for k, v in results.items()}

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
            raise PropertyNotImplementedError(f"{name} property not present in results!")

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_potential_energy(self, atoms=None):
        return self.get_property(name="energy", atoms=atoms)
