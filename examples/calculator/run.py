"""
Calculator example: wrap a marathon model as an ASE calculator.

Uses the LJ toy model directly (no checkpoint) to demonstrate the calculator.
"""

import numpy as np
import jax

from calculator import Calculator
from lj_data import steps
from model import LennardJones

key = jax.random.key(1)
key, init_key = jax.random.split(key)

# -- model setup --

model = LennardJones(
    initial_sigma=2.0,
    initial_epsilon=1.5,
    cutoff=6.0,
    onset=5.0,
)
params = model.init(init_key, *model.dummy_inputs())

# trivial baseline (no per-element offset for this demo)
species_weights = {18: 0.0}  # Argon

# -- create calculator --

calc = Calculator(
    apply_fn=model.apply,
    species_weights=species_weights,
    params=params,
    cutoff=model.cutoff,
    stress=False,
)

# -- run on a few structures --

for i, atoms in enumerate(steps[:5]):
    results = calc.calculate(atoms)
    ref_energy = atoms.get_potential_energy()

    print(f"structure {i}:")
    print(f"  predicted energy: {results['energy']:.4f} eV")
    print(f"  reference energy: {ref_energy:.4f} eV")
    print(
        f"  force MAE: {np.mean(np.abs(results['forces'] - atoms.get_forces())):.4f} eV/A"
    )

print("done!")
