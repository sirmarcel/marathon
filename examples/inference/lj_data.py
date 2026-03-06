from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs


def to_log(atoms):
    calc = SinglePointCalculator(atoms, **atoms.calc.results)
    atoms = atoms.copy()
    atoms.calc = calc

    return atoms


atoms = bulk("Ar", cubic=False) * [5, 5, 5]
MaxwellBoltzmannDistribution(atoms, temperature_K=80)
Stationary(atoms)

sigma = 2.0
epsilon = 1.5
rc = 6.0
ro = 5.0

calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, smooth=True)


dt = 1 * fs
nsteps = 240
atoms.calc = calc
verlet = VelocityVerlet(atoms, timestep=dt)

steps = []
for _ in verlet.irun(steps=nsteps):
    steps.append(to_log(atoms))
