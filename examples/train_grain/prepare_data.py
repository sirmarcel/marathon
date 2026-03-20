"""
Prepare LJ trajectory data for grain-based training.

This script converts the in-memory LJ trajectory into mmap format
suitable for DataSource.

Usage: python prepare_data.py
"""

import jax

from lj_data import steps

from marathon import comms
from marathon.data import get_splits
from marathon.grain import prepare

reporter = comms.reporter()
reporter.start("prepare_data")

# Split data
n_data = len(steps)
n_train = 150
n_valid = 50

key = jax.random.key(0)
idx_train, idx_valid, _ = get_splits(
    n_data, n_train, n_valid, n_data - n_train - n_valid, key
)

data_train = [steps[i] for i in idx_train]
data_valid = [steps[i] for i in idx_valid]

# Prepare mmap datasets
reporter.step("preparing training data")
prepare(data_train, folder="data_train", batch_size=50)

reporter.step("preparing validation data")
prepare(data_valid, folder="data_valid", batch_size=50)

reporter.done()
comms.state("Data preparation complete!")
comms.talk("Created data_train/ and data_valid/ directories")
comms.talk("You can now run: python run.py")
