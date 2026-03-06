import warnings

warnings.warn(
    "marathon.extra.hermes is deprecated, use marathon.grain instead",
    DeprecationWarning,
    stacklevel=2,
)

from marathon.grain import *
from marathon.grain import __all__
