import warnings

warnings.warn(
    "marathon.extra.hermes is deprecated, use marathon.grain instead",
    DeprecationWarning,
    stacklevel=2,
)

from marathon.grain import *  # noqa: F403
from marathon.grain import __all__ as __all__
