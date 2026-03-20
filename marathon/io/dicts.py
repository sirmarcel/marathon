"""Turning dataclasses (in particular flax.nn.Module) into dicts and vice versa.

`dataclass` provides a way to get a `dict` from an object. All we do is
to add an identifier for the classname that we can then import. The
result of this is a "spec dict", an idea from https://github.com/sirmarcel/specable.

It looks like this:

```
{handle: payload}
```

"handle" is a string, identifies the class to be instantiated
"payload" is a a mapping (normally dict), that we pass to __init__

"""

import dataclasses
import importlib
from collections.abc import Mapping

# -- main functionality --


def to_dict(module):
    """Serialize a dataclass (e.g. flax Module) to a spec dict: {qualified_classname: kwargs}.

    Strips parent/name fields (assumes flax Module).
    """
    handle = f"{module.__module__}.{module.__class__.__name__}"

    inner = dataclasses.asdict(module)

    # redundant information
    del inner["parent"]
    del inner["name"]

    return {handle: inner}


def from_dict(dct, allow_stubs=False, default_namespace=None):
    """Reconstruct a dataclass instance from a spec dict by dynamically importing the class."""
    handle, inner = parse_dict(dct, allow_stubs=allow_stubs)

    if default_namespace and "." not in handle:
        handle = f"{default_namespace}.{handle}"

    module = ".".join(handle.split(".")[:-1])
    module = importlib.import_module(module)

    kind = handle.split(".")[-1]

    cls = getattr(module, kind)

    return cls(**inner)


# -- helpers --


def is_valid(dct):
    if isinstance(dct, Mapping):
        if len(dct) == 1:
            handle = next(iter(dct))
            if isinstance(handle, str):
                if isinstance(dct[handle], Mapping):
                    return True

    return False


def parse_dict(dct, allow_stubs=False):
    if allow_stubs and isinstance(dct, str):
        return dct, {}

    if not is_valid(dct):
        raise ValueError("Improper spec dict format: " + str(dct))

    handle = next(iter(dct))
    inner = dct[handle]

    return handle, inner
