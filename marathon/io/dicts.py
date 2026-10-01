"""Turning dataclasses (in particular flax.nn.Module) into dicts and vice versa.

`dataclass` provides a way to get a `dict` from an object. All we do is
to add an identifier for the classname that we can then import. The
result of this is a "spec dict", an idea from https://github.com/sirmarcel/specable.

It looks like this:

```
{handle: payload}
```

"handle" is a string, identifies the class to be instantiated
"payload" is a mapping (normally dict), that we pass to __init__

If the class is a dataclass (mostly flax/linen modules), `from_dict` coerces
the payload into the types its fields declare. This counters type drift in a
yaml roundtrip: yaml writes tuple and list as the same `[...]`, so a tuple comes
back as a list (and a `FrozenDict` as a `dict`). The restored module then differs
from the original, which breaks comparing configs between runs, and a list field
makes it unhashable, which breaks using it as a static argument under `jit`.

Coercion is lenient: whatever has an obvious conversion is converted (list to
tuple, int to float, "1e-3" to 0.001, ...), recursively through containers and
unions. If a declared type is understood but the value cannot be converted to
it, a `TypeError` naming the field (e.g. `mod.Model.pairs[1][0]`) is raised.
Values whose declared type we don't understand are passed through unchanged.
A field defaulting to `None` accepts `None` even if its hint is not `Optional`
(the `x: dict = None` idiom).

Nested dataclasses are not supported: `asdict` turns them into plain dicts
without a handle, and `from_dict` does not rebuild them.

"""

import dataclasses
import importlib
import numbers
import types
from collections.abc import Mapping, Sequence

import typing

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

    return construct(cls, inner, handle)


# -- helpers --


def construct(cls, inner, path):
    if not dataclasses.is_dataclass(cls):
        return cls(**inner)

    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {}

    defaults = {f.name: f.default for f in dataclasses.fields(cls)}

    kwargs = {}
    for key, value in inner.items():
        if key not in hints:
            kwargs[key] = value  # unknown or unresolved: let cls deal with it
        elif value is None and defaults.get(key, dataclasses.MISSING) is None:
            kwargs[key] = None  # `x: dict = None` idiom
        else:
            kwargs[key] = coerce(value, hints[key], f"{path}.{key}")

    return cls(**kwargs)


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


# -- coercion machinery --


def coerce(value, hint, path="value"):
    """Convert value to the type hint, raising TypeError if that is not possible.

    Understood: Any, object, None, unions, bool, int, float, str, tuple, list,
    Sequence, and Mapping (incl. subclasses like FrozenDict). Anything else is
    passed through unchanged.
    """
    if hint is typing.Any or hint is object:
        return value

    if hint is None or hint is type(None):
        if value is None:
            return None
        _fail(value, hint, path)

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin is typing.Union or origin is types.UnionType:
        return _coerce_union(value, args, path)

    cls = origin if origin is not None else hint
    if not isinstance(cls, type):
        return value  # e.g. callable, Literal, TypeVar, unresolved strings

    if cls in _SCALARS:
        return _SCALARS[cls](value, hint, path)

    if cls in (tuple, list) or cls is Sequence:
        return _coerce_sequence(value, cls, args, hint, path)

    if issubclass(cls, Mapping):
        return _coerce_mapping(value, cls, args, hint, path)

    return value


def _to_bool(value, hint, path):
    if isinstance(value, bool):
        return value
    _fail(value, hint, path)


def _to_int(value, hint, path):
    match value:
        case bool():
            pass
        case numbers.Integral():
            return int(value)
        case numbers.Real() if float(value).is_integer():
            return int(value)
        case str():
            return _to_int(_parse_float(value, hint, path), hint, path)
    _fail(value, hint, path)


def _to_float(value, hint, path):
    match value:
        case bool():
            pass
        case numbers.Real():
            return float(value)
        case str():
            return _parse_float(value, hint, path)
    _fail(value, hint, path)


def _to_str(value, hint, path):
    match value:
        case str():
            return value
        case bool():
            pass
        case numbers.Real():
            return str(value)
    _fail(value, hint, path)


_SCALARS = {bool: _to_bool, int: _to_int, float: _to_float, str: _to_str}


def _coerce_sequence(value, cls, args, hint, path):
    if not isinstance(value, (tuple, list)):
        _fail(value, hint, path)

    if cls is tuple and args and args[-1] is not Ellipsis:
        if len(value) != len(args):
            _fail(value, hint, path)
        items = [coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args))]
    elif args:
        items = [coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value)]
    else:
        items = value

    return list(items) if cls is list else tuple(items)


def _coerce_mapping(value, cls, args, hint, path):
    if not isinstance(value, Mapping):
        _fail(value, hint, path)

    if args:
        k_hint, v_hint = args
        items = {
            coerce(k, k_hint, f"{path}.<key>"): coerce(v, v_hint, f"{path}.{k}")
            for k, v in value.items()
        }
    else:
        items = dict(value)

    if cls in (dict, Mapping):
        return items
    return cls(items)


def _coerce_union(value, args, path):
    # prefer a member the value already is, so `int | str` keeps "5" as a str
    for arg in args:
        cls = typing.get_origin(arg) or arg
        if cls is type(None) or arg is None:
            if value is None:
                return None
        elif isinstance(cls, type) and isinstance(value, cls):
            if not (cls is int and isinstance(value, bool)):
                return coerce(value, arg, path)

    for arg in args:
        try:
            return coerce(value, arg, path)
        except TypeError:
            pass

    _fail(value, " | ".join(map(str, args)), path)


def _parse_float(value, hint, path):
    try:
        return float(value)
    except ValueError:
        _fail(value, hint, path)


def _fail(value, hint, path):
    raise TypeError(f"{path}: cannot convert {value!r} to {hint}")
