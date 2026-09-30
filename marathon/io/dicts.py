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

When the class is a dataclass, `from_dict` coerces the payload into the types
declared by its fields (`coerce`). Serialising to `.yaml` loses type information
(tuples come back as lists, `FrozenDict` as `dict`), so this restores what the
class declares. Values that cannot be converted to a declared type raise;
declarations we don't understand are passed through.

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

    return _construct(cls, inner, handle)


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

    if cls is bool:
        if isinstance(value, bool):
            return value
        _fail(value, hint, path)

    if cls is int:
        if isinstance(value, bool):
            _fail(value, hint, path)
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, str):
            value = _parse_float(value, hint, path)
        if isinstance(value, numbers.Real) and float(value).is_integer():
            return int(value)
        _fail(value, hint, path)

    if cls is float:
        if isinstance(value, bool):
            _fail(value, hint, path)
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, str):
            return _parse_float(value, hint, path)
        _fail(value, hint, path)

    if cls is str:
        if isinstance(value, str):
            return value
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return str(value)
        _fail(value, hint, path)

    if cls in (tuple, list) or cls is Sequence:
        if not isinstance(value, (tuple, list)):
            _fail(value, hint, path)

        if cls is tuple and args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                _fail(value, hint, path)
            items = [
                coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args))
            ]
        elif args:
            items = [coerce(v, args[0], f"{path}[{i}]") for i, v in enumerate(value)]
        else:
            items = value

        return list(items) if cls is list else tuple(items)

    if issubclass(cls, Mapping):
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

    return value


# -- helpers --


def _construct(cls, inner, path):
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
