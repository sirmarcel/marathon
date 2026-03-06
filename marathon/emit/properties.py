# Reporting configuration for emit module.
# Similar to data/properties.py and evaluate/properties.py

from marathon.evaluate.properties import DEFAULT_NORMALIZATION

DEFAULT_PROPERTIES = {
    "energy": {"report_unit": (1000, "meV"), "symbol": "E"},
    "forces": {"report_unit": (1000, "meV/Å"), "symbol": "F"},
    "stress": {"report_unit": (1000, "meV"), "symbol": "σ"},
}


def _resolve_base_key(key):
    # _per_structure keys inherit properties from their base key
    if key.endswith("_per_structure"):
        return key[: -len("_per_structure")]
    return key


def get_scale(key, properties=DEFAULT_PROPERTIES):
    base_key = _resolve_base_key(key)
    return properties.get(base_key, {}).get("report_unit", (1, ""))[0]


def get_base_unit(key, properties=DEFAULT_PROPERTIES):
    base_key = _resolve_base_key(key)
    return properties.get(base_key, {}).get("report_unit", (1, ""))[1]


def get_full_unit(key, properties=DEFAULT_PROPERTIES, normalization=DEFAULT_NORMALIZATION):
    unit = get_base_unit(key, properties)
    # _per_structure keys skip /atom suffix (they're unnormalized by definition)
    if not key.endswith("_per_structure") and normalization.get(key) == "atom":
        unit = unit + "/atom"
    return unit


def get_symbol(key, properties=DEFAULT_PROPERTIES):
    base_key = _resolve_base_key(key)
    return properties.get(base_key, {}).get("symbol", base_key)


# -- test --


def test_emit_properties():
    assert get_scale("energy") == 1000
    assert get_scale("forces") == 1000
    assert get_scale("stress") == 1000
    assert get_scale("unknown") == 1

    assert get_base_unit("energy") == "meV"
    assert get_base_unit("forces") == "meV/Å"
    assert get_base_unit("stress") == "meV"
    assert get_base_unit("unknown") == ""

    assert get_full_unit("energy") == "meV/atom"
    assert get_full_unit("forces") == "meV/Å"
    assert get_full_unit("stress") == "meV/atom"
    assert get_full_unit("energy", normalization={}) == "meV"

    assert get_symbol("energy") == "E"
    assert get_symbol("forces") == "F"
    assert get_symbol("stress") == "σ"
    assert get_symbol("unknown") == "unknown"

    # Custom property
    custom = {"custom": {"report_unit": (1, "units"), "symbol": "C"}}
    assert get_scale("custom", custom) == 1
    assert get_full_unit("custom", custom, {}) == "units"
    assert get_full_unit("custom", custom, {"custom": "atom"}) == "units/atom"
    assert get_symbol("custom", custom) == "C"

    # _per_structure keys inherit from base key
    assert get_scale("energy_per_structure") == 1000
    assert get_base_unit("energy_per_structure") == "meV"
    assert get_full_unit("energy_per_structure") == "meV"  # no /atom suffix
    assert get_symbol("energy_per_structure") == "E"

    assert get_scale("stress_per_structure") == 1000
    assert get_full_unit("stress_per_structure") == "meV"
    assert get_symbol("stress_per_structure") == "σ"


test_emit_properties()
