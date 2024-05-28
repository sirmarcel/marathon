import numpy as np


def s_to_string(s, precision="ms"):
    if s > 5 * 60 and precision != "s":  # for >5m, don't report ms
        precision = "s"
    if s > 60 * 60 and precision != "m":  # for >1h, don't report s/ms
        precision = "m"

    if s <= 1 and precision != "ms":
        precision = "ms"

    if s < 60 and precision != "s":
        precision = "s"

    divisor = {
        "ms": 1,
        "s": 1000,
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
    }[precision]

    ms = 1000 * s
    ms = np.round(ms / divisor) * divisor

    s, ms = np.divmod(ms, 1000)
    m, s = np.divmod(s, 60)
    h, m = np.divmod(m, 60)

    ms = int(ms)
    s = int(s)
    m = int(m)
    h = int(h)

    string = ""
    for key, value in {"h": h, "m": m, "s": s, "ms": ms}.items():
        if value != 0:
            string += f"{value}{key}"

    return string


# -- test --

h = 1
m = 2
s = 35
ms = 123

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h, "m") == "1h3m"


h = 0
m = 6
s = 35
ms = 123

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h) == "6m35s"

h = 1
m = 6
s = 35
ms = 123

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h) == "1h7m"
