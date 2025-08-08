from marathon.evaluate import get_metrics_fn as _get_metrics_fn


def get_metrics_fn(samples=None, stats=None, keys=["energy", "forces"], uq_keys=[]):
    assert stats is None

    main_metrics_fn = _get_metrics_fn(samples=samples, keys=keys)

    def metrics_fn(auxs):
        metrics = main_metrics_fn(auxs)

        for key in uq_keys:
            n = auxs[key + "_n"].sum()
            nll = auxs[key + "_nll"].sum()
            crps = auxs[key + "_crps"].sum()
            var = 1e6 * auxs[key + "_var"].sum()
            std = 1e3 * auxs[key + "_std"].sum()

            if key == "energy":
                metrics[key]["nll"] = nll / n
                metrics[key]["crps"] = crps / n
                metrics[key]["var"] = var / n
                metrics[key]["std"] = std / n

            elif key == "forces":
                metrics[key]["nll"] = nll / (n * 3)
                metrics[key]["crps"] = crps / (n * 3)
                metrics[key]["var"] = var / (n * 3)
                metrics[key]["std"] = std / (n * 3)

                nll = auxs[key + "_nll"].sum(axis=0)
                crps = auxs[key + "_crps"].sum(axis=0)

                metrics[key]["nll_per_component"] = nll / n
                metrics[key]["crps_per_component"] = crps / n

            elif key == "stress":
                metrics[key]["nll"] = nll / (n * 9)
                metrics[key]["crps"] = crps / (n * 9)
                metrics[key]["var"] = var / (n * 9)
                metrics[key]["std"] = std / (n * 9)

                nll = auxs[key + "_nll"].sum(axis=0)
                crps = auxs[key + "_crps"].sum(axis=0)

                metrics[key]["nll_per_component"] = nll / n
                metrics[key]["crps_per_component"] = crps / n

        return metrics

    return metrics_fn
