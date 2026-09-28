"""Heterogeneous BatchTopK crosscoders. Attributes are imported lazily."""

from importlib import import_module

_EXPORTS = {
    "CrossCoder": (".dictionary", "CrossCoder"),
    "BatchTopKCrossCoder": (".dictionary", "BatchTopKCrossCoder"),
    "HeterogeneousCrossCoder": (".dictionary", "HeterogeneousCrossCoder"),
    "BatchTopKHeterogeneousCrossCoder": (".dictionary", "BatchTopKHeterogeneousCrossCoder"),
    "ActivationCache": (".cache", "ActivationCache"),
    "HeterogeneousActivationCacheTuple": (".cache", "HeterogeneousActivationCacheTuple"),
    "HeterogeneousActivationBatchDataset": (".cache", "HeterogeneousActivationBatchDataset"),
    "IncrementalActivationShardWriter": (".cache", "IncrementalActivationShardWriter"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *__all__))
