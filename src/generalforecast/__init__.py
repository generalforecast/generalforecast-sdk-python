# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""General Forecast: the Python SDK for structured data foundation models.

`gf.contexts`, `gf.experiments`, `gf.predictors`, `gf.snapshots`,
`gf.runs`, `gf.sources`, `gf.keys` and `gf.models` are the resources of a
default `Client`, built on first use; `gf.login()` logs the process in.
`gf.Client` and `gf.AsyncClient` make clients of your own. Everything loads
on first use, so `import generalforecast` stays cheap.
"""

import importlib
from typing import TYPE_CHECKING, Any

from generalforecast.errors import (
    AmbiguousReference,
    CapacityExceeded,
    Conflict,
    Error,
    Internal,
    InvalidInput,
    ModelNotReady,
    NotFound,
    PaymentRequired,
    Timeout,
    Unauthorized,
    Unavailable,
    Unsupported,
)

if TYPE_CHECKING:
    from generalforecast import errors as errors
    from generalforecast import types as types
    from generalforecast.client import AsyncClient as AsyncClient
    from generalforecast.client import Client as Client
    from generalforecast.client import login as login
    from generalforecast.client import logout as logout
    from generalforecast.client import whoami as whoami
    from generalforecast.estimators import Classifier as Classifier
    from generalforecast.estimators import Forecaster as Forecaster
    from generalforecast.estimators import Regressor as Regressor
    from generalforecast.resources import (
        Contexts,
        Experiments,
        Keys,
        Models,
        Predictors,
        Runs,
        Snapshots,
        Sources,
    )
    from generalforecast.types import STANDARD_PERMISSIONS as STANDARD_PERMISSIONS
    from generalforecast.types import Holdout as Holdout
    from generalforecast.types import Predictor as Predictor
    from generalforecast.types import Selection as Selection

    __version__: str
    contexts: Contexts
    experiments: Experiments
    predictors: Predictors
    sources: Sources
    keys: Keys
    models: Models
    snapshots: Snapshots
    runs: Runs

__all__ = [
    "STANDARD_PERMISSIONS",
    "AmbiguousReference",
    "AsyncClient",
    "CapacityExceeded",
    "Classifier",
    "Client",
    "Conflict",
    "Error",
    "Forecaster",
    "Holdout",
    "Internal",
    "InvalidInput",
    "ModelNotReady",
    "NotFound",
    "PaymentRequired",
    "Predictor",
    "Regressor",
    "Selection",
    "Timeout",
    "Unauthorized",
    "Unavailable",
    "Unsupported",
    "__version__",
    "contexts",
    "errors",
    "experiments",
    "keys",
    "login",
    "logout",
    "models",
    "predictors",
    "runs",
    "snapshots",
    "sources",
    "types",
    "whoami",
]

_EXPORTS = {
    "Client": ("generalforecast.client", "Client"),
    "AsyncClient": ("generalforecast.client", "AsyncClient"),
    "login": ("generalforecast.client", "login"),
    "logout": ("generalforecast.client", "logout"),
    "whoami": ("generalforecast.client", "whoami"),
    "Classifier": ("generalforecast.estimators", "Classifier"),
    "Regressor": ("generalforecast.estimators", "Regressor"),
    "Forecaster": ("generalforecast.estimators", "Forecaster"),
    "Predictor": ("generalforecast.types", "Predictor"),
    "Holdout": ("generalforecast.types", "Holdout"),
    "Selection": ("generalforecast.types", "Selection"),
    "STANDARD_PERMISSIONS": ("generalforecast.types", "STANDARD_PERMISSIONS"),
    "errors": ("generalforecast.errors", None),
    "types": ("generalforecast.types", None),
}

# The default client's resources; looked up on each use, so a new login takes effect.
_RESOURCES = frozenset(
    {
        "contexts",
        "experiments",
        "predictors",
        "snapshots",
        "runs",
        "sources",
        "keys",
        "models",
    }
)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("generalforecast")
    except PackageNotFoundError:
        return "0.0.0"


def __getattr__(name: str) -> Any:
    """Imports an export, reads the version, or reaches the default client, on first use."""
    if name in _RESOURCES:
        from generalforecast.client import default

        return getattr(default(), name)
    if name == "__version__":
        value = globals()[name] = _version()
        return value
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = importlib.import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Lists the public exports."""
    return sorted(__all__)
