# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""General Forecast: the Python SDK and `gf` CLI for the hosted forecasting service.

Exports `Predictor`, the estimators `Classifier`, `Regressor` and
`Forecaster`, the hosted `Client` and `AsyncClient`, the error classes, and
the `errors` submodule. Every prediction runs on the service. Everything
loads on first use, so `import generalforecast` stays cheap.
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
    Unauthorized,
    Unavailable,
    Unsupported,
)

if TYPE_CHECKING:
    from generalforecast import errors as errors
    from generalforecast.async_client import AsyncClient as AsyncClient
    from generalforecast.client import Client as Client
    from generalforecast.estimators import Classifier as Classifier
    from generalforecast.estimators import Forecaster as Forecaster
    from generalforecast.estimators import Regressor as Regressor
    from generalforecast.predictor import Predictor as Predictor

    __version__: str

__all__ = [
    "AmbiguousReference",
    "AsyncClient",
    "CapacityExceeded",
    "Classifier",
    "Client",
    "Conflict",
    "Error",
    "Forecaster",
    "Internal",
    "InvalidInput",
    "ModelNotReady",
    "NotFound",
    "PaymentRequired",
    "Predictor",
    "Regressor",
    "Unauthorized",
    "Unavailable",
    "Unsupported",
    "__version__",
    "errors",
]

_EXPORTS = {
    "Predictor": ("generalforecast.predictor", "Predictor"),
    "Client": ("generalforecast.client", "Client"),
    "AsyncClient": ("generalforecast.async_client", "AsyncClient"),
    "Classifier": ("generalforecast.estimators", "Classifier"),
    "Regressor": ("generalforecast.estimators", "Regressor"),
    "Forecaster": ("generalforecast.estimators", "Forecaster"),
    "errors": ("generalforecast.errors", None),
}


def _version() -> str:
    """The installed version; reading the package metadata is itself an import."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("generalforecast")
    except PackageNotFoundError:  # A checkout imported without being installed.
        return "0.0.0"


def __getattr__(name: str) -> Any:
    """Imports an export, or reads the version, on first use."""
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
