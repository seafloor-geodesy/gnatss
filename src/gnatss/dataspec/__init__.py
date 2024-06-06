from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from .utils import _get_fields
from .v1 import DataV1


class _Data:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DataSpec:
    __data_version: ClassVar[dict[str, BaseModel]] = {
        "v1": DataV1,
    }

    def __init__(self, version: Literal["v1"] = "v1") -> None:
        self._version = version
        self._data = self.__data_version.get(version)

        self._model_fields = self._data.model_fields
        self._setup_private_attributes()

        self.tx_fields = _get_fields(self._model_fields, self.tx)
        self.rx_fields = _get_fields(self._model_fields, self.rx)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._version})"

    def _setup_private_attributes(self):
        for k, v in self._data.__private_attributes__.items():
            if v.default is PydanticUndefined:
                msg: str = f"{k} must be defined within data model version {self._version}"
                raise AttributeError(msg)
            setattr(self, k.strip("_"), v.default)

    def _get_gnss_fields(self, fields):
        return {k: v for k, v in fields.items() if k.startswith(self.gnss)}

    def _get_platform_fields(self, fields, data_type):
        return {
            k: v
            for k, v in fields.items()
            if not k.startswith(self.gnss) and not k.endswith(f"_{data_type}")
        }

    def _get_gnss_cov_fields(self, fields, diagonal=False):
        cov_fields = {k: v for k, v in fields.items() if self.covariance in k}
        if diagonal:
            diag_fields = {}
            for f in ["xx", "yy", "zz"]:
                name = next(filter(lambda k: f in k.lower(), cov_fields.keys()))
                diag_fields[name] = cov_fields[name]
            return diag_fields
        return cov_fields

    @staticmethod
    def _get_time_field(fields, time_key):
        return {time_key: fields[time_key]}

    @property
    def version(self):
        return self._version

    @property
    def data(self):
        return _Data(**self._model_fields)

    @property
    def gnss_tx_fields(self):
        return self._get_gnss_fields(self.tx_fields)

    @property
    def gnss_rx_fields(self):
        return self._get_gnss_fields(self.rx_fields)

    @property
    def platform_tx_fields(self):
        return self._get_platform_fields(self.tx_fields, self.tx)

    @property
    def platform_rx_fields(self):
        return self._get_platform_fields(self.rx_fields, self.rx)

    @property
    def transducer_tx_fields(self):
        return {
            k: v
            for k, v in self.tx_fields.items()
            if k.endswith(f"_{self.tx}") and k != self.tx_time
        }

    @property
    def transducer_rx_fields(self):
        return {
            k: v
            for k, v in self.rx_fields.items()
            if k.endswith(f"_{self.rx}") and k != self.rx_time
        }

    @property
    def gnss_rx_cov_fields(self):
        return self._get_gnss_cov_fields(self.gnss_rx_fields)

    @property
    def gnss_tx_cov_fields(self):
        return self._get_gnss_cov_fields(self.gnss_tx_fields)

    @property
    def gnss_rx_diag_cov_fields(self):
        return self._get_gnss_cov_fields(self.gnss_rx_fields, diagonal=True)

    @property
    def gnss_tx_diag_cov_fields(self):
        return self._get_gnss_cov_fields(self.gnss_tx_fields, diagonal=True)

    @property
    def tx_time_field(self):
        return self._get_time_field(self.tx_fields, self.tx_time)

    @property
    def rx_time_field(self):
        return self._get_time_field(self.rx_fields, self.rx_time)
