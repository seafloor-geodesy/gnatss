from __future__ import annotations

DATA_TYPES = {"transmit": "0", "receive": "1"}


def _get_fields(model_fields, data_type):
    filtered_fields = {}
    for name, field in model_fields.items():
        if data_type in name or DATA_TYPES[data_type] in name:
            filtered_fields[name] = field
    return filtered_fields


def _get_cov_diagonal(model_fields):
    diagonals = ["XX", "YY", "ZZ"]
    fields = {}
    for k, v in model_fields.items():
        if any(d in k for d in diagonals):
            fields[k] = v
    return fields
