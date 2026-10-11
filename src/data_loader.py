from __future__ import annotations

import re
from io import BytesIO
from typing import BinaryIO

import pandas as pd

from config import CANONICAL_FIELDS, FIELD_ALIASES


def _normalise_name(value: object) -> str:
    """Convert a column name into a normalised format."""
    return (
        re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
        .strip("_")
    )


def load_transport_file(
    source: bytes | BinaryIO,
    filename: str,
) -> pd.DataFrame:
    """
    Load a user-uploaded CSV or Parquet file.

    No dataset is loaded automatically. Data is loaded only when
    the user explicitly uploads a supported file.
    """
    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if extension not in {"csv", "parquet", "pq"}:
        raise ValueError(
            "Unsupported file format. Please upload a CSV or Parquet file."
        )

    stream = BytesIO(source) if isinstance(source, bytes) else source

    try:
        if extension == "csv":
            frame = pd.read_csv(stream, low_memory=False)
        else:
            frame = pd.read_parquet(stream)

    except Exception as exc:
        raise ValueError(
            f"Unable to read the uploaded {extension.upper()} file. "
            "Check that the file is valid and not corrupted."
        ) from exc

    if frame.empty:
        raise ValueError(
            "The uploaded file contains no data rows."
        )

    if len(frame.columns) == 0:
        raise ValueError(
            "The uploaded file does not contain any columns."
        )

    if not frame.columns.is_unique:
        raise ValueError(
            "The uploaded file contains duplicate column names. "
            "Rename the duplicate columns and upload it again."
        )

    return frame


def suggest_column_mapping(
    columns: list[object],
) -> dict[str, str | None]:
    """
    Suggest mappings between uploaded columns and TransitIQ fields.
    """
    normalised = {
        _normalise_name(column): str(column)
        for column in columns
    }

    suggestions: dict[str, str | None] = {}

    for field in CANONICAL_FIELDS:
        aliases = FIELD_ALIASES.get(field, {field})

        suggestions[field] = next(
            (
                normalised[name]
                for name in aliases
                if name in normalised
            ),
            None,
        )

    return suggestions


def apply_column_mapping(
    frame: pd.DataFrame,
    mapping: dict[str, str | None],
) -> pd.DataFrame:
    """
    Rename uploaded columns to TransitIQ's canonical field names.
    """
    rename_map: dict[str, str] = {}
    assigned_targets: set[str] = set()
    assigned_sources: set[str] = set()

    for field, source in mapping.items():

        if field not in CANONICAL_FIELDS:
            raise ValueError(
                f"Unsupported target field: {field}"
            )

        if source is None or source == "":
            continue

        if source not in frame.columns:
            raise ValueError(
                f"Column not found in uploaded data: {source}"
            )

        if field in assigned_targets:
            raise ValueError(
                f"More than one source column was mapped to {field}."
            )

        if source in assigned_sources:
            raise ValueError(
                f"The source column {source} cannot be mapped more than once."
            )

        if field in frame.columns and source != field:
            raise ValueError(
                f"Cannot map {source} to {field}: "
                f"a {field} column already exists."
            )

        rename_map[source] = field
        assigned_targets.add(field)
        assigned_sources.add(source)

    return frame.rename(columns=rename_map).copy()