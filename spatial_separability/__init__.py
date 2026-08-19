"""Project-wide access to the precomputed spatial-separability p table."""

from .p_table import (
    P_FREQUENCIES_HZ,
    P_FREQUENCY_BIN_INDICES,
    P_SIGNED_DELTA_DEGREES,
    P_TABLE_VERSION,
    P_THETA_A_MODULO_DEGREES,
    load_p_table,
    lookup_p,
    validate_p_table_context,
)

__all__ = [
    "P_FREQUENCIES_HZ",
    "P_FREQUENCY_BIN_INDICES",
    "P_SIGNED_DELTA_DEGREES",
    "P_TABLE_VERSION",
    "P_THETA_A_MODULO_DEGREES",
    "load_p_table",
    "lookup_p",
    "validate_p_table_context",
]
