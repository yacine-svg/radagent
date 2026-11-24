"""Fusion maths — re-exported from contracts.py so A's evaluation and B's agents use the
same formula. B adds the fusion *node* in this file; do not redefine fuse() here."""
from contracts import (CONFLICT_HIGH, CONFLICT_LOW, UNCERTAIN_BAND,  # noqa: F401
                       detect_conflict, fuse, in_uncertain_band)
