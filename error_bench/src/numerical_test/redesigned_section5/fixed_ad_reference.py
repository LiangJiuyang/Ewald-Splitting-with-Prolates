#!/usr/bin/env python3
"""Public import surface for the operator-matched fixed-AD reference.

The implementation lives in ``ad_operator_audit/ad_operator_reference.py``
so the derivation audit, shell diagnostics, and main validation share exactly
one source of operator definitions.
"""

from ad_operator_audit.ad_operator_reference import *  # noqa: F401,F403

