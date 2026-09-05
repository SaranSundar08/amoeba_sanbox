"""Compatibility imports for controller classes.

New code should import :class:`MPPI` from ``mppi`` and amoeba variants from
``amoeba_controllers``.  This module keeps older experiments working.
"""

from amoeba_controllers import (
    AmoebaHybrid, AmoebaLocal, AmoebaRepairHybrid, CONTROLLERS)
from mppi import MPPI

__all__ = ["MPPI", "AmoebaLocal", "AmoebaHybrid", "AmoebaRepairHybrid",
           "CONTROLLERS"]
