#!/usr/bin/env python3
"""Stable command-line entry point.

The implementation lives in focused modules; this wrapper also preserves the
``episode`` and ``draw_world`` imports used by older experiments.
"""

from cli import main
from simulation import episode
from visualization import draw_world

__all__ = ["main", "episode", "draw_world"]


if __name__ == "__main__":
    main()
