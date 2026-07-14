#!/usr/bin/env python3
"""Weighted flow accumulation over an 8-neighbour flow-fraction grid.

Created on Tue Jul 14 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Companion module: 03_flow_router.py (MDinf accumulation) and
04_flow_accumulation.py (every method) both import this, so the two
stages accumulate with the *identical* algorithm - a stream network
thresholded in stage 3 and an accumulation raster written in stage 4
can never drift apart. It lives in its own file because module names
starting with a digit cannot be imported.

THE ALGORITHM
    Accumulation is the classic upstream-area recurrence
    ``A(c) = area(c) + sum_over_donors( f(donor -> c) * A(donor) )``
    (Mark, 1988), evaluated in topological order over the weighted flow
    graph using Kahn's (1962) queue algorithm, JIT-compiled with Numba.
    Each cell is visited exactly once, so the run time is O(n cells).
    Flow directed at NoData cells or off the grid edge leaves the
    domain. Cells caught in a directed cycle (should not occur in a
    well-formed flow-direction raster) are left with partial
    accumulation and returned in the unresolved count.

INPUT CONVENTION
    ``frac`` holds, for each cell, the fraction of its outflow sent to
    each of its 8 neighbours in the canonical order N, NE, E, SE, S,
    SW, W, NW - the band order of the MFD / MD-infinity rasters and of
    the ``DROW`` / ``DCOL`` offsets below. A cell whose 8 fractions are
    all zero is a sink.

REFERENCES (Harvard style, as in the stage scripts)
    Kahn, A.B. (1962) 'Topological sorting of large networks',
    Communications of the ACM, 5(11), pp. 558-562.
    Mark, D.M. (1988) 'Network models in geomorphology', in Anderson,
    M.G. (ed.) Modelling Geomorphological Systems. Chichester: Wiley,
    pp. 73-97.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Canonical neighbour order used throughout: N, NE, E, SE, S, SW, W, NW.
DROW = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
DCOL = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


@njit(cache=True)
def accumulate(frac, valid, cell_area, drow, dcol):
    """Weighted flow accumulation in topological order (Kahn's algorithm).

    frac  : float32 (rows, cols, 8) outflow fraction per neighbour
    valid : bool    (rows, cols)    data mask
    Returns (accumulation float64 (rows, cols), number of unprocessed cells).
    """
    rows, cols, _ = frac.shape
    acc = np.zeros((rows, cols), dtype=np.float64)
    indeg = np.zeros((rows, cols), dtype=np.int32)

    # In-degree = number of valid donors draining into each cell.
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            acc[r, c] = cell_area
            for k in range(8):
                if frac[r, c, k] > 0.0:
                    nr = r + drow[k]
                    nc = c + dcol[k]
                    if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc]:
                        indeg[nr, nc] += 1

    # Seed the queue with cells that receive no inflow (local maxima).
    queue = np.empty(rows * cols, dtype=np.int64)
    tail = 0
    for r in range(rows):
        for c in range(cols):
            if valid[r, c] and indeg[r, c] == 0:
                queue[tail] = r * cols + c
                tail += 1

    # Pop cells whose upstream area is complete and push it downstream.
    head = 0
    processed = 0
    while head < tail:
        idx = queue[head]
        head += 1
        processed += 1
        r = idx // cols
        c = idx % cols
        a = acc[r, c]
        for k in range(8):
            f = frac[r, c, k]
            if f > 0.0:
                nr = r + drow[k]
                nc = c + dcol[k]
                if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc]:
                    acc[nr, nc] += f * a
                    indeg[nr, nc] -= 1
                    if indeg[nr, nc] == 0:
                        queue[tail] = nr * cols + nc
                        tail += 1

    n_valid = 0
    for r in range(rows):
        for c in range(cols):
            if valid[r, c]:
                n_valid += 1
    return acc, n_valid - processed
