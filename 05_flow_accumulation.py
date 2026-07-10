#!/usr/bin/env python3
"""Flow accumulation from flow-direction rasters (D8, MFD, D-infinity, MD-infinity).

Created on Tue Jul 7 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 5, optional (see README.md):
    reads   data/03_flows/flow_direction_*.tif   (03_flow_router.py output)
    writes  data/05_accumulation/flow_accumulation_<method>.tif

Reads any of the flow-direction rasters in the input folder and computes the
corresponding flow-accumulation raster into the output folder.

How it works
------------
Every supported flow-direction format is first converted to a common
representation: for each cell, the *fraction* of its outflow sent to each of
its 8 neighbours (band order N, NE, E, SE, S, SW, W, NW):

* **D8** (O'Callaghan & Mark, 1984): the single coded receiver
  (64=N 128=NE 1=E 2=SE 4=S 8=SW 16=W 32=NW) gets fraction 1.0.
  Flats (-1) and pits (-2) have no outflow and act as sinks.
* **MFD** (Quinn et al., 1991): the 8 input bands *are* the fractions.
* **D-infinity** (Tarboton, 1997): the flow angle theta (radians,
  counter-clockwise from east) is split between the two facet-adjacent
  neighbours: with s = theta / (pi/4), the fractions are (1 - frac(s)) to
  direction floor(s) and frac(s) to direction floor(s)+1 (angle order
  E, NE, N, NW, W, SW, S, SE). Flats (-1) and pits (-2) are sinks.
* **MD-infinity** (Seibert & McGlynn, 2007): the 8 input bands *are* the
  fractions.

Accumulation itself is the classic upstream-area recurrence
``A(c) = area(c) + sum_over_donors( f(donor -> c) * A(donor) )``
(Mark, 1988), evaluated in topological order over the weighted flow graph
using Kahn's (1962) queue algorithm, JIT-compiled with Numba. Each cell is
visited exactly once, so the run time is O(n cells). Flow directed at NoData
cells or off the grid edge leaves the domain. Cells caught in a directed
cycle (should not occur in a well-formed flow-direction raster) are reported
and left with partial accumulation.

Output
------
``outputs/flow_accumulation_<method>.tif`` - float32, deflate-compressed
GeoTIFF. Values are the upslope contributing area **including the cell
itself**, either in square metres (default) or in cell counts (``--units
cells``). NoData is NaN.

Spatial reference
-----------------
* Horizontal: EPSG:3067 (ETRS89 / TM35FIN), units metres, 2 m cells.
* Vertical datum of the source elevation data: N2000, units metres.
  (Flow accumulation itself carries no height values; the datum is recorded
  in the output metadata for provenance.)

Credits
-------
* Source data: flow-direction rasters in ``inputs/``, derived from a 2 m
  digital elevation model in EPSG:3067 / N2000 - presumed to be the
  National Land Survey of Finland (Maanmittauslaitos) 2 m elevation model
  (KM2), licensed CC BY 4.0. Edit ``SOURCE_DATA_CREDIT`` below if the
  provenance differs.
* Flow-direction algorithm authors:
  O'Callaghan & Mark (1984) [D8]; Quinn, Beven, Chevallier & Planchon
  (1991) [MFD]; Tarboton (1997) [D-infinity]; Seibert & McGlynn (2007)
  [MD-infinity]. Full references in ``METHODS`` below.
* Accumulation strategy: Mark (1988) upstream-area recurrence; Kahn (1962)
  topological ordering.
* Tools that enabled this work: Python, NumPy (Harris et al., 2020),
  Numba (Lam, Pitrou & Seibert, 2015), rasterio (Gillies et al.) on
  GDAL (GDAL/OGR contributors, OSGeo).

Usage
-----
    python 05_flow_accumulation.py                      # all rasters in inputs/
    python 05_flow_accumulation.py inputs/flow_direction_d8.tif [more ...]
    python 05_flow_accumulation.py --units cells        # counts instead of m2

Run inside the ``water`` conda environment:
    conda activate water
    python 05_flow_accumulation.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from numba import njit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical neighbour order used throughout: N, NE, E, SE, S, SW, W, NW
# (matches the band order of the MFD / MD-infinity input rasters).
NEIGHBOUR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DROW = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
DCOL = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)

# D8 receiver codes in canonical neighbour order.
D8_CODES = (64, 128, 1, 2, 4, 8, 16, 32)

# D-infinity: angle sectors are numbered counter-clockwise from east
# (0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE); this maps a sector number
# to the canonical neighbour index above.
ANGLE_SECTOR_TO_NEIGHBOUR = np.array([2, 1, 0, 7, 6, 5, 4, 3], dtype=np.int64)

SOURCE_DATA_CREDIT = (
    "Flow-direction rasters derived from a 2 m DEM in EPSG:3067 (ETRS89 / "
    "TM35FIN), vertical datum N2000; presumed source: National Land Survey "
    "of Finland 2 m elevation model (KM2), CC BY 4.0."
)

TOOL_CREDITS = (
    "Python, NumPy (Harris et al. 2020, doi:10.1038/s41586-020-2649-2), "
    "Numba (Lam, Pitrou & Seibert 2015, doi:10.1145/2833157.2833162), "
    "rasterio (Gillies et al.), GDAL (GDAL/OGR contributors, OSGeo)."
)

ACCUMULATION_METHOD = (
    "Upslope contributing-area recurrence A(c) = area(c) + "
    "sum(f(d->c) * A(d)) over donor cells d (Mark 1988), evaluated in "
    "topological order with Kahn's (1962) queue algorithm; single O(n) pass "
    "over the weighted 8-neighbour flow graph. Flow to NoData or off-grid "
    "is lost from the domain; flats and pits act as sinks."
)


@dataclass(frozen=True)
class Method:
    key: str
    name: str
    citation: str


METHODS = {
    "d8": Method(
        "d8",
        "D8 (single flow direction)",
        "O'Callaghan, J.F. & Mark, D.M. (1984). The extraction of drainage "
        "networks from digital elevation data. Computer Vision, Graphics, "
        "and Image Processing 28(3), 323-344.",
    ),
    "mfd": Method(
        "mfd",
        "MFD (multiple flow direction)",
        "Quinn, P., Beven, K., Chevallier, P. & Planchon, O. (1991). The "
        "prediction of hillslope flow paths for distributed hydrological "
        "modelling using digital terrain models. Hydrological Processes "
        "5(1), 59-79.",
    ),
    "dinf": Method(
        "dinf",
        "D-infinity (single-direction angle, two-neighbour split)",
        "Tarboton, D.G. (1997). A new method for the determination of flow "
        "directions and upslope areas in grid digital elevation models. "
        "Water Resources Research 33(2), 309-319.",
    ),
    "mdinf": Method(
        "mdinf",
        "MD-infinity (triangular multiple flow direction)",
        "Seibert, J. & McGlynn, B.L. (2007). A new triangular multiple flow "
        "direction algorithm for computing upslope areas from gridded "
        "digital elevation models. Water Resources Research 43, W04501.",
    ),
}


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _accumulate(frac, valid, cell_area, drow, dcol):
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


# ---------------------------------------------------------------------------
# Format -> outflow-fraction conversion
# ---------------------------------------------------------------------------

def _fractions_from_d8(src) -> tuple[np.ndarray, np.ndarray]:
    codes = src.read(1)
    nodata = src.nodata
    valid = np.ones(codes.shape, dtype=bool) if nodata is None else codes != int(nodata)
    frac = np.zeros(codes.shape + (8,), dtype=np.float32)
    for k, code in enumerate(D8_CODES):
        frac[..., k][codes == code] = 1.0
    # -1 (flat) and -2 (pit) keep all-zero fractions: sinks.
    return frac, valid


def _fractions_from_dinf(src) -> tuple[np.ndarray, np.ndarray]:
    angle = src.read(1)
    valid = ~np.isnan(angle)
    frac = np.zeros(angle.shape + (8,), dtype=np.float32)
    flowing = valid & (angle >= 0.0)  # -1 flat / -2 pit -> sinks
    theta = np.mod(angle, 2.0 * np.pi, where=flowing, out=np.zeros_like(angle))
    s = theta / (np.pi / 4.0)
    sector = np.floor(s).astype(np.int64) % 8
    w_next = (s - np.floor(s)).astype(np.float32)
    for sec in range(8):
        m = flowing & (sector == sec)
        if not m.any():
            continue
        k1 = ANGLE_SECTOR_TO_NEIGHBOUR[sec]
        k2 = ANGLE_SECTOR_TO_NEIGHBOUR[(sec + 1) % 8]
        frac[..., k1][m] += 1.0 - w_next[m]
        frac[..., k2][m] += w_next[m]
    return frac, valid


def _fractions_from_bands(src) -> tuple[np.ndarray, np.ndarray]:
    """MFD / MD-infinity: 8 bands of fractions in N..NW order."""
    if src.count != 8:
        raise ValueError(f"expected 8 bands of flow fractions, found {src.count}")
    frac = np.empty((src.height, src.width, 8), dtype=np.float32)
    for b in range(8):
        frac[..., b] = src.read(b + 1)
    # A NaN band means "no flow to that neighbour"; a cell is NoData only
    # when all 8 bands are NaN.
    valid = ~np.isnan(frac).all(axis=2)
    np.nan_to_num(frac, copy=False, nan=0.0)
    np.clip(frac, 0.0, None, out=frac)
    return frac, valid


CONVERTERS = {
    "d8": _fractions_from_d8,
    "dinf": _fractions_from_dinf,
    "mfd": _fractions_from_bands,
    "mdinf": _fractions_from_bands,
}


def detect_method(path: Path, tags: dict) -> Method:
    """Identify the flow-direction format from GeoTIFF tags or the filename."""
    tag = tags.get("algorithm", "").lower().replace("-", "").replace("_", "")
    for key in ("mdinf", "dinf", "mfd", "d8"):  # longest match first
        if tag == key:
            return METHODS[key]
    stem = path.stem.lower()
    for key in ("mdinf", "dinf", "mfd", "d8"):
        if key in stem.split("_"):
            return METHODS[key]
    raise ValueError(
        f"cannot identify flow-direction format of {path.name}; expected an "
        "'algorithm' tag or a filename containing d8/mfd/dinf/mdinf"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process(path: Path, out_dir: Path, units: str) -> Path:
    t0 = time.perf_counter()
    with rasterio.open(path) as src:
        method = detect_method(path, src.tags())
        print(f"\n{path.name}: {method.name}")
        if src.crs is None or src.crs.to_epsg() != 3067:
            print(f"  WARNING: expected EPSG:3067, raster reports {src.crs}")
        transform = src.transform
        cell_area_m2 = abs(transform.a * transform.e)
        source_tags = src.tags()
        profile = src.profile
        frac, valid = CONVERTERS[method.key](src)

    print(f"  grid {frac.shape[1]} x {frac.shape[0]} cells, "
          f"cell {abs(transform.a)} x {abs(transform.e)} m, "
          f"{int(valid.sum())} valid cells")

    cell_value = cell_area_m2 if units == "m2" else 1.0
    acc, unresolved = _accumulate(frac, valid, cell_value, DROW, DCOL)
    del frac
    if unresolved:
        print(f"  WARNING: {unresolved} cells form directed cycles; "
              "their accumulation is incomplete")

    out = acc.astype(np.float32)
    out[~valid] = np.nan
    unit_label = "m2 (upslope contributing area incl. the cell itself)" \
        if units == "m2" else "cells (upslope cell count incl. the cell itself)"
    print(f"  max accumulation: {np.nanmax(out):,.0f} "
          f"{'m2' if units == 'm2' else 'cells'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"flow_accumulation_{method.key}.tif"
    profile.update(
        count=1, dtype="float32", nodata=float("nan"),
        compress="deflate", predictor=3, tiled=True,
        blockxsize=512, blockysize=512, bigtiff="if_safer",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.update_tags(
            title=f"Flow accumulation ({method.name})",
            units=unit_label,
            flow_direction_algorithm=method.name,
            flow_direction_citation=method.citation,
            accumulation_method=ACCUMULATION_METHOD,
            source_flow_direction_raster=path.name,
            source_flow_direction_encoding=source_tags.get("encoding", ""),
            source_data_credit=SOURCE_DATA_CREDIT,
            horizontal_crs="EPSG:3067 (ETRS89 / TM35FIN), units metres",
            vertical_datum="N2000, units metres (datum of the source DEM)",
            software_credits=TOOL_CREDITS,
            generated_by="05_flow_accumulation.py",
        )
    print(f"  -> {out_path}  ({time.perf_counter() - t0:.1f} s)")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute flow accumulation from D8 / MFD / D-infinity / "
                    "MD-infinity flow-direction rasters.")
    parser.add_argument(
        "rasters", nargs="*", type=Path,
        help="flow-direction GeoTIFFs (default: all inputs/flow_direction_*.tif)")
    parser.add_argument(
        "--inputs-dir", type=Path,
        default=Path(__file__).parent / "data" / "03_flows")
    parser.add_argument(
        "--outputs-dir", type=Path,
        default=Path(__file__).parent / "data" / "05_accumulation")
    parser.add_argument(
        "--units", choices=("m2", "cells"), default="m2",
        help="output as contributing area in m2 (default) or as cell counts")
    args = parser.parse_args(argv)

    rasters = args.rasters or sorted(args.inputs_dir.glob("flow_direction_*.tif"))
    if not rasters:
        parser.error(f"no flow-direction rasters found in {args.inputs_dir}")

    for path in rasters:
        process(path, args.outputs_dir, args.units)
    return 0


if __name__ == "__main__":
    sys.exit(main())
