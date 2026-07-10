#!/usr/bin/env python3
"""Delineate catchments from flow-direction rasters.

Created on Tue Jul 7 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 4, optional (see README.md):
    reads   data/03_flows/flow_direction_*.tif   (03_flow_router.py output)
    writes  data/04_catchments/

Implements the DEM-based watershed delineation theory of:
    Rolim da Paz, A. (2025) "Digital Elevation Models for Environmental
    Studies", Chapter 8 - Watersheds. Springer.

Three delineation methods are available:

  Method 1 - Outlet coordinates (section 8.1)
      Delineates the catchment contributing to one or more outlet points
      given as projected coordinates. Outlets are snapped to the cell of
      maximum flow accumulation within --snap-dist (the "outlet pixel
      selection" problem, section 8.1.3). With several outlets, nested
      areas become incremental basins (Fig. 8.4).

  Method 2 - Confluence criterion (sections 8.7.1-8.7.2)
      Automatic subdivision into sub-basins, one per stream segment
      (reach between confluences). The stream network is defined by a
      minimum accumulated drainage area (--stream-threshold). Options:
        --min-area   merge sub-basins smaller than a minimum size into
                     their downstream neighbour (size restriction,
                     section 8.7.2, TerrSet-style agglomeration);
        --min-order  only stream segments of Strahler order >= N
                     (section 8.6) act as sub-basin outlets;
        --coords     restrict the subdivision to the basin of this outlet.

  Method 3 - Pfafstetter / Ottobasins (section 8.7.3)
      Recursive subdivision of the basin of --coords (default: the cell
      of maximum accumulation) into 9 coded elements per level: the 4
      largest tributaries get codes 2,4,6,8 (downstream to upstream) and
      the 5 interbasin reaches of the main stem get codes 1,3,5,7,9.
      --levels controls the recursion depth.

Supported flow-direction formats (auto-detected from GeoTIFF metadata):
  d8     single band, ESRI codes 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE
         (-1 = flat, -2 = pit, nodata = 0)
  dinf   single band, Tarboton flow angle in [0, 2*pi) CCW from east
         (-1 = flat, -2 = pit, NaN = nodata)
  mfd    8 bands = flow fraction to the N,NE,E,SE,S,SW,W,NW neighbour
  mdinf  same encoding as mfd

All formats are converted to a unified model: 8 per-neighbour flow-weight
planes. Flow accumulation and catchment membership honour the full
multiple-flow-direction partitioning: for dinf/mfd/mdinf the catchment of
an outlet is the set of cells sending at least --mfd-frac of their flow
to it (for d8 this reduces exactly to the classic binary membership).
Structural products (stream segments, Strahler order, sub-basin
partition, Pfafstetter stems) use the dominant flow direction so that
every cell belongs to exactly one sub-basin.

Examples:
  python 04_delineate_catchments.py --method 1 --coords 385000,6672000
  python 04_delineate_catchments.py --method 1 --coords 385000,6672000 --coords 380000,6670000 --input flow_direction_mfd.tif
  python 04_delineate_catchments.py --method 2 --stream-threshold 1.0 --min-area 5.0
  python 04_delineate_catchments.py --method 2 --min-order 2
  python 04_delineate_catchments.py --method 3 --levels 2
  python 04_delineate_catchments.py --method 1 --coords 385000,6672000 --save-acc --save-streams

Outputs (per input raster) are written to data/04_catchments:
  catchments_<name>_method<M>.tif    int32 label raster (0 = outside)
  catchments_<name>_method<M>.gpkg   layers: 'catchments' (polygons with
                                     area, perimeter, drainage density,
                                     order/code) and 'streams' (segments
                                     with Strahler order)
  acc_<name>.tif                     flow accumulation in km2 (--save-acc)

The USER SETTINGS block right below holds the defaults: they are used
whenever the script is started without command-line arguments; any option
given on the command line applies instead.
"""

from __future__ import annotations

# ===========================================================================
# USER SETTINGS - the defaults; used when the script is run with no
#                 command-line arguments, overridden by any CLI flag
# ===========================================================================

METHOD = 1              # 1 = outlet coordinates (section 8.1)
                        # 2 = confluence criterion (sections 8.7.1-8.7.2)
                        # 3 = Pfafstetter / Ottobasins (section 8.7.3)

INPUTS_DIR = "data/03_flows"        # relative paths are resolved next to
OUTPUTS_DIR = "data/04_catchments"  # this script
INPUT_FILES = None      # None = process all *.tif in INPUTS_DIR, or a list,
                        # e.g. ["flow_direction_d8.tif", "flow_direction_mfd.tif"]
FORMAT = None           # None = auto-detect, or force "d8"/"dinf"/"mfd"/"mdinf"

# --- outlet coordinates, in the raster CRS (here EPSG:3067) -----------------
# Method 1: required; one (X, Y) per outlet, several = incremental basins.
# Methods 2/3: optional; first pair = basin outlet to subdivide
#              (None = whole raster for method 2, max-accumulation outlet
#               for method 3).
COORDS = [(379999, 6703741)]        # e.g. [(379999, 6703741), (376000, 6698000)]
SNAP_DIST = 100.0       # m; outlets snap to the max-accumulation cell within
                        # this distance (outlet pixel selection, section 8.1.3)

# --- stream network ---------------------------------------------------------
STREAM_THRESHOLD = 1.0  # km2; min accumulated drainage area that defines
                        # streams (controls network density, sections 7 & 8.7.1)

# --- method 2 options -------------------------------------------------------
MIN_AREA = None         # km2, e.g. 2.0; merge smaller sub-basins into their
                        # downstream neighbour (size restriction, section 8.7.2)
MIN_ORDER = None        # e.g. 2; only stream segments of Strahler order >= N
                        # form sub-basins (section 8.6)

# --- method 3 options -------------------------------------------------------
LEVELS = 1              # Pfafstetter recursion levels (1-9)

# --- multiple-flow-direction rasters (dinf / mfd / mdinf) -------------------
MFD_FRAC = 0.5          # a cell belongs to a catchment if >= this fraction of
                        # its flow reaches the outlet (0 = any contribution)

# --- extra outputs ----------------------------------------------------------
SAVE_ACC = False        # write flow-accumulation raster (km2) - useful for
                        # picking outlet coordinates
SAVE_STREAMS = False    # method 1: also write the stream-network layer

# ========================== end of USER SETTINGS ===========================

import argparse
import heapq
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio import features as rio_features
from numba import njit


def _xy(transform, row, col):
    """Cell-centre coordinates (avoids rasterio.transform helpers, whose
    inverse path calls numpy.linalg and can crash on broken BLAS setups)."""
    return transform * (col + 0.5, row + 0.5)


def _rowcol(transform, x, y):
    cf, rf = (~transform) * (x, y)
    return int(math.floor(rf)), int(math.floor(cf))

# Direction convention: index d = 0..7 -> N, NE, E, SE, S, SW, W, NW
DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
DIR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
# ESRI D8 code -> direction index
ESRI_CODES = {64: 0, 128: 1, 1: 2, 2: 3, 4: 4, 8: 5, 16: 6, 32: 7}
# Dinf angular sector k (k*45 deg CCW from east) -> direction index
ANG2DIR = np.array([2, 1, 0, 7, 6, 5, 4, 3], dtype=np.int64)


# --------------------------------------------------------------------------
# numba kernels (flat 1-D indexing; weights w have shape (8, ncells))
# --------------------------------------------------------------------------

@njit(cache=True)
def _topo_accumulate(w, off, valid, cell_area):
    """Weighted flow accumulation via Kahn's algorithm.

    Returns (acc [m2], topological order of all processed cells).
    """
    n = w.shape[1]
    indeg = np.zeros(n, dtype=np.int32)
    for d in range(8):
        o = off[d]
        for i in range(n):
            if w[d, i] > 0.0:
                indeg[i + o] += 1
    acc = np.zeros(n, dtype=np.float64)
    order = np.empty(n, dtype=np.int32)
    tail = 0
    for i in range(n):
        if valid[i]:
            acc[i] = cell_area
            if indeg[i] == 0:
                order[tail] = i
                tail += 1
    head = 0
    while head < tail:
        i = order[head]
        head += 1
        a = acc[i]
        for d in range(8):
            wd = w[d, i]
            if wd > 0.0:
                j = i + off[d]
                acc[j] += a * wd
                indeg[j] -= 1
                if indeg[j] == 0:
                    order[tail] = j
                    tail += 1
    return acc, order[:tail]


@njit(cache=True)
def _accum_dominant(ds, order, valid, cell_area):
    """Flow accumulation over the dominant-direction (D8-ified) tree."""
    n = ds.size
    acc = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if valid[i]:
            acc[i] = cell_area
    for k in range(order.size):
        i = order[k]
        j = ds[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc


@njit(cache=True)
def _fraction_to_outlets(w, off, order, is_outlet):
    """Fraction of each cell's flow that reaches the outlet cell(s).

    Walks the topological order in reverse: f(c) = sum_d w_d * f(down_d).
    For D8 the result is exactly 0/1 (binary membership, Fig. 8.3).
    """
    n = w.shape[1]
    f = np.zeros(n, dtype=np.float32)
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        if is_outlet[i]:
            f[i] = 1.0
        else:
            s = 0.0
            for d in range(8):
                wd = w[d, i]
                if wd > 0.0:
                    s += wd * f[i + off[d]]
            f[i] = s
    return f


@njit(cache=True)
def _propagate_labels_up(ds, order, labels):
    """Each unlabelled cell inherits the label of its dominant-downstream
    cell (reverse topological order: downstream first)."""
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        if labels[i] == 0:
            j = ds[i]
            if j >= 0:
                labels[i] = labels[j]


@njit(cache=True)
def _propagate_codes_up(ds, order, labels, domain, lo, hi):
    """Pfafstetter code propagation: cells labelled `domain` inherit the
    downstream cell's code when it lies in [lo, hi]."""
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        if labels[i] == domain:
            j = ds[i]
            if j >= 0:
                lj = labels[j]
                if lo <= lj <= hi:
                    labels[i] = lj


@njit(cache=True)
def _trace_main_stem(ds, acc, labels, domain, nrows, ncols, outlet, buf):
    """Trace the main stem upstream from `outlet` within cells labelled
    `domain`, choosing at each step the inflowing cell with the largest
    accumulation. Fills `buf`; returns the stem length."""
    n = 0
    cur = outlet
    buf[n] = cur
    n += 1
    while True:
        r = cur // ncols
        c = cur % ncols
        best = -1
        best_acc = -1.0
        for d in range(8):
            rr = r + DR[d]
            cc = c + DC[d]
            if 0 <= rr < nrows and 0 <= cc < ncols:
                nb = rr * ncols + cc
                if labels[nb] == domain and ds[nb] == cur and acc[nb] > best_acc:
                    best = nb
                    best_acc = acc[nb]
        if best < 0:
            break
        cur = best
        buf[n] = cur
        n += 1
    return n


@njit(cache=True)
def _find_tributaries(ds, acc, labels, domain, nrows, ncols, stem, min_acc):
    """Inflows to the main stem (other than the stem itself) with
    accumulation >= min_acc. Returns (mouth cells, stem position, acc)."""
    cap = 7 * stem.size + 1
    mouth = np.empty(cap, dtype=np.int32)
    pos = np.empty(cap, dtype=np.int32)
    cacc = np.empty(cap, dtype=np.float64)
    m = 0
    for k in range(stem.size):
        s = stem[k]
        nxt = stem[k + 1] if k + 1 < stem.size else -1
        r = s // ncols
        c = s % ncols
        for d in range(8):
            rr = r + DR[d]
            cc = c + DC[d]
            if 0 <= rr < nrows and 0 <= cc < ncols:
                nb = rr * ncols + cc
                if nb != nxt and labels[nb] == domain and ds[nb] == s and acc[nb] >= min_acc:
                    mouth[m] = nb
                    pos[m] = k
                    cacc[m] = acc[nb]
                    m += 1
    return mouth[:m], pos[:m], cacc[:m]


@njit(cache=True)
def _stream_inflows(ds, stream):
    """Number of upstream stream cells draining into each stream cell."""
    n = ds.size
    inflow = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if stream[i]:
            j = ds[i]
            if j >= 0 and stream[j]:
                inflow[j] += 1
    return inflow


@njit(cache=True)
def _build_segments(ds, dd, stream, inflow, starts, straight, diag):
    """Split the stream network into segments between confluences.

    A segment starts at a headwater (inflow 0) or a junction cell
    (inflow >= 2) and runs downstream until the next junction or the
    network end. Returns per-cell segment id (1-based), the cell after
    each segment's end (-1 = none), and segment lengths in metres.
    """
    nseg = starts.size
    seg_of = np.zeros(ds.size, dtype=np.int32)
    next_cell = np.full(nseg, -1, dtype=np.int32)
    end_cell = np.empty(nseg, dtype=np.int32)
    length = np.zeros(nseg, dtype=np.float64)
    for s in range(nseg):
        i = starts[s]
        seg_of[i] = s + 1
        while True:
            j = ds[i]
            if j < 0 or not stream[j]:
                end_cell[s] = i
                break
            length[s] += straight if dd[i] % 2 == 0 else diag
            if inflow[j] >= 2:
                end_cell[s] = i
                next_cell[s] = j
                break
            seg_of[j] = s + 1
            i = j
    return seg_of, end_cell, next_cell, length


# --------------------------------------------------------------------------
# Flow grid: load any format into the unified 8-plane weight model
# --------------------------------------------------------------------------

@dataclass
class FlowGrid:
    algo: str
    path: str
    weights: np.ndarray          # (8, n) float32
    valid: np.ndarray            # (n,) bool
    nrows: int
    ncols: int
    transform: object
    crs: object
    cell_size: float
    # derived (filled by build_derived)
    off: np.ndarray = None       # (8,) int64 flat offsets
    acc: np.ndarray = None       # (n,) float64 weighted accumulation, m2
    order: np.ndarray = None     # (m,) int32 topological order
    dd: np.ndarray = None        # (n,) int8 dominant direction, -1 = sink
    ds: np.ndarray = None        # (n,) int32 dominant downstream cell, -1
    acc_dd: np.ndarray = None    # (n,) float64 dominant-direction accumulation

    @property
    def n(self):
        return self.nrows * self.ncols

    @property
    def cell_area(self):
        return self.cell_size * self.cell_size


def detect_format(src) -> str:
    tag = src.tags().get("algorithm", "").lower()
    if tag in ("d8", "dinf", "mfd", "mdinf"):
        return tag
    if src.count == 8:
        return "mfd"
    if src.count != 1:
        raise ValueError(f"cannot auto-detect format of {src.name}: {src.count} bands")
    sample = src.read(1, out_shape=(min(512, src.height), min(512, src.width)))
    sample = sample[np.isfinite(sample)] if sample.dtype.kind == "f" else sample
    if sample.dtype.kind in "iu":
        return "d8"
    mx = sample.max() if sample.size else 0
    return "dinf" if mx <= 2 * math.pi + 1e-3 else "d8"


def load_flow_grid(path: str, fmt: str | None = None) -> FlowGrid:
    with rasterio.open(path) as src:
        algo = fmt or detect_format(src)
        h, wid = src.height, src.width
        n = h * wid
        cell = abs(src.res[0])
        if src.crs is not None and src.crs.is_geographic:
            print("  WARNING: geographic CRS - cell areas assume square metric cells")

        if algo == "d8":
            code = src.read(1)
            valid2 = code != 0
            w = np.zeros((8, n), dtype=np.float32)
            flat = code.ravel()
            for c, d in ESRI_CODES.items():
                w[d, flat == c] = 1.0
        elif algo == "dinf":
            ang = src.read(1)
            valid2 = np.isfinite(ang)
            flows = valid2 & (ang >= 0)
            a = np.where(flows, ang, 0).astype(np.float64).ravel()
            af = a / (math.pi / 4.0)
            k0 = np.floor(af).astype(np.int64) % 8
            frac = (af - np.floor(af)).astype(np.float32)
            k1 = (k0 + 1) % 8
            fl = flows.ravel()
            w = np.zeros((8, n), dtype=np.float32)
            for k in range(8):
                d = ANG2DIR[k]
                m = fl & (k0 == k)
                w[d, m] += 1.0 - frac[m]
                m = fl & (k1 == k)
                w[d, m] += frac[m]
        else:  # mfd / mdinf: 8 bands, fractions to N..NW = our exact order
            w = src.read().astype(np.float32, copy=False).reshape(8, n)
            # some writers store NaN (not 0) in bands receiving no flow, so a
            # cell is valid when ANY band is finite; nodata cells are all-NaN
            valid2 = np.isfinite(w).any(axis=0).reshape(h, wid)
            np.nan_to_num(w, copy=False)
            s = w.sum(axis=0)
            m = s > 0
            for d in range(8):
                np.divide(w[d], s, out=w[d], where=m)
            del s, m

        # zero weights on invalid source cells and weights whose target is
        # off-grid or nodata (that flow leaves the modelled area)
        for d in range(8):
            w2 = w[d].reshape(h, wid)
            w2[~valid2] = 0
            dr, dc = int(DR[d]), int(DC[d])
            tv = np.zeros_like(valid2)
            rs_src = slice(max(-dr, 0), h - max(dr, 0))
            cs_src = slice(max(-dc, 0), wid - max(dc, 0))
            rs_tgt = slice(max(dr, 0), h - max(-dr, 0))
            cs_tgt = slice(max(dc, 0), wid - max(-dc, 0))
            tv[rs_src, cs_src] = valid2[rs_tgt, cs_tgt]
            w2[~tv] = 0
        del tv

        return FlowGrid(
            algo=algo, path=path, weights=w, valid=valid2.ravel(),
            nrows=h, ncols=wid, transform=src.transform, crs=src.crs,
            cell_size=cell,
        )


def build_derived(g: FlowGrid):
    """Accumulation, topological order and dominant-direction products."""
    g.off = DR * g.ncols + DC
    t0 = time.time()
    g.acc, g.order = _topo_accumulate(g.weights, g.off, g.valid, g.cell_area)
    nvalid = int(g.valid.sum())
    if g.order.size < nvalid:
        print(f"  WARNING: {nvalid - g.order.size} cells in flow-direction cycles were skipped")
    wmax = g.weights.max(axis=0)
    g.dd = np.where(wmax > 0, g.weights.argmax(axis=0), -1).astype(np.int8)
    del wmax
    idx = np.flatnonzero(g.dd >= 0)
    g.ds = np.full(g.n, -1, dtype=np.int32)
    g.ds[idx] = idx + g.off[g.dd[idx]]
    del idx
    g.acc_dd = _accum_dominant(g.ds, g.order, g.valid, g.cell_area)
    imax = int(np.argmax(g.acc_dd))
    x, y = cell_to_xy(g, imax)
    print(f"  flow accumulation done in {time.time() - t0:.1f} s | "
          f"max {g.acc.max() / 1e6:.2f} km2 at ({x:.0f}, {y:.0f})")


def cell_to_xy(g: FlowGrid, cell: int):
    r, c = divmod(int(cell), g.ncols)
    return _xy(g.transform, r, c)


def snap_outlet(g: FlowGrid, x: float, y: float, snap_dist: float) -> int:
    """Snap a coordinate to the max-accumulation cell within snap_dist."""
    r, c = _rowcol(g.transform, x, y)
    if not (0 <= r < g.nrows and 0 <= c < g.ncols):
        raise SystemExit(f"ERROR: coordinate ({x}, {y}) is outside the raster")
    rad = max(0, int(round(snap_dist / g.cell_size)))
    r0, r1 = max(0, r - rad), min(g.nrows, r + rad + 1)
    c0, c1 = max(0, c - rad), min(g.ncols, c + rad + 1)
    win = g.acc_dd.reshape(g.nrows, g.ncols)[r0:r1, c0:c1]
    dr, dc = np.unravel_index(np.argmax(win), win.shape)
    cell = (r0 + int(dr)) * g.ncols + (c0 + int(dc))
    if not g.valid[cell]:
        raise SystemExit(f"ERROR: no valid cell within {snap_dist} m of ({x}, {y})")
    return cell


# --------------------------------------------------------------------------
# Stream network: segments + Strahler order (sections 8.1.2 iv, 8.6)
# --------------------------------------------------------------------------

@dataclass
class StreamNetwork:
    stream: np.ndarray       # (n,) bool
    seg_of: np.ndarray       # (n,) int32, 1-based segment id, 0 = not stream
    starts: np.ndarray       # (nseg,) first cell of each segment
    end_cell: np.ndarray     # (nseg,) last cell
    next_cell: np.ndarray    # (nseg,) cell downstream of end (-1 = none)
    length: np.ndarray       # (nseg,) metres
    seg_down: np.ndarray     # (nseg+1,) 1-based downstream segment id, 0 = none
    order: np.ndarray        # (nseg+1,) Strahler order (index 0 unused)

    @property
    def nseg(self):
        return self.starts.size


def build_stream_network(g: FlowGrid, threshold_km2: float,
                         stream_mask: np.ndarray | None = None) -> StreamNetwork:
    if stream_mask is None:
        stream_mask = g.valid & (g.acc_dd >= threshold_km2 * 1e6)
    inflow = _stream_inflows(g.ds, stream_mask)
    starts = np.flatnonzero(stream_mask & ((inflow == 0) | (inflow >= 2))).astype(np.int32)
    diag = g.cell_size * math.sqrt(2)
    seg_of, end_cell, next_cell, length = _build_segments(
        g.ds, g.dd, stream_mask, inflow, starts, g.cell_size, diag)
    nseg = starts.size
    seg_down = np.zeros(nseg + 1, dtype=np.int32)
    for s in range(nseg):
        if next_cell[s] >= 0:
            seg_down[s + 1] = seg_of[next_cell[s]]
    order = strahler_orders(seg_down, nseg)
    return StreamNetwork(stream_mask, seg_of, starts, end_cell, next_cell,
                         length, seg_down, order)


def strahler_orders(seg_down: np.ndarray, nseg: int) -> np.ndarray:
    """Strahler ordering per the rules of section 8.6."""
    from collections import deque
    indeg = np.zeros(nseg + 1, dtype=np.int32)
    for s in range(1, nseg + 1):
        indeg[seg_down[s]] += 1
    order = np.ones(nseg + 1, dtype=np.int32)
    maxin = np.zeros(nseg + 1, dtype=np.int32)
    cntmax = np.zeros(nseg + 1, dtype=np.int32)
    remaining = indeg.copy()
    q = deque(s for s in range(1, nseg + 1) if indeg[s] == 0)
    while q:
        s = q.popleft()
        order[s] = 1 if maxin[s] == 0 else (maxin[s] + 1 if cntmax[s] >= 2 else maxin[s])
        d = seg_down[s]
        if d:
            if order[s] > maxin[d]:
                maxin[d], cntmax[d] = order[s], 1
            elif order[s] == maxin[d]:
                cntmax[d] += 1
            remaining[d] -= 1
            if remaining[d] == 0:
                q.append(d)
    return order


def filter_network_by_order(g: FlowGrid, net: StreamNetwork, min_order: int) -> StreamNetwork:
    """Keep only stream cells in segments of Strahler order >= min_order,
    then rebuild the segments (chains across pruned junctions merge)."""
    ord_cell = np.zeros(g.n, dtype=np.int32)
    on = net.seg_of > 0
    ord_cell[on] = net.order[net.seg_of[on]]
    keep = ord_cell >= min_order
    return build_stream_network(g, 0, stream_mask=keep)


# --------------------------------------------------------------------------
# Method 1: outlet coordinates (section 8.1)
# --------------------------------------------------------------------------

def method1_outlets(g: FlowGrid, coords: list[tuple[float, float]],
                    snap_dist: float, mfd_frac: float):
    labels = np.zeros(g.n, dtype=np.int32)
    outlets = []
    for oid, (x, y) in enumerate(coords, start=1):
        cell = snap_outlet(g, x, y, snap_dist)
        sx, sy = cell_to_xy(g, cell)
        outlets.append((oid, cell))
        print(f"  outlet {oid}: ({x:.1f}, {y:.1f}) snapped to ({sx:.1f}, {sy:.1f}), "
              f"upstream area {g.acc_dd[cell] / 1e6:.3f} km2")
    # assign upstream outlets first -> incremental basins (Fig. 8.4)
    for oid, cell in sorted(outlets, key=lambda t: g.acc_dd[t[1]]):
        is_outlet = np.zeros(g.n, dtype=np.bool_)
        is_outlet[cell] = True
        f = _fraction_to_outlets(g.weights, g.off, g.order, is_outlet)
        memb = (f > 0) if mfd_frac <= 0 else (f >= mfd_frac)
        labels[memb & (labels == 0)] = oid
        del f, memb
    attrs = {oid: {} for oid, _ in outlets}
    return labels, attrs


# --------------------------------------------------------------------------
# Method 2: confluence criterion (+ size restriction) (sections 8.7.1-8.7.2)
# --------------------------------------------------------------------------

def method2_confluence(g: FlowGrid, net: StreamNetwork,
                       min_area_km2: float | None,
                       basin_outlet: int | None):
    if net.nseg == 0:
        raise SystemExit("ERROR: no stream cells - lower --stream-threshold")
    labels = net.seg_of.astype(np.int32).copy()
    _propagate_labels_up(g.ds, g.order, labels)
    if basin_outlet is not None:
        basin = np.zeros(g.n, dtype=np.int32)
        basin[basin_outlet] = 1
        _propagate_labels_up(g.ds, g.order, basin)
        labels[basin == 0] = 0
        del basin

    seg_order = net.order  # (nseg+1,)
    counts = np.bincount(labels[labels > 0], minlength=net.nseg + 1)
    areas = counts * g.cell_area / 1e6  # km2 per segment id

    group_of = np.arange(net.nseg + 1)

    def find(s):
        while group_of[s] != s:
            group_of[s] = group_of[group_of[s]]
            s = group_of[s]
        return s

    if min_area_km2:
        garea = areas.copy().astype(np.float64)
        gdown = net.seg_down.copy()
        gorder = seg_order.copy()
        heap = [(garea[s], s) for s in range(1, net.nseg + 1)
                if 0 < garea[s] < min_area_km2]
        heapq.heapify(heap)
        merged = 0
        while heap:
            a, s = heapq.heappop(heap)
            r = find(s)
            if r != s or a != garea[r] or a >= min_area_km2 or a == 0:
                continue  # stale entry
            d = find(gdown[r])
            if d == 0 or d == r:
                continue  # terminal basin: nothing downstream to merge into
            group_of[r] = d
            garea[d] += garea[r]
            gorder[d] = max(gorder[d], gorder[r])
            merged += 1
            if garea[d] < min_area_km2:
                heapq.heappush(heap, (garea[d], d))
        # relabel cells to group roots, renumber 1..K
        roots = np.array([find(s) for s in range(net.nseg + 1)], dtype=np.int32)
        labels = roots[labels]
        seg_order = gorder
        print(f"  size restriction: merged {merged} sub-basins into downstream neighbours")

    # renumber to compact 1..K
    used = np.unique(labels[labels > 0])
    remap = np.zeros(labels.max() + 1, dtype=np.int32)
    remap[used] = np.arange(1, used.size + 1)
    labels = remap[labels]
    attrs = {int(remap[u]): {"strahler_order": int(seg_order[u])} for u in used}
    return labels, attrs


# --------------------------------------------------------------------------
# Method 3: Pfafstetter / Ottobasins (section 8.7.3)
# --------------------------------------------------------------------------

def method3_pfafstetter(g: FlowGrid, outlet: int, levels: int, min_trib_acc_m2: float):
    labels = np.zeros(g.n, dtype=np.int32)
    labels[outlet] = -1
    _propagate_labels_up(g.ds, g.order, labels)  # basin cells -> -1
    nbasin = int((labels == -1).sum())
    print(f"  root basin: {nbasin * g.cell_area / 1e6:.2f} km2")
    stembuf = np.empty(g.n, dtype=np.int32)

    def subdivide(domain: int, out_cell: int, level: int):
        ns = _trace_main_stem(g.ds, g.acc_dd, labels, domain,
                              g.nrows, g.ncols, out_cell, stembuf)
        stem = stembuf[:ns].copy()
        mouth, pos, cacc = _find_tributaries(g.ds, g.acc_dd, labels, domain,
                                             g.nrows, g.ncols, stem, min_trib_acc_m2)
        if mouth.size == 0:
            return False  # leaf: keeps the parent code
        t = min(4, mouth.size)
        sel = np.argsort(cacc)[::-1][:t]         # t largest tributaries
        sel = sel[np.argsort(pos[sel], kind="stable")]  # downstream -> upstream
        jpos = pos[sel].astype(np.int64)
        base = 0 if domain == -1 else domain * 10
        # interbasin codes 1,3,..,2t+1 along the stem; junction cell stays
        # with the downstream reach
        reach_idx = np.searchsorted(jpos, np.arange(ns), side="left")
        labels[stem] = (base + 2 * reach_idx + 1).astype(np.int32)
        labels[mouth[sel]] = base + 2 * (np.arange(t, dtype=np.int32) + 1)
        _propagate_codes_up(g.ds, g.order, labels, domain,
                            base + 1, base + 2 * t + 1)
        if level > 1:
            for i in range(t):  # tributary basins: even codes
                subdivide(int(base + 2 * (i + 1)), int(mouth[sel][i]), level - 1)
            for r in range(t + 1):  # interbasins: odd codes
                if not np.any(reach_idx == r):
                    continue  # empty reach (two tributaries at the same cell)
                out_c = int(stem[0]) if r == 0 else int(stem[jpos[r - 1] + 1])
                subdivide(int(base + 2 * r + 1), out_c, level - 1)
        return True

    if not subdivide(-1, outlet, levels):
        print("  no tributaries above the threshold: basin not subdivided")
        labels[labels == -1] = 1
    attrs = {int(v): {"pfaf_code": int(v)} for v in np.unique(labels[labels > 0])}
    return labels, attrs


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def label_stream_length(g: FlowGrid, net: StreamNetwork, labels: np.ndarray):
    """Total stream length (m) per catchment label (for drainage density)."""
    cells = np.flatnonzero(net.stream & (labels > 0))
    if cells.size == 0:
        return {}
    step = np.where(g.dd[cells] % 2 == 0, g.cell_size, g.cell_size * math.sqrt(2))
    step[g.dd[cells] < 0] = 0.0
    labs = labels[cells]
    uniq, inv = np.unique(labs, return_inverse=True)
    sums = np.bincount(inv, weights=step)
    return {int(u): float(s) for u, s in zip(uniq, sums)}


def write_label_raster(path: str, g: FlowGrid, labels: np.ndarray, desc: str):
    prof = dict(driver="GTiff", height=g.nrows, width=g.ncols, count=1,
                dtype="int32", crs=g.crs, transform=g.transform, nodata=0,
                compress="lzw", tiled=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(labels.reshape(g.nrows, g.ncols), 1)
        dst.set_band_description(1, desc)
        dst.update_tags(source=os.path.basename(g.path), algorithm=g.algo)


def write_acc_raster(path: str, g: FlowGrid):
    prof = dict(driver="GTiff", height=g.nrows, width=g.ncols, count=1,
                dtype="float32", crs=g.crs, transform=g.transform,
                nodata=-1, compress="lzw", tiled=True)
    acc = (g.acc / 1e6).astype(np.float32)
    acc[~g.valid] = -1
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(acc.reshape(g.nrows, g.ncols), 1)
        dst.set_band_description(1, f"{g.algo} flow accumulation (km2)")


def catchments_to_gdf(g: FlowGrid, labels: np.ndarray, attrs: dict,
                      stream_len: dict):
    import geopandas as gpd
    from shapely.geometry import shape as shp_shape
    from shapely.geometry import MultiPolygon

    labels2 = labels.reshape(g.nrows, g.ncols)
    polys: dict[int, list] = {}
    for geom, val in rio_features.shapes(labels2, mask=labels2 != 0,
                                         transform=g.transform, connectivity=8):
        polys.setdefault(int(val), []).append(shp_shape(geom))
    uniq, counts = np.unique(labels[labels > 0], return_counts=True)
    count_of = dict(zip(uniq.tolist(), counts.tolist()))
    rows, geoms = [], []
    for val in sorted(polys):
        plist = polys[val]
        geom = plist[0] if len(plist) == 1 else MultiPolygon(plist)
        area_km2 = count_of.get(val, 0) * g.cell_area / 1e6
        sl_km = stream_len.get(val, 0.0) / 1000.0
        row = {
            "id": val,
            "area_km2": round(area_km2, 6),
            "perimeter_km": round(geom.length / 1000.0, 6),
            "stream_len_km": round(sl_km, 6),
            "drainage_density": round(sl_km / area_km2, 6) if area_km2 > 0 else 0.0,
        }
        row.update(attrs.get(val, {}))
        rows.append(row)
        geoms.append(geom)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=g.crs)


def streams_to_gdf(g: FlowGrid, net: StreamNetwork):
    import geopandas as gpd
    from shapely.geometry import LineString

    rows, geoms = [], []
    for s in range(net.nseg):
        cells = [int(net.starts[s])]
        i = cells[0]
        while True:
            j = int(g.ds[i])
            if j < 0 or not net.stream[j]:
                break
            cells.append(j)  # include the junction cell so lines connect
            if net.seg_of[j] != s + 1:
                break
            i = j
        if len(cells) < 2:
            continue
        pts = [_xy(g.transform, c // g.ncols, c % g.ncols) for c in cells]
        rows.append({
            "segment_id": s + 1,
            "strahler_order": int(net.order[s + 1]),
            "acc_km2": round(float(g.acc_dd[net.end_cell[s]]) / 1e6, 6),
            "length_km": round(float(net.length[s]) / 1000.0, 6),
        })
        geoms.append(LineString(pts))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=g.crs)


def summarize(labels: np.ndarray, g: FlowGrid, stream_len: dict):
    uniq, counts = np.unique(labels[labels > 0], return_counts=True)
    if uniq.size == 0:
        print("  RESULT: no catchments delineated")
        return
    areas = counts * g.cell_area / 1e6
    total_len = sum(stream_len.values()) / 1000.0
    dd = total_len / areas.sum() if areas.sum() > 0 else 0.0
    print(f"  RESULT: {uniq.size} catchment(s) | area min/mean/max = "
          f"{areas.min():.3f} / {areas.mean():.3f} / {areas.max():.3f} km2 | "
          f"total {areas.sum():.2f} km2 | drainage density {dd:.3f} km/km2")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_coords(values):
    out = []
    for v in values or []:
        try:
            x, y = (float(p) for p in v.replace(";", ",").split(","))
        except ValueError:
            raise SystemExit(f"ERROR: bad --coords '{v}', expected X,Y")
        out.append((x, y))
    return out


def short_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len("flow_direction_"):] if stem.startswith("flow_direction_") else stem


def settings_to_argv() -> list[str]:
    """Translate the USER SETTINGS block into command-line arguments
    (used when the script is run without arguments)."""
    here = os.path.dirname(os.path.abspath(__file__))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(here, p)

    argv = ["--method", str(METHOD),
            "--inputs-dir", resolve(INPUTS_DIR),
            "--outputs-dir", resolve(OUTPUTS_DIR),
            "--snap-dist", str(SNAP_DIST),
            "--stream-threshold", str(STREAM_THRESHOLD),
            "--levels", str(LEVELS),
            "--mfd-frac", str(MFD_FRAC)]
    for f in INPUT_FILES or []:
        argv += ["--input", f]
    if FORMAT:
        argv += ["--format", FORMAT]
    for x, y in COORDS or []:
        argv += ["--coords", f"{x},{y}"]
    if MIN_AREA is not None:
        argv += ["--min-area", str(MIN_AREA)]
    if MIN_ORDER is not None:
        argv += ["--min-order", str(MIN_ORDER)]
    if SAVE_ACC:
        argv += ["--save-acc"]
    if SAVE_STREAMS:
        argv += ["--save-streams"]
    return argv


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", type=int, choices=(1, 2, 3), required=True,
                    help="1=outlet coordinates, 2=confluence criterion, 3=Pfafstetter")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--inputs-dir", default=os.path.join(here, "data", "03_flows"))
    ap.add_argument("--outputs-dir", default=os.path.join(here, "data", "04_catchments"))
    ap.add_argument("--input", action="append", default=None, metavar="NAME",
                    help="process only this file from inputs dir (repeatable); default: all *.tif")
    ap.add_argument("--format", choices=("d8", "dinf", "mfd", "mdinf"), default=None,
                    help="override flow-direction format auto-detection")
    ap.add_argument("--coords", action="append", metavar="X,Y",
                    help="outlet coordinates in the raster CRS (method 1: required, "
                         "repeatable; methods 2/3: optional basin outlet)")
    ap.add_argument("--snap-dist", type=float, default=100.0,
                    help="snap outlets to the max-accumulation cell within this "
                         "distance in metres (default 100)")
    ap.add_argument("--stream-threshold", type=float, default=1.0, metavar="KM2",
                    help="min accumulated drainage area defining streams (default 1.0 km2)")
    ap.add_argument("--min-area", type=float, default=None, metavar="KM2",
                    help="method 2: merge sub-basins smaller than this into their "
                         "downstream neighbour (size restriction, section 8.7.2)")
    ap.add_argument("--min-order", type=int, default=None, metavar="N",
                    help="method 2: only stream segments of Strahler order >= N form sub-basins")
    ap.add_argument("--levels", type=int, default=1,
                    help="method 3: Pfafstetter recursion levels (default 1, max 9)")
    ap.add_argument("--mfd-frac", type=float, default=0.5, metavar="F",
                    help="dinf/mfd/mdinf: a cell belongs to a catchment if >= F of its "
                         "flow reaches the outlet (default 0.5; 0 = any contribution)")
    ap.add_argument("--save-acc", action="store_true",
                    help="also write the flow-accumulation raster (km2)")
    ap.add_argument("--save-streams", action="store_true",
                    help="method 1: also write the stream-network layer")
    if argv is None and len(sys.argv) <= 1:
        argv = settings_to_argv()  # no CLI arguments: use the USER SETTINGS
        print("running with the USER SETTINGS from the top of the script")
    args = ap.parse_args(argv)

    coords = parse_coords(args.coords)
    if args.method == 1 and not coords:
        ap.error("method 1 requires at least one --coords X,Y")
    if args.method == 3 and not 1 <= args.levels <= 9:
        ap.error("--levels must be between 1 and 9")

    if args.input:
        files = [os.path.join(args.inputs_dir, f) for f in args.input]
    else:
        files = sorted(os.path.join(args.inputs_dir, f)
                       for f in os.listdir(args.inputs_dir)
                       if f.lower().endswith((".tif", ".tiff")))
    if not files:
        raise SystemExit(f"ERROR: no rasters found in {args.inputs_dir}")
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise SystemExit(f"ERROR: input not found: {', '.join(missing)}")
    os.makedirs(args.outputs_dir, exist_ok=True)

    for path in files:
        t0 = time.time()
        name = short_name(path)
        print(f"\n=== {os.path.basename(path)}")
        g = load_flow_grid(path, args.format)
        print(f"  format {g.algo} | {g.nrows}x{g.ncols} cells @ {g.cell_size} m | CRS {g.crs}")
        build_derived(g)

        if args.save_acc:
            accp = os.path.join(args.outputs_dir, f"acc_{name}.tif")
            write_acc_raster(accp, g)
            print(f"  wrote {accp}")

        net = build_stream_network(g, args.stream_threshold)
        print(f"  stream network: threshold {args.stream_threshold} km2 -> "
              f"{net.nseg} segments, max Strahler order "
              f"{int(net.order[1:].max()) if net.nseg else 0}")

        if args.method == 1:
            labels, attrs = method1_outlets(g, coords, args.snap_dist, args.mfd_frac)
        elif args.method == 2:
            basin_outlet = snap_outlet(g, *coords[0], args.snap_dist) if coords else None
            net2 = net
            if args.min_order:
                net2 = filter_network_by_order(g, net, args.min_order)
                print(f"  order filter (>= {args.min_order}): {net2.nseg} segments remain")
            labels, attrs = method2_confluence(g, net2, args.min_area, basin_outlet)
            net = net2
        else:
            if coords:
                outlet = snap_outlet(g, *coords[0], args.snap_dist)
            else:
                outlet = int(np.argmax(g.acc_dd))
                x, y = cell_to_xy(g, outlet)
                print(f"  no --coords: using max-accumulation cell ({x:.1f}, {y:.1f}) as outlet")
            labels, attrs = method3_pfafstetter(
                g, outlet, args.levels, args.stream_threshold * 1e6)

        g.weights = None  # free ~1.2 GB before polygonizing

        stream_len = label_stream_length(g, net, labels)
        summarize(labels, g, stream_len)

        base = os.path.join(args.outputs_dir, f"catchments_{name}_method{args.method}")
        write_label_raster(base + ".tif", g, labels,
                           f"catchment labels (method {args.method}, {g.algo})")
        gdf = catchments_to_gdf(g, labels, attrs, stream_len)
        gdf.to_file(base + ".gpkg", layer="catchments", driver="GPKG")
        if args.method in (2, 3) or args.save_streams:
            sgdf = streams_to_gdf(g, net)
            if len(sgdf):
                sgdf.to_file(base + ".gpkg", layer="streams", driver="GPKG")
        print(f"  wrote {base}.tif and {base}.gpkg ({len(gdf)} catchments) "
              f"in {time.time() - t0:.1f} s total")


if __name__ == "__main__":
    main()
