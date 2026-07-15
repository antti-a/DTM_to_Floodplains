#!/usr/bin/env python3
"""Geomorphic floodplains (GFPLAIN) from the pipeline's D8 products.

Created on Wed Jul 8 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 6, optional (see README.md):
    reads   data/02_filled/*.tif                            (02_fill_dem.py)
            data/03_flows/flow_direction_d8.tif             (03_flow_router.py)
            data/04_accumulation/flow_accumulation_d8.tif   (04_flow_accumulation.py)
    writes  data/06_floodplains/floodplains.tif

GFPLAIN (Nardi et al., 2019) delineates the geomorphic floodplain from
terrain alone: every stream cell carries a flood level ``h = a * A**b``
(h in metres, A = upstream area in km2) and a hillslope cell belongs to the
floodplain if it rises no more than ``h`` above the stream cell it drains
to. Both ``a`` and ``b`` are adjustable, to be calibrated against
observed/modelled flood extents - the literature exponent b ~ 0.30 (Nardi
et al., 2019) is usually fixed first, then ``a`` fitted.

This stage recomputes nothing the pipeline already produced - the filled
DEM, the D8 flow directions and the D8 flow accumulation are read as-is;
pyflwdir is used only to turn the existing D8 raster into a flow graph (one
downstream index per cell plus a down-to-upstream cell ordering).

How it works
------------
1. The filled DEM tiles are mosaicked in memory (tile validation,
   mosaicking and raster loading come from the companion module
   pipeline_io.py; no mosaic file is written).
2. ``flow_direction_d8.tif`` is remapped to pyflwdir's uint8 convention
   (identical ESRI direction codes; -1 flat / -2 pit -> 0 = pit,
   0 nodata -> 247) and loaded with ``pyflwdir.from_array(ftype="d8")``.
3. The stage-4 D8 flow accumulation, converted to km2, gives the upstream
   area A: stream cells (A >= ``upa-min``) are floodplain by definition and
   carry ``h = a * A**b``; walking from down- to upstream, a hillslope cell
   joins the floodplain of the stream cell it drains to if its elevation
   rises no more than that h above it.

The kernel is adapted from :func:`pyflwdir.dem.floodplains` (plain,
uncompiled Python in pyflwdir <= 0.5.11, far too slow for a 36M-cell grid),
extended with the explicit coefficient ``a`` (pyflwdir's built-in is the
``a = 1`` special case).

Output
------
``data/06_floodplains/floodplains.tif`` - int8, deflate-compressed GeoTIFF:
1 = floodplain, 0 = upland, -1 = nodata.

Spatial reference
-----------------
* Horizontal: EPSG:3067 (ETRS89 / TM35FIN), units metres, 2 m cells.
* Vertical datum of the source elevation data: N2000, units metres.

Credits
-------
* Source data: filled DEM and D8 rasters from the earlier pipeline stages,
  derived from a 2 m digital elevation model in EPSG:3067 / N2000 - presumed
  to be the National Land Survey of Finland (Maanmittauslaitos) 2 m
  elevation model (KM2), licensed CC BY 4.0. Edit
  ``SOURCE_DATA_CREDIT_PRESUMED`` in pipeline_io.py if the provenance
  differs.
* Concept: Nardi et al. (2019) GFPLAIN. Full reference in
  ``GFPLAIN_CITATION`` below.
* Tools that enabled this work: Python, NumPy (Harris et al., 2020),
  Numba (Lam, Pitrou and Seibert, 2015), pyflwdir (Eilander et al., 2021),
  rasterio (Gillies et al.) on GDAL (GDAL/OGR contributors, OSGeo).

Usage (inside the ``water`` conda environment, ``conda activate water``)
-----
    python 06_floodplains.py              # defaults from USER SETTINGS below
    python 06_floodplains.py --a 0.8      # recalibrated coefficient
    python 06_floodplains.py --a 1.0 --b 0.35 --upa-min 0.5
"""

from __future__ import annotations

# ===========================================================================
# USER SETTINGS - these feed the argparse defaults, so they apply to a
#                 no-argument run; any CLI flag overrides them
# ===========================================================================

INPUTS_DIR = "data/02_filled"       # filled DEM tiles; relative paths are
OUTPUTS_DIR = "data/06_floodplains" # resolved next to this script
DEM_FILES = None        # None = mosaic all *.tif in INPUTS_DIR, or a list,
                        # e.g. ["filled_carved_L4142E.tif"]

D8_RASTER = "data/03_flows/flow_direction_d8.tif"
                        # D8 flow directions (03_flow_router.py output)
UPAREA_RASTER = "data/04_accumulation/flow_accumulation_d8.tif"
                        # D8 flow accumulation (04_flow_accumulation.py
                        # output); its m2/cells units tag is honoured

# The stream-initiation threshold (--upa-min default) is the shared
# UPA_MIN constant in pipeline_io.py, common to stages 3, 5 and 6.

# GFPLAIN power law h = a * A**b (h in m, A in km2): calibrate `a` (and, if
# needed, `b`) against observed/modelled flood extents; the literature value
# b ~ 0.30 (Nardi et al., 2019) is fixed first, then `a` fitted.
GFPLAIN_A = 0.63        # coefficient a [-]
FLOODPLAIN_B = 0.3      # exponent b [-]

# ========================== end of USER SETTINGS ===========================

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

from pipeline_io import (
    NODATA, SOURCE_DATA_CREDIT_KNOWN, SOURCE_DATA_CREDIT_PRESUMED,
    TOOL_CREDITS_PYFLWDIR, UPA_MIN, build_flwdir, build_mosaic,
    collect_provenance, find_dems, load_d8, load_uparea, resolve_near,
    validate_tiles, write_raster,
)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FP_NODATA = -1          # floodplain raster: 1 floodplain, 0 upland, -1 nodata

GFPLAIN_CITATION = (
    "Nardi, F., Annis, A., Di Baldassarre, G., Vivoni, E.R. and Grimaldi, S. "
    "(2019) 'GFPLAIN250m, a global high-resolution dataset of Earth's "
    "floodplains', Scientific Data, 6, 180309, doi:10.1038/sdata.2018.309."
)


# ---------------------------------------------------------------------------
# GFPLAIN kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _gfplain_kernel(idxs_ds, seq, elevtn, uparea, upa_min, a, b):
    """GFPLAIN floodplain flag per cell: 1 floodplain, 0 upland, -1 nodata.

    Adapted from :func:`pyflwdir.dem.floodplains` (Nardi et al., 2019)
    extended with the coefficient ``a``: a stream cell (``uparea >= upa_min``)
    carries the flood level ``h = a * A**b``; a non-stream cell belongs to the
    floodplain if it rises no more than ``h`` above the stream cell it drains
    to (pyflwdir's built-in is the ``a = 1`` special case).
    """
    drainh = np.full(uparea.size, -9999.0, dtype=np.float32)
    drainz = np.full(uparea.size, -9999.0, dtype=np.float32)
    fldpln = np.full(uparea.size, -1, dtype=np.int8)
    for i in range(seq.size):
        fldpln[seq[i]] = 0
    for i in range(seq.size):  # down- to upstream
        idx0 = seq[i]
        if uparea[idx0] >= upa_min:
            drainh[idx0] = a * uparea[idx0] ** b
            drainz[idx0] = elevtn[idx0]
            fldpln[idx0] = 1
        else:
            idx_ds = idxs_ds[idx0]
            if fldpln[idx_ds] == 1:
                z0 = drainz[idx_ds]
                h0 = drainh[idx_ds]
                if elevtn[idx0] - z0 <= h0:
                    fldpln[idx0] = 1
                    drainz[idx0] = z0
                    drainh[idx0] = h0
    return fldpln


def compute_floodplains(flw, elevtn, uparea, upa_min, a, b):
    """Return the GFPLAIN floodplain grid (int8) for h = a * A**b."""
    return _gfplain_kernel(
        flw.idxs_ds, flw.idxs_seq, elevtn.ravel(), uparea.ravel(),
        np.float32(upa_min), np.float32(a), np.float32(b),
    ).reshape(flw.shape)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    # The USER SETTINGS block at the top of the script feeds the argparse
    # defaults directly, so there is exactly one source of truth per value.
    ap = argparse.ArgumentParser(
        description="GFPLAIN geomorphic floodplains (h = a*A**b) from the "
                    "pipeline's filled DEM, D8 flow directions and D8 flow "
                    "accumulation.")
    ap.add_argument("--dem", nargs="+", default=None, metavar="TIF",
                    help="filled DEM tiles (default: all in --inputs-dir)")
    ap.add_argument("--inputs-dir", type=Path,
                    default=resolve_near(INPUTS_DIR, HERE))
    ap.add_argument("--outputs-dir", type=Path,
                    default=resolve_near(OUTPUTS_DIR, HERE))
    ap.add_argument("--d8", type=Path,
                    default=resolve_near(D8_RASTER, HERE),
                    help="D8 flow-direction raster (03_flow_router.py output)")
    ap.add_argument("--uparea", type=Path,
                    default=resolve_near(UPAREA_RASTER, HERE),
                    help="D8 flow-accumulation raster "
                         "(04_flow_accumulation.py output)")
    ap.add_argument("--upa-min", type=float, default=UPA_MIN, metavar="KM2",
                    help="stream-initiation threshold: cells with at least "
                         "this upstream area are stream cells "
                         f"(default {UPA_MIN:g})")
    ap.add_argument("--a", type=float, default=GFPLAIN_A,
                    help=f"GFPLAIN coefficient a in h = a*A**b (default "
                         f"{GFPLAIN_A:g}; calibrate against flood hazard maps)")
    ap.add_argument("--b", type=float, default=FLOODPLAIN_B,
                    help=f"GFPLAIN exponent b in h = a*A**b (default "
                         f"{FLOODPLAIN_B:g}, after Nardi et al., 2019)")
    args = ap.parse_args(argv)
    if args.dem is None and DEM_FILES:
        args.dem = [str(args.inputs_dir / f) for f in DEM_FILES]

    t0 = time.perf_counter()
    dem_paths = find_dems(args.dem, args.inputs_dir)
    print(f"DEM tiles ({len(dem_paths)}): "
          + ", ".join(p.name for p in dem_paths))
    validate_tiles(dem_paths)
    forwarded = collect_provenance(dem_paths)

    elevtn, transform, crs = build_mosaic(dem_paths)
    shape = elevtn.shape
    print(f"mosaic {shape[1]} x {shape[0]} cells, "
          f"cell {abs(transform.a)} x {abs(transform.e)} m")

    d8u8, routing_alg = load_d8(args.d8, transform, shape, crs)
    flw = build_flwdir(d8u8, transform)
    del d8u8

    uparea = load_uparea(args.uparea, transform, shape, crs, routing_alg)
    n_stream = int((uparea >= np.float32(args.upa_min)).sum())
    print(f"stream cells: {n_stream} (upstream area >= {args.upa_min:g} km2, "
          f"max {float(uparea.max()):.2f} km2)")
    if n_stream == 0:
        sys.exit(f"no cell reaches --upa-min {args.upa_min:g} km2; "
                 f"no stream to grow a floodplain from")

    fldpln = compute_floodplains(
        flw, elevtn, uparea, args.upa_min, args.a, args.b
    )
    fldpln[elevtn == NODATA] = FP_NODATA
    del uparea

    out_path = write_raster(
        args.outputs_dir / "floodplains.tif", fldpln, transform, crs,
        nodata=FP_NODATA, dtype="int8",
        tags=dict(
            title="Geomorphic floodplains (GFPLAIN)",
            algorithm="GFPLAIN: stream cells carry the flood level "
                      "h = a * A**b (h in m, A = upstream area in km2); a "
                      "cell joins the floodplain of the stream cell it "
                      "drains to (D8) if it rises no more than h above it",
            citation=GFPLAIN_CITATION,
            parameters=f"a={args.a:g}, b={args.b:g}, "
                       f"upa_min={args.upa_min:g} km2; h = a * A**b",
            flow_routing_algorithm=routing_alg,
            stream_threshold_km2=f"{args.upa_min:g}",
            gfplain_a=f"{args.a:g}",
            gfplain_b=f"{args.b:g}",
            class_encoding="1 = floodplain, 0 = upland, -1 = nodata",
            source_dem_tiles=", ".join(p.name for p in dem_paths),
            source_flow_direction_raster=args.d8.name,
            source_flow_accumulation_raster=args.uparea.name,
            source_data_credit=(SOURCE_DATA_CREDIT_KNOWN
                                if "dem_source_tiles" in forwarded
                                else SOURCE_DATA_CREDIT_PRESUMED),
            horizontal_crs="EPSG:3067 (ETRS89 / TM35FIN), units metres",
            vertical_datum="N2000, units metres (datum of the source DEM)",
            software_credits=TOOL_CREDITS_PYFLWDIR,
            generated_by="06_floodplains.py",
            **forwarded,
        ),
    )

    n_fp = int((fldpln == 1).sum())
    cell_km2 = abs(transform.a * transform.e) / 1e6
    print(f"floodplain: {n_fp} cells = {n_fp * cell_km2:.2f} km2 "
          f"(a={args.a:g}, b={args.b:g}, upa_min={args.upa_min:g} km2)")
    print(f"-> {out_path}  ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
