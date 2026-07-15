#!/usr/bin/env python3
"""Height above nearest drain (HAND) from the pipeline's D8 products.

Created on Wed Jul 8 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 5, optional (see README.md):
    reads   data/02_filled/*.tif                            (02_fill_dem.py)
            data/03_flows/flow_direction_d8.tif             (03_flow_router.py)
            data/04_accumulation/flow_accumulation_d8.tif   (04_flow_accumulation.py)
    writes  data/05_hand/hand.tif

HAND (Nobre et al., 2016) is the vertical distance between a pixel and the
stream pixel it drains to along the D8 flow path: the local flood-relevant
"height above the river". This stage recomputes nothing the pipeline already
produced - the filled DEM, the D8 flow directions and the D8 flow
accumulation are read as-is; pyflwdir is used only to turn the existing D8
raster into a flow graph (one downstream index per pixel plus a down-to-
upstream pixel ordering).

How it works
------------
1. The filled DEM tiles are mosaicked in memory (tile validation,
   mosaicking and raster loading come from the companion module
   pipeline_io.py; no mosaic file is written).
2. ``flow_direction_d8.tif`` is remapped to pyflwdir's uint8 convention
   (identical ESRI direction codes; -1 flat / -2 pit -> 0 = pit,
   0 nodata -> 247) and loaded with ``pyflwdir.from_array(ftype="d8")``.
3. Stream (drain) pixels are those whose stage-4 D8 flow accumulation,
   converted to km2, reaches the stream-initiation threshold ``upa-min``.
4. A Numba kernel walks the flow graph from down- to upstream: a drain pixel
   gets HAND = 0, every other pixel gets its downstream neighbour's HAND plus
   the elevation difference to it.

The kernel is adapted from :func:`pyflwdir.dem.height_above_nearest_drain`
(plain, uncompiled Python in pyflwdir <= 0.5.11, far too slow for a 36M-pixel
grid) with one deliberate deviation: pyflwdir assigns height-above-*pit* to
pixels whose flow path ends in a pit or at the grid edge without ever meeting
a drain; on a whole-DEM run that would fill every small edge catchment below
the threshold with misleading values, so those pixels are written as nodata
instead.

Output
------
``data/05_hand/hand.tif`` - float32, deflate-compressed GeoTIFF. Values are
metres above the nearest drain pixel along the D8 flow path; drain pixels are
0. NoData (-9999) marks pixels outside the DEM and pixels that drain to a
pit/edge without meeting a stream.

Spatial reference
-----------------
* The output inherits the grid and CRS of the input DEM (EPSG:3067 /
  TM35FIN, 2 m pixels, when the source is Finnish KM2 data). The km2
  stream threshold assumes a projected CRS with metre units.
* HAND is a height difference in the vertical datum of the source DEM
  (N2000 for KM2); any single consistent metric datum works.

Credits
-------
* Source data: filled DEM and D8 rasters from the earlier pipeline stages,
  derived from a 2 m digital elevation model in EPSG:3067 / N2000 - presumed
  to be the National Land Survey of Finland (Maanmittauslaitos) 2 m
  elevation model (KM2), licensed CC BY 4.0. Edit
  ``SOURCE_DATA_CREDIT_PRESUMED`` in pipeline_io.py if the provenance
  differs.
* Concept: Nobre et al. (2016), the HAND terrain descriptor. Full
  reference in ``HAND_CITATION`` below.
* Tools that enabled this work: Python, NumPy (Harris et al., 2020),
  Numba (Lam, Pitrou and Seibert, 2015), pyflwdir (Eilander et al., 2021),
  rasterio (Gillies et al.) on GDAL (GDAL/OGR contributors, OSGeo).

Usage (inside the ``water`` conda environment, ``conda activate water``)
-----
    python 05_hand.py                     # defaults from USER SETTINGS below
    python 05_hand.py --upa-min 0.5       # coarser stream network
    python 05_hand.py --d8 other/fdir.tif --uparea other/acc.tif
"""

from __future__ import annotations

# ===========================================================================
# USER SETTINGS - these feed the argparse defaults, so they apply to a
#                 no-argument run; any CLI flag overrides them
# ===========================================================================

INPUTS_DIR = "data/02_filled"       # filled DEM tiles; relative paths are
OUTPUTS_DIR = "data/05_hand"        # resolved next to this script
DEM_FILES = None        # None = mosaic all *.tif in INPUTS_DIR, or a list,
                        # e.g. ["filled_carved_L4142E.tif"]

D8_RASTER = "data/03_flows/flow_direction_d8.tif"
                        # D8 flow directions (03_flow_router.py output)
UPAREA_RASTER = "data/04_accumulation/flow_accumulation_d8.tif"
                        # D8 flow accumulation (04_flow_accumulation.py
                        # output); its m2/pixels units tag is honoured

# The stream-initiation threshold (--upa-min default) is the shared
# UPA_MIN constant in pipeline_io.py, common to stages 3, 5 and 6.

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

HAND_CITATION = (
    "Nobre, A.D., Cuartas, L.A., Momo, M.R., Severo, D.L., Pinheiro, A. and "
    "Nobre, C.A. (2016) 'HAND contour: a new proxy predictor of inundation "
    "extent', Hydrological Processes, 30(2), pp. 320-333, "
    "doi:10.1002/hyp.10581 (HAND originally: Rennó et al., 2008, "
    "doi:10.1016/j.rse.2008.03.018)."
)

HAND_DEVIATION = (
    "Pixels whose D8 flow path ends in a pit or at the grid edge without "
    "meeting a drain pixel are nodata (-9999); pyflwdir's "
    "height_above_nearest_drain assigns height-above-pit there instead."
)


# ---------------------------------------------------------------------------
# HAND kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _hand_kernel(idxs_ds, seq, drain, elevtn):
    """Height above nearest drain, down- to upstream in one pass.

    Adapted from :func:`pyflwdir.dem.height_above_nearest_drain` (Nobre et
    al., 2016). ``seq`` orders the valid pixels from down- to upstream, so a
    pixel's downstream HAND is always resolved before the pixel itself. One
    deviation from pyflwdir: a pixel only gets a value if it is a drain or
    its downstream pixel already has one, so paths that end in a pit or at
    the grid edge without meeting a drain stay nodata instead of reporting
    height-above-pit.
    """
    hand = np.full(elevtn.size, NODATA, dtype=np.float32)
    for i in range(seq.size):
        idx0 = seq[i]
        if drain[idx0]:
            hand[idx0] = 0.0
        else:
            idx_ds = idxs_ds[idx0]
            if idx_ds != idx0 and hand[idx_ds] != NODATA:
                hand[idx0] = hand[idx_ds] + (elevtn[idx0] - elevtn[idx_ds])
    return hand


def compute_hand(flw, drain, elevtn, nodata=NODATA):
    """Return HAND [m] on the filled DEM; nodata kept where the DEM has it."""
    hand = _hand_kernel(
        flw.idxs_ds, flw.idxs_seq, drain.ravel(), elevtn.ravel()
    ).reshape(flw.shape)
    hand[elevtn == nodata] = nodata
    return hand


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    # The USER SETTINGS block at the top of the script feeds the argparse
    # defaults directly, so there is exactly one source of truth per value.
    ap = argparse.ArgumentParser(
        description="Height above nearest drain (HAND) from the pipeline's "
                    "filled DEM, D8 flow directions and D8 flow accumulation.")
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
                    help="stream-initiation threshold: pixels with at least "
                         f"this upstream area are drains (default {UPA_MIN:g})")
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
    print(f"mosaic {shape[1]} x {shape[0]} pixels, "
          f"pixel {abs(transform.a)} x {abs(transform.e)} m")

    d8u8, routing_alg = load_d8(args.d8, transform, shape, crs)
    flw = build_flwdir(d8u8, transform)
    del d8u8

    uparea = load_uparea(args.uparea, transform, shape, crs, routing_alg)
    drain = uparea >= np.float32(args.upa_min)
    n_drain = int(drain.sum())
    print(f"stream pixels: {n_drain} (upstream area >= {args.upa_min:g} km2, "
          f"max {float(uparea.max()):.2f} km2)")
    if n_drain == 0:
        sys.exit(f"no pixel reaches --upa-min {args.upa_min:g} km2; "
                 f"nothing to measure HAND against")
    del uparea

    hand = compute_hand(flw, drain, elevtn)
    no_drain = int(((hand == NODATA) & (elevtn != NODATA)).sum())
    del drain

    out_path = write_raster(
        args.outputs_dir / "hand.tif", hand, transform, crs,
        nodata=NODATA, dtype="float32",
        tags=dict(
            title="Height above nearest drain (HAND)",
            algorithm="HAND on D8 flow paths (down- to upstream propagation "
                      "of the elevation difference to the drained stream "
                      "pixel)",
            citation=HAND_CITATION,
            parameters=f"upa_min={args.upa_min:g} km2 (stream-initiation "
                       f"threshold on the D8 flow accumulation)",
            flow_routing_algorithm=routing_alg,
            stream_threshold_km2=f"{args.upa_min:g}",
            units="m above the nearest drain pixel along the D8 flow path",
            deviation=HAND_DEVIATION,
            source_dem_tiles=", ".join(p.name for p in dem_paths),
            source_flow_direction_raster=args.d8.name,
            source_flow_accumulation_raster=args.uparea.name,
            source_data_credit=(SOURCE_DATA_CREDIT_KNOWN
                                if "dem_source_tiles" in forwarded
                                else SOURCE_DATA_CREDIT_PRESUMED),
            software_credits=TOOL_CREDITS_PYFLWDIR,
            generated_by="05_hand.py",
            **forwarded,
        ),
    )

    valid = hand[hand != NODATA]
    print(f"HAND median: {float(np.median(valid)):.2f} m over "
          f"{valid.size} pixels; {no_drain} pixels drain to a pit/edge "
          f"without meeting a stream -> nodata")
    print(f"-> {out_path}  ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
