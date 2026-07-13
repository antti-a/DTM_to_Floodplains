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
1. The filled DEM tiles are mosaicked in memory (same tile validation as
   03_flow_router.py; no mosaic file is written).
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
  elevation model (KM2), licensed CC BY 4.0. Edit ``SOURCE_DATA_CREDIT``
  below if the provenance differs.
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
# USER SETTINGS - the defaults; used when the script is run with no
#                 command-line arguments, overridden by any CLI flag
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

UPA_MIN = 0.2           # km2; stream-initiation threshold - cells with at
                        # least this much upstream area are stream cells

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
import pyflwdir
import rasterio
from numba import njit
from rasterio.merge import merge as rio_merge

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODATA = -9999.0        # DEM nodata in stages 1-3

# pyflwdir's uint8 D8 convention (pyflwdir.core_d8): the direction codes are
# the same ESRI codes 03_flow_router.py writes; only the specials differ.
D8_PIT_PYFLWDIR = 0     # our -1 (flat) and -2 (pit) become pits = outlets
D8_NODATA_PYFLWDIR = 247

FP_NODATA = -1          # floodplain raster: 1 floodplain, 0 upland, -1 nodata

SOURCE_DATA_CREDIT = (
    "Filled DEM, D8 flow-direction and D8 flow-accumulation rasters from "
    "the earlier pipeline stages, derived from a 2 m DEM in EPSG:3067 "
    "(ETRS89 / TM35FIN), vertical datum N2000; presumed source: National "
    "Land Survey of Finland 2 m elevation model (KM2), CC BY 4.0."
)

TOOL_CREDITS = (
    "Python, NumPy (Harris et al., 2020, doi:10.1038/s41586-020-2649-2), "
    "Numba (Lam, Pitrou and Seibert, 2015, doi:10.1145/2833157.2833162), "
    "pyflwdir (Eilander et al., 2021, doi:10.5194/hess-25-5287-2021), "
    "rasterio (Gillies et al.), GDAL (GDAL/OGR contributors, OSGeo)."
)

GFPLAIN_CITATION = (
    "Nardi, F., Annis, A., Di Baldassarre, G., Vivoni, E.R. and Grimaldi, S. "
    "(2019) 'GFPLAIN250m, a global high-resolution dataset of Earth's "
    "floodplains', Scientific Data, 6, 180309, doi:10.1038/sdata.2018.309."
)


# ---------------------------------------------------------------------------
# Input helpers (same tile conventions as 03_flow_router.py)
# ---------------------------------------------------------------------------

def find_dems(dem_args, dem_dir):
    """Resolve the DEM tile paths: --dem wins, else every GeoTIFF in inputs."""
    if dem_args:
        paths = [Path(p) for p in dem_args]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            sys.exit("DEM not found: " + ", ".join(str(p) for p in missing))
        return paths
    paths = sorted(dem_dir.glob("*.tif")) + sorted(dem_dir.glob("*.tiff"))
    if not paths:
        sys.exit(f"No GeoTIFF found in {dem_dir} (and no --dem given). "
                 f"Run 02_fill_dem.py first.")
    return paths


def validate_tiles(paths):
    """Require one shared CRS, cell size and grid lattice across the tiles.

    The tiles are merged and processed as one surface, so a tile on a
    shifted grid or in another CRS would be silently resampled - fail
    instead. Overlapping tiles are fine for a mosaic (last one wins) but
    worth a note.
    """
    infos = []
    for path in paths:
        with rasterio.open(path) as src:
            infos.append((path.name, src.crs, src.transform, src.bounds))

    name0, crs0, t0, _ = infos[0]
    res0 = (abs(t0.a), abs(t0.e))
    for name, crs, t, _ in infos[1:]:
        if crs != crs0:
            sys.exit(f"{name}: CRS {crs} != {crs0} ({name0})")
        if (abs(t.a), abs(t.e)) != res0:
            sys.exit(f"{name}: cell size {(abs(t.a), abs(t.e))} != {res0} ({name0})")
        dx = (t.c - t0.c) / t.a
        dy = (t.f - t0.f) / t.e
        if abs(dx - round(dx)) > 1e-6 or abs(dy - round(dy)) > 1e-6:
            sys.exit(f"{name}: grid origin misaligned with {name0} by "
                     f"({dx % 1:.6f}, {dy % 1:.6f}) cells")

    for i, (name_a, _, _, ba) in enumerate(infos):
        for name_b, _, _, bb in infos[i + 1:]:
            if (ba.left < bb.right and bb.left < ba.right
                    and ba.bottom < bb.top and bb.bottom < ba.top):
                print(f"Note: tiles {name_a} and {name_b} overlap; "
                      f"the mosaic keeps the later one there.")


def build_mosaic(paths, nodata=NODATA):
    """Merge the tiles in memory; return (float32 elevation, transform, crs).

    Unlike 03_flow_router.py this never writes a mosaic file - nothing here
    needs one on disk. The float64 flat-fix gradients of the filled tiles
    only matter for *routing*, which stage 3 already did, so the elevations
    are cast to float32 (the floodplain height test stays good to well
    under a mm).
    """
    sources = [rasterio.open(p) for p in paths]
    try:
        if all(s.nodata is not None for s in sources):
            nodata = sources[0].nodata
        mosaic, transform = rio_merge(sources, nodata=nodata)
        crs = sources[0].crs
    finally:
        for s in sources:
            s.close()
    band = mosaic[0]
    band[~np.isfinite(band)] = nodata
    elevtn = band.astype(np.float32)
    del mosaic, band
    return elevtn, transform, crs


def _check_grid(what, path, src, transform, shape, crs):
    """Fail hard when a raster is not on the DEM mosaic's grid."""
    if src.crs != crs:
        sys.exit(f"{path}: CRS {src.crs} != DEM mosaic CRS {crs}")
    if (src.height, src.width) != shape:
        sys.exit(f"{path}: grid {src.width} x {src.height} != DEM mosaic "
                 f"{shape[1]} x {shape[0]}; {what} and the DEM tiles must "
                 f"come from the same pipeline run")
    if not src.transform.almost_equals(transform, precision=1e-6):
        sys.exit(f"{path}: transform {src.transform} != DEM mosaic "
                 f"transform {transform}")


def load_d8(d8_path, transform, shape, crs):
    """Read flow_direction_d8.tif and remap it to pyflwdir's uint8 codes.

    The direction codes (64=N 128=NE 1=E 2=SE 4=S 8=SW 16=W 32=NW) are
    already pyflwdir's; only the specials move: -1 (flat) and -2 (pit)
    become pyflwdir pits (0, i.e. outlets) and 0 (nodata) becomes 247.
    """
    d8_path = Path(d8_path)
    if not d8_path.is_file():
        sys.exit(f"D8 flow-direction raster not found: {d8_path}. "
                 f"Run 03_flow_router.py first.")
    with rasterio.open(d8_path) as src:
        _check_grid("the D8 raster", d8_path, src, transform, shape, crs)
        codes = src.read(1)
    d8u8 = np.where(codes == 0, D8_NODATA_PYFLWDIR,
                    np.where(codes < 0, D8_PIT_PYFLWDIR,
                             codes)).astype(np.uint8)
    del codes
    return d8u8


def build_flwdir(d8u8, transform):
    """Existing D8 raster -> pyflwdir flow graph (no routing recomputed)."""
    return pyflwdir.from_array(
        d8u8, ftype="d8", check_ftype=True, transform=transform, latlon=False
    )


def load_uparea(uparea_path, transform, shape, crs):
    """Read the stage-4 accumulation raster; return upstream area in km2.

    The units tag written by 04_flow_accumulation.py decides the conversion
    (m2 or cell counts); NaN (nodata) becomes 0, which can never reach the
    stream threshold.
    """
    uparea_path = Path(uparea_path)
    if not uparea_path.is_file():
        sys.exit(f"Flow-accumulation raster not found: {uparea_path}. "
                 f"Run 04_flow_accumulation.py first.")
    with rasterio.open(uparea_path) as src:
        _check_grid("the accumulation raster", uparea_path, src,
                    transform, shape, crs)
        units = src.tags().get("units", "")
        acc = src.read(1)
    if units.startswith("m2"):
        factor = 1e-6
    elif units.startswith("cells"):
        factor = abs(transform.a * transform.e) / 1e6
    else:
        sys.exit(f"{uparea_path}: cannot tell m2 from cell counts - the "
                 f"'units' tag is {units!r}; expected a "
                 f"04_flow_accumulation.py output")
    uparea = np.nan_to_num(acc, nan=0.0).astype(np.float32)
    uparea *= np.float32(factor)
    del acc
    return uparea


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
# Output helper
# ---------------------------------------------------------------------------

def write_raster(path, array, transform, crs, nodata, dtype, tags):
    """Write a single-band GeoTIFF (tiled + compressed) with metadata tags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # predictor 3 = floating-point, 2 = integer horizontal differencing
    predictor = 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2
    profile = dict(
        driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs=crs, transform=transform, nodata=nodata,
        tiled=True, blockxsize=256, blockysize=256, compress="deflate",
        predictor=predictor, bigtiff="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        dst.update_tags(**tags)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def settings_to_argv() -> list[str]:
    """Translate the USER SETTINGS block into command-line arguments
    (used when the script is run without arguments)."""
    here = Path(__file__).resolve().parent

    def resolve(p):
        p = Path(p)
        return str(p if p.is_absolute() else here / p)

    argv = ["--inputs-dir", resolve(INPUTS_DIR),
            "--outputs-dir", resolve(OUTPUTS_DIR),
            "--d8", resolve(D8_RASTER),
            "--uparea", resolve(UPAREA_RASTER),
            "--upa-min", str(UPA_MIN),
            "--a", str(GFPLAIN_A),
            "--b", str(FLOODPLAIN_B)]
    for f in DEM_FILES or []:
        argv += ["--dem", resolve(Path(INPUTS_DIR) / f)]
    return argv


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="GFPLAIN geomorphic floodplains (h = a*A**b) from the "
                    "pipeline's filled DEM, D8 flow directions and D8 flow "
                    "accumulation.")
    ap.add_argument("--dem", nargs="+", default=None, metavar="TIF",
                    help="filled DEM tiles (default: all in --inputs-dir)")
    ap.add_argument("--inputs-dir", type=Path,
                    default=here / "data" / "02_filled")
    ap.add_argument("--outputs-dir", type=Path,
                    default=here / "data" / "06_floodplains")
    ap.add_argument("--d8", type=Path,
                    default=here / "data" / "03_flows" / "flow_direction_d8.tif",
                    help="D8 flow-direction raster (03_flow_router.py output)")
    ap.add_argument("--uparea", type=Path,
                    default=(here / "data" / "04_accumulation"
                             / "flow_accumulation_d8.tif"),
                    help="D8 flow-accumulation raster "
                         "(04_flow_accumulation.py output)")
    ap.add_argument("--upa-min", type=float, default=0.2, metavar="KM2",
                    help="stream-initiation threshold: cells with at least "
                         "this upstream area are stream cells (default 0.2)")
    ap.add_argument("--a", type=float, default=0.63,
                    help="GFPLAIN coefficient a in h = a*A**b (default 0.63; "
                         "calibrate against flood hazard maps)")
    ap.add_argument("--b", type=float, default=0.3,
                    help="GFPLAIN exponent b in h = a*A**b (default 0.3, "
                         "after Nardi et al., 2019)")
    if argv is None and len(sys.argv) <= 1:
        argv = settings_to_argv()  # no CLI arguments: use the USER SETTINGS
        print("running with the USER SETTINGS from the top of the script")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    dem_paths = find_dems(args.dem, args.inputs_dir)
    print(f"DEM tiles ({len(dem_paths)}): "
          + ", ".join(p.name for p in dem_paths))
    validate_tiles(dem_paths)

    elevtn, transform, crs = build_mosaic(dem_paths)
    shape = elevtn.shape
    print(f"mosaic {shape[1]} x {shape[0]} cells, "
          f"cell {abs(transform.a)} x {abs(transform.e)} m")

    d8u8 = load_d8(args.d8, transform, shape, crs)
    flw = build_flwdir(d8u8, transform)
    del d8u8

    uparea = load_uparea(args.uparea, transform, shape, crs)
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
            class_encoding="1 = floodplain, 0 = upland, -1 = nodata",
            source_dem_tiles=", ".join(p.name for p in dem_paths),
            source_flow_direction_raster=args.d8.name,
            source_flow_accumulation_raster=args.uparea.name,
            source_data_credit=SOURCE_DATA_CREDIT,
            horizontal_crs="EPSG:3067 (ETRS89 / TM35FIN), units metres",
            vertical_datum="N2000, units metres (datum of the source DEM)",
            software_credits=TOOL_CREDITS,
            generated_by="06_floodplains.py",
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
