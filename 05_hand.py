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

HAND (Nobre et al., 2016) is the vertical distance between a cell and the
stream cell it drains to along the D8 flow path: the local flood-relevant
"height above the river". This stage recomputes nothing the pipeline already
produced - the filled DEM, the D8 flow directions and the D8 flow
accumulation are read as-is; pyflwdir is used only to turn the existing D8
raster into a flow graph (one downstream index per cell plus a down-to-
upstream cell ordering).

How it works
------------
1. The filled DEM tiles are mosaicked in memory (same tile validation as
   03_flow_router.py; no mosaic file is written).
2. ``flow_direction_d8.tif`` is remapped to pyflwdir's uint8 convention
   (identical ESRI direction codes; -1 flat / -2 pit -> 0 = pit,
   0 nodata -> 247) and loaded with ``pyflwdir.from_array(ftype="d8")``.
3. Stream (drain) cells are those whose stage-4 D8 flow accumulation,
   converted to km2, reaches the stream-initiation threshold ``upa-min``.
4. A Numba kernel walks the flow graph from down- to upstream: a drain cell
   gets HAND = 0, every other cell gets its downstream neighbour's HAND plus
   the elevation difference to it.

The kernel is adapted from :func:`pyflwdir.dem.height_above_nearest_drain`
(plain, uncompiled Python in pyflwdir <= 0.5.11, far too slow for a 36M-cell
grid) with one deliberate deviation: pyflwdir assigns height-above-*pit* to
cells whose flow path ends in a pit or at the grid edge without ever meeting
a drain; on a whole-DEM run that would fill every small edge catchment below
the threshold with misleading values, so those cells are written as nodata
instead.

Output
------
``data/05_hand/hand.tif`` - float32, deflate-compressed GeoTIFF. Values are
metres above the nearest drain cell along the D8 flow path; drain cells are
0. NoData (-9999) marks cells outside the DEM and cells that drain to a
pit/edge without meeting a stream.

Spatial reference
-----------------
* Horizontal: EPSG:3067 (ETRS89 / TM35FIN), units metres, 2 m cells.
* Vertical: N2000, units metres (HAND is a height difference in the same
  vertical datum as the source DEM).

Credits
-------
* Source data: filled DEM and D8 rasters from the earlier pipeline stages,
  derived from a 2 m digital elevation model in EPSG:3067 / N2000 - presumed
  to be the National Land Survey of Finland (Maanmittauslaitos) 2 m
  elevation model (KM2), licensed CC BY 4.0. Edit ``SOURCE_DATA_CREDIT``
  below if the provenance differs.
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
# USER SETTINGS - the defaults; used when the script is run with no
#                 command-line arguments, overridden by any CLI flag
# ===========================================================================

INPUTS_DIR = "data/02_filled"       # filled DEM tiles; relative paths are
OUTPUTS_DIR = "data/05_hand"        # resolved next to this script
DEM_FILES = None        # None = mosaic all *.tif in INPUTS_DIR, or a list,
                        # e.g. ["filled_carved_L4142E.tif"]

D8_RASTER = "data/03_flows/flow_direction_d8.tif"
                        # D8 flow directions (03_flow_router.py output)
UPAREA_RASTER = "data/04_accumulation/flow_accumulation_d8.tif"
                        # D8 flow accumulation (04_flow_accumulation.py
                        # output); its m2/cells units tag is honoured

UPA_MIN = 0.2           # km2; stream-initiation threshold - cells with at
                        # least this much upstream area are drains (HAND = 0)

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

NODATA = -9999.0        # DEM nodata in stages 1-3, reused for the HAND raster

# pyflwdir's uint8 D8 convention (pyflwdir.core_d8): the direction codes are
# the same ESRI codes 03_flow_router.py writes; only the specials differ.
D8_PIT_PYFLWDIR = 0     # our -1 (flat) and -2 (pit) become pits = outlets
D8_NODATA_PYFLWDIR = 247

SOURCE_DATA_CREDIT = (
    "Filled DEM, D8 flow-direction and D8 flow-accumulation rasters from "
    "the earlier pipeline stages, derived from a 2 m DEM in EPSG:3067 "
    "(ETRS89 / TM35FIN), vertical datum N2000; presumed source: National "
    "Land Survey of Finland 2 m elevation model (KM2), CC BY 4.0."
)

# Used instead when the inputs forward stage-1 provenance tags.
SOURCE_DATA_CREDIT_KNOWN = (
    "National Land Survey of Finland 2 m elevation model (KM2), CC BY 4.0, "
    "carved with the SYKE 'Tierumpujen uomakorjaus' culvert correction "
    "(CC BY 4.0); source tiles in the dem_source_tiles tag."
)

TOOL_CREDITS = (
    "Python, NumPy (Harris et al., 2020, doi:10.1038/s41586-020-2649-2), "
    "Numba (Lam, Pitrou and Seibert, 2015, doi:10.1145/2833157.2833162), "
    "pyflwdir (Eilander et al., 2021, doi:10.5194/hess-25-5287-2021), "
    "rasterio (Gillies et al.), GDAL (GDAL/OGR contributors, OSGeo)."
)

HAND_CITATION = (
    "Nobre, A.D., Cuartas, L.A., Momo, M.R., Severo, D.L., Pinheiro, A. and "
    "Nobre, C.A. (2016) 'HAND contour: a new proxy predictor of inundation "
    "extent', Hydrological Processes, 30(2), pp. 320-333, "
    "doi:10.1002/hyp.10581 (HAND originally: Rennó et al., 2008, "
    "doi:10.1016/j.rse.2008.03.018)."
)

HAND_DEVIATION = (
    "Cells whose D8 flow path ends in a pit or at the grid edge without "
    "meeting a drain cell are nodata (-9999); pyflwdir's "
    "height_above_nearest_drain assigns height-above-pit there instead."
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


def collect_provenance(paths):
    """Merge the tiles' provenance tags (stamped by stages 1-2).

    ``dem_source_tiles`` becomes the sorted union across the tiles;
    ``dem_carve`` / ``dem_fill`` are forwarded only when every tagged tile
    agrees. Untagged (pre-convention) tiles only cost a note, never a
    failure, so old intermediates keep working.
    """
    tile_tags = []
    for path in paths:
        with rasterio.open(path) as src:
            tile_tags.append(src.tags())

    merged = {}
    sources = set()
    for tags in tile_tags:
        sources.update(
            t for t in tags.get("dem_source_tiles", "").split(", ") if t
        )
    if sources:
        merged["dem_source_tiles"] = ", ".join(sorted(sources))
    else:
        print("Note: the tiles carry no provenance tags (pre-convention "
              "inputs); recording the presumed source only.")
    for key in ("dem_carve", "dem_fill"):
        values = {tags[key] for tags in tile_tags if key in tags}
        if len(values) == 1:
            merged[key] = values.pop()
        elif len(values) > 1:
            print(f"Note: the tiles disagree on the {key} tag "
                  f"({', '.join(sorted(values))}); tag omitted.")
    return merged


def build_mosaic(paths, nodata=NODATA):
    """Merge the tiles in memory; return (float32 elevation, transform, crs).

    Unlike 03_flow_router.py this never writes a mosaic file - nothing here
    needs one on disk. The float64 flat-fix gradients of the filled tiles
    only matter for *routing*, which stage 3 already did, so the elevations
    are cast to float32 (HAND differences stay good to well under a mm).
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
    Returns ``(d8u8, routing_alg)``; the routing algorithm comes from the
    raster's ``flow_routing_algorithm`` tag ("d8" when the tag is missing,
    hard error when it names another algorithm - the codes would be
    garbage).
    """
    d8_path = Path(d8_path)
    if not d8_path.is_file():
        sys.exit(f"D8 flow-direction raster not found: {d8_path}. "
                 f"Run 03_flow_router.py first.")
    with rasterio.open(d8_path) as src:
        _check_grid("the D8 raster", d8_path, src, transform, shape, crs)
        codes = src.read(1)
        routing_alg = src.tags().get("flow_routing_algorithm")
    if routing_alg is None:
        print(f"WARNING: {d8_path.name} carries no flow_routing_algorithm "
              f"tag (pre-convention raster); assuming d8")
        routing_alg = "d8"
    elif routing_alg != "d8":
        sys.exit(f"{d8_path}: flow_routing_algorithm tag says "
                 f"{routing_alg!r}; this stage needs a D8 code raster "
                 f"(flow_direction_d8.tif from 03_flow_router.py)")
    d8u8 = np.where(codes == 0, D8_NODATA_PYFLWDIR,
                    np.where(codes < 0, D8_PIT_PYFLWDIR,
                             codes)).astype(np.uint8)
    del codes
    return d8u8, routing_alg


def build_flwdir(d8u8, transform):
    """Existing D8 raster -> pyflwdir flow graph (no routing recomputed)."""
    return pyflwdir.from_array(
        d8u8, ftype="d8", check_ftype=True, transform=transform, latlon=False
    )


def load_uparea(uparea_path, transform, shape, crs, routing_alg=None):
    """Read the stage-4 accumulation raster; return upstream area in km2.

    The units tag written by 04_flow_accumulation.py decides the conversion
    (m2 or cell counts); NaN (nodata) becomes 0, which can never reach the
    stream threshold. When ``routing_alg`` is given, the raster's own
    ``flow_routing_algorithm`` tag is cross-checked against it (warning
    only - the flow-direction raster's tag wins).
    """
    uparea_path = Path(uparea_path)
    if not uparea_path.is_file():
        sys.exit(f"Flow-accumulation raster not found: {uparea_path}. "
                 f"Run 04_flow_accumulation.py first.")
    with rasterio.open(uparea_path) as src:
        _check_grid("the accumulation raster", uparea_path, src,
                    transform, shape, crs)
        units = src.tags().get("units", "")
        uparea_alg = src.tags().get("flow_routing_algorithm")
        acc = src.read(1)
    if routing_alg is not None:
        if uparea_alg is None:
            print(f"WARNING: {uparea_path.name} carries no "
                  f"flow_routing_algorithm tag; cannot cross-check it "
                  f"against the flow-direction raster ({routing_alg})")
        elif uparea_alg != routing_alg:
            print(f"WARNING: {uparea_path.name} says flow_routing_algorithm="
                  f"{uparea_alg}, the flow-direction raster says "
                  f"{routing_alg}; recording {routing_alg}")
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
# HAND kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _hand_kernel(idxs_ds, seq, drain, elevtn):
    """Height above nearest drain, down- to upstream in one pass.

    Adapted from :func:`pyflwdir.dem.height_above_nearest_drain` (Nobre et
    al., 2016). ``seq`` orders the valid cells from down- to upstream, so a
    cell's downstream HAND is always resolved before the cell itself. One
    deviation from pyflwdir: a cell only gets a value if it is a drain or
    its downstream cell already has one, so paths that end in a pit or at
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
            "--upa-min", str(UPA_MIN)]
    for f in DEM_FILES or []:
        argv += ["--dem", resolve(Path(INPUTS_DIR) / f)]
    return argv


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Height above nearest drain (HAND) from the pipeline's "
                    "filled DEM, D8 flow directions and D8 flow accumulation.")
    ap.add_argument("--dem", nargs="+", default=None, metavar="TIF",
                    help="filled DEM tiles (default: all in --inputs-dir)")
    ap.add_argument("--inputs-dir", type=Path,
                    default=here / "data" / "02_filled")
    ap.add_argument("--outputs-dir", type=Path,
                    default=here / "data" / "05_hand")
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
                         "this upstream area are drains (default 0.2)")
    if argv is None and len(sys.argv) <= 1:
        argv = settings_to_argv()  # no CLI arguments: use the USER SETTINGS
        print("running with the USER SETTINGS from the top of the script")
    args = ap.parse_args(argv)

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
    drain = uparea >= np.float32(args.upa_min)
    n_drain = int(drain.sum())
    print(f"stream cells: {n_drain} (upstream area >= {args.upa_min:g} km2, "
          f"max {float(uparea.max()):.2f} km2)")
    if n_drain == 0:
        sys.exit(f"no cell reaches --upa-min {args.upa_min:g} km2; "
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
                      "cell)",
            citation=HAND_CITATION,
            parameters=f"upa_min={args.upa_min:g} km2 (stream-initiation "
                       f"threshold on the D8 flow accumulation)",
            flow_routing_algorithm=routing_alg,
            stream_threshold_km2=f"{args.upa_min:g}",
            units="m above the nearest drain cell along the D8 flow path",
            deviation=HAND_DEVIATION,
            source_dem_tiles=", ".join(p.name for p in dem_paths),
            source_flow_direction_raster=args.d8.name,
            source_flow_accumulation_raster=args.uparea.name,
            source_data_credit=(SOURCE_DATA_CREDIT_KNOWN
                                if "dem_source_tiles" in forwarded
                                else SOURCE_DATA_CREDIT),
            horizontal_crs="EPSG:3067 (ETRS89 / TM35FIN), units metres",
            vertical_datum="N2000, units metres (datum of the source DEM)",
            software_credits=TOOL_CREDITS,
            generated_by="05_hand.py",
            **forwarded,
        ),
    )

    valid = hand[hand != NODATA]
    print(f"HAND median: {float(np.median(valid)):.2f} m over "
          f"{valid.size} cells; {no_drain} cells drain to a pit/edge "
          f"without meeting a stream -> nodata")
    print(f"-> {out_path}  ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
