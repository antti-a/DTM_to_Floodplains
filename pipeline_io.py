#!/usr/bin/env python3
"""pipeline_io.py - the pipeline's shared I/O layer, companion to stages 3-6.

Created on Tue Jul 14 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Companion module (unnumbered, like accumulation.py and mdinf.py, because
module names starting with a digit cannot be imported). The stage scripts
used to carry private copies of this plumbing; a fix or a convention
change now lands here once instead of three times. Nothing in this file
computes hydrology - it only knows the pipeline's file and metadata
conventions.

WHAT LIVES HERE
    find_dems           resolve DEM tile paths (CLI override, else a folder)
    validate_tiles      require one CRS / cell size / grid lattice
    collect_provenance  merge the dem_* provenance tags stamped by stages 1-2
    build_mosaic        merge the tiles in memory -> (float32, transform, crs)
                        (stage 3 keeps its own on-disk float64 mosaic: pysheds
                        reads a file, and routing needs the flat-fix gradients)
    check_grid          fail hard when a raster is off the mosaic's grid
    load_d8             flow_direction_d8.tif -> pyflwdir's uint8 codes
    build_flwdir        D8 codes -> pyflwdir flow graph
    load_uparea         stage-4 accumulation raster -> upstream area in km2
    swap_in             lock-tolerant os.replace for finished outputs
    write_raster        tagged single-band GeoTIFF (tiled, compressed,
                        written via swap_in)
    resolve_near        resolve a USER SETTINGS path next to a script

    NODATA              the DEM nodata used through stages 1-3 (-9999.0)
    D8_*_PYFLWDIR       pyflwdir's D8 special codes
    SOURCE_DATA_CREDIT_KNOWN / _PRESUMED, TOOL_CREDITS[_PYFLWDIR]
                        credit strings for the output tags
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NODATA = -9999.0        # DEM nodata in stages 1-3, reused for hand.tif

# pyflwdir's uint8 D8 convention (pyflwdir.core_d8): the direction codes are
# the same ESRI codes 03_flow_router.py writes; only the specials differ.
D8_PIT_PYFLWDIR = 0     # our -1 (flat) and -2 (pit) become pits = outlets
D8_NODATA_PYFLWDIR = 247

# Used when the inputs carry no stage-1 provenance tags (pre-convention).
SOURCE_DATA_CREDIT_PRESUMED = (
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
    "rasterio (Gillies et al.), GDAL (GDAL/OGR contributors, OSGeo)."
)

TOOL_CREDITS_PYFLWDIR = (
    "Python, NumPy (Harris et al., 2020, doi:10.1038/s41586-020-2649-2), "
    "Numba (Lam, Pitrou and Seibert, 2015, doi:10.1145/2833157.2833162), "
    "pyflwdir (Eilander et al., 2021, doi:10.5194/hess-25-5287-2021), "
    "rasterio (Gillies et al.), GDAL (GDAL/OGR contributors, OSGeo)."
)


# ---------------------------------------------------------------------------
# DEM tiles in: discovery, validation, provenance, mosaic
# ---------------------------------------------------------------------------

def find_dems(dem_args, dem_dir):
    """Resolve the DEM tile paths: --dem wins, else every GeoTIFF in dem_dir."""
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
              "inputs); outputs will not name the source DEM tiles.")
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

    Unlike 03_flow_router.py this never writes a mosaic file - nothing in
    stages 5-6 needs one on disk. The float64 flat-fix gradients of the
    filled tiles only matter for *routing*, which stage 3 already did, so
    the elevations are cast to float32 (elevation differences stay good to
    well under a mm).
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


# ---------------------------------------------------------------------------
# Pipeline rasters in: D8 directions and flow accumulation
# ---------------------------------------------------------------------------

def check_grid(what, path, src, transform, shape, crs):
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
        check_grid("the D8 raster", d8_path, src, transform, shape, crs)
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
    import pyflwdir  # only stages 5-6 need it; keep stages 3-4 lighter
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
        check_grid("the accumulation raster", uparea_path, src,
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
# Outputs: lock-tolerant swap-in, tagged GeoTIFF writer
# ---------------------------------------------------------------------------

def swap_in(tmp, path):
    """Move a finished temp file onto its final name; return the real path.

    Overwriting a file that a viewer, Excel or a sync tool still holds
    open fails on Windows (OSError 22/13), and a long run must not die
    on its final writes - so every output is written to a temp name and
    swapped in, falling back to a numbered sibling when the target is
    locked.
    """
    try:
        os.replace(tmp, path)
        return path
    except OSError:
        for i in range(1, 100):
            alt = path.with_name(f"{path.stem}_{i}{path.suffix}")
            try:
                os.replace(tmp, alt)
            except OSError:
                continue
            print(f"Note: {path.name} is held open by another program; "
                  f"wrote {alt.name} instead.")
            return alt
        raise


def write_raster(path, array, transform, crs, nodata, dtype, tags):
    """Write a single-band GeoTIFF (tiled + compressed) with metadata tags.

    Written to a temp name and swapped in (see swap_in); returns the path
    actually written.
    """
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
    tmp = path.with_name(path.stem + ".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        dst.update_tags(**tags)
    return swap_in(tmp, path)


def resolve_near(p, here):
    """Resolve a USER SETTINGS path: relative paths live next to the script."""
    p = Path(p)
    return p if p.is_absolute() else here / p
