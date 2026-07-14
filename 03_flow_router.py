#!/usr/bin/env python3
r"""
03_flow_router.py - route flow over a DEM, map the streams (D8 by
default; MFD, Dinf and MDinf on request).

Created on Sat Jul 4 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 3 (see README.md):
    reads   data/02_filled/*.tif        (02_fill_dem.py output)
    writes  data/03_flows/              (networks, summary CSV and the
                                         flow_direction_*.tif rasters that
                                         04_flow_accumulation.py reads)

Takes one or more hydrologically conditioned DEM tiles (GeoTIFFs in
data/02_filled), mosaics them into a single surface so streams can cross
tile edges, and runs the selected flow-routing algorithms on it. The
default is D8 alone - the format the rest of the pipeline consumes;
select more with --fdir (e.g. --fdir all). Per selected algorithm it
writes to data/03_flows:

  flow_network_<alg>.geojson   the stream network
  flow_networks_summary.csv    one row per algorithm: stream cells and
                               area, cells unique to that algorithm,
                               segment count and length, drainage
                               density, largest catchment, Jaccard
                               overlap with every other network, runtime

    With more than one algorithm selected the CSV is where they are
    compared: the dispersive methods' unique cells are their braided,
    anastomosing footprint, and the Jaccard overlaps say how much any
    two networks agree. The GeoJSON lines are a D8-path approximation
    (see METHOD).

It also writes each selected algorithm's flow-DIRECTION raster - the
hand-off the downstream pipeline stages read - as GeoTIFFs whose band
descriptions and tags document their encoding:

  flow_direction_d8.tif        one int32 band of direction codes
                               (64=N 128=NE 1=E 2=SE 4=S 8=SW 16=W 32=NW)
  flow_direction_mfd.tif       eight float32 bands: the fraction of each
                               cell's flow sent to its N, NE, E, SE, S,
                               SW, W, NW neighbour
  flow_direction_dinf.tif      one float32 band: flow angle in [0, 2*pi)
                               radians counter-clockwise from east
  flow_direction_mdinf.tif     eight float32 bands of flow fractions, like
                               MFD - needs the companion module mdinf.py

    (in d8 and dinf, -1 marks a flat and -2 a pit)

    flow_direction_mdinf.tif is special: MDinf runs in WhiteboxTools,
    which computes its facet directions internally, returns only the
    accumulation, and has no tool that exports them. The optional
    companion module mdinf.py therefore recomputes the directions with
    the same mathematics (ported from Jan Seibert & Marc Vis's own
    implementation via WhiteboxTools' MIT-licensed source) - it is only
    imported when the mdinf raster is requested, and if the file is not
    next to this script the run prints a note and carries on.

The tuning knobs are constants at the top of the script:

    DEFAULT_MIN_AREA_KM2    minimum catchment (contributing) area that
                            starts a stream - edit it to taste, or
                            override a single run with --area
    DEFAULT_METHODS         which algorithms run when --fdir is not
                            given (just "d8" out of the box)

USAGE
    python 03_flow_router.py                  # D8 only (the default)
    python 03_flow_router.py --area 0.5       # override the stream threshold
    python 03_flow_router.py --fdir all       # run all four algorithms
    python 03_flow_router.py --fdir d8 dinf   # ... or a chosen subset
    python 03_flow_router.py --describe       # print the algorithm definitions
    python 03_flow_router.py --dem a.tif b.tif --out results

SHARING
    The script is a single file with no hard-coded paths: put it next to
    a "dem" folder holding one or more GeoTIFF tiles (same CRS, cell size
    and grid alignment) and run it with any Python >= 3.9 that has the
    dependencies installed. One optional extra: share mdinf.py alongside
    if the MDinf direction raster is wanted (everything else works
    without it; it needs numba, which pysheds installs anyway).

        pip install pysheds whitebox rasterio pyproj numpy
        (or conda -c conda-forge; whitebox fetches its own binary once)

    GeoJSON output is RFC 7946 compliant (coordinates in WGS84), so the
    files drop straight into QGIS, geojson.io, kepler.gl, Leaflet, ...

METHOD
    All input tiles are checked to share one grid lattice and merged into
    one mosaic, which is routed as a single surface - a stream flowing
    off one tile continues onto the next instead of stopping at the seam.
    For every algorithm the contributing area is accumulated with that
    algorithm's own flow partitioning; cells whose contributing area
    reaches the threshold are that algorithm's stream cells. Those masks
    are taken exactly as the accumulation gives them, with no cleaning or
    pruning, and tabulated against each other in the summary CSV - when
    several algorithms are selected, the differences between them are
    the whole point: D8 gives a sparse
    single-threaded tree, while the dispersive methods (MFD, Dinf, MDinf)
    part around subtle highs and rejoin, so their masks carry extra,
    unique cells (braided, anastomosing bands) that the unique-cell
    counts and Jaccard overlaps expose. For the GeoJSON each mask is
    vectorized by connecting its cells along D8 steepest-descent paths;
    a D8 tree can only converge, so the vector lines cannot braid and
    off-tree mask cells drop out - treat the GeoJSON as a line
    approximation and the CSV as the measurement.

    The DEM is assumed hydrologically conditioned by the fill/carve
    pipeline that produced data/02_filled: pits and depressions filled AND flat
    ties resolved (02_fill_dem.py bakes tiny fix_flats gradients into
    float64 outputs - float32 would collapse them back into ties).
    Routing is not conditioning, so none happens here; the script only
    verifies that the mosaic actually drains and stops with a pointer at
    the filling step if it does not. D8, MFD and Dinf run in pysheds;
    MDinf runs in WhiteboxTools (pysheds does not implement it).

THE FOUR ALGORITHMS (precise definitions, and credit where due)

  D8 - "deterministic eight-neighbour", O'Callaghan and Mark (1984)
    Each cell discharges ALL of its flow to exactly one of its eight
    neighbours (4 cardinal, 4 diagonal): the one with the steepest
    downward drop rate (z_cell - z_neighbour) / d, where d is the cell
    size towards cardinal neighbours and cell size * sqrt(2) towards
    diagonal ones. The result is a spanning tree of one-cell-wide flow
    paths; contributing area grows in whole-cell steps. Convergent,
    simple and fast, but it cannot represent divergent flow and imprints
    45-degree artefacts on hillslopes.
      Founder: O'Callaghan, J.F. and Mark, D.M. (1984) 'The extraction of
      drainage networks from digital elevation data', Computer Vision,
      Graphics, and Image Processing, 28(3), pp. 323-344.

  MFD - "multiple flow direction", Quinn et al. (1991)
    Flow from a cell is divided among ALL lower neighbours at once.
    Each downslope neighbour i receives the fraction

        f_i = tan(beta_i)^p / sum_j tan(beta_j)^p

    of the cell's flow, where beta_i is the downward slope towards
    neighbour i and j runs over every downslope neighbour (pysheds uses
    the original slope-proportional weighting; Freeman (1991) proposed
    p = 1.1). Dispersing flow over all descending directions represents
    divergent hillslope flow realistically at the cost of smearing flow
    paths, so accumulation is fractional rather than whole-cell.
      Founder: Quinn, P., Beven, K., Chevallier, P. and Planchon, O. (1991)
      'The prediction of hillslope flow paths for distributed hydrological
      modelling using digital terrain models', Hydrological Processes,
      5(1), pp. 59-79. (Exponent variant: Freeman, T.G. (1991) 'Calculating
      catchment area with divergent flow based on a regular grid',
      Computers & Geosciences, 17(3), pp. 413-422.)

  Dinf - "D-infinity", Tarboton (1997)
    The flow direction is a CONTINUOUS angle in [0, 2*pi), taken as the
    steepest downward slope over the eight planar triangular facets
    formed between the cell centre and each pair of adjacent neighbours.
    Flow is then apportioned to the two grid neighbours bracketing that
    angle, proportionally to how close the angle lies to each; an angle
    aligned exactly with a neighbour sends everything to that single
    cell. This avoids D8's directional artefacts while keeping
    dispersion bounded (at most two receivers per cell).
      Founder: Tarboton, D.G. (1997) 'A new method for the determination
      of flow directions and upslope areas in grid digital elevation
      models', Water Resources Research, 33(2), pp. 309-319.

  MDinf - "multiple direction D-infinity", Seibert and McGlynn (2007)
    A marriage of Dinf and MFD. Like Tarboton's Dinf, the terrain around
    each cell is modelled as eight planar triangular facets, each with a
    continuous aspect angle; but instead of following only the single
    steepest facet, flow is dispersed over ALL downslope facets. Each
    downslope facet receives a share proportional to its slope raised to
    an exponent p (1.1 here, after Freeman), and that facet's share is
    then split between the two grid neighbours bracketing its aspect
    angle, exactly as in Dinf. The result keeps Dinf's sub-grid angular
    precision while representing divergent flow like MFD, without either
    method's blind spot.
      Founder: Seibert, J. and McGlynn, B.L. (2007) 'A new triangular
      multiple flow direction algorithm for computing upslope areas from
      gridded digital elevation models', Water Resources Research, 43(4),
      W04501.
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from pyproj import Transformer
from pysheds.grid import Grid

# Minimum catchment (contributing) area that defines a stream, in km2.
# The one tuning knob: edit it here, or override per run with --area.
DEFAULT_MIN_AREA_KM2 = 1

# Which algorithms run when --fdir is not given. D8 alone is the default:
# it is the format the downstream stages (04, 05, 06) consume. Each
# selected algorithm gets its network GeoJSON, its summary-CSV row and its
# flow-direction raster (flow_direction_*.tif). The formats differ:
# D8 = one band of integer direction codes, Dinf = one band of flow angle
# in radians, MFD and MDinf = eight bands of flow fractions (N, NE, ... NW).
# The MDinf raster needs the optional companion module mdinf.py next to
# this script; without it, the run prints a note and skips the raster.
DEFAULT_METHODS = ("d8",)

MDINF_EXPONENT = 1.1  # facet-slope exponent p for MDinf (Freeman's value)

ALGORITHMS = {
    "d8": {
        "title": "D8",
        "routing": "d8",
        "founder": "O'Callaghan and Mark (1984)",
        "one_liner": "all flow to the single steepest-descent neighbour",
    },
    "mfd": {
        "title": "MFD",
        "routing": "mfd",
        "founder": "Quinn et al. (1991)",
        "one_liner": "flow split across every downslope neighbour, weighted by slope",
    },
    "dinf": {
        "title": "Dinf",
        "routing": "dinf",
        "founder": "Tarboton (1997)",
        "one_liner": "continuous facet angle, flow split between the two bracketing cells",
    },
    "mdinf": {
        "title": "MDinf",
        "routing": "mdinf",  # runs in WhiteboxTools, not pysheds
        "founder": "Seibert and McGlynn (2007)",
        "one_liner": "flow dispersed over all downslope triangular facets, Dinf-style",
    },
}


def find_dems(dem_args, script_dir):
    """Resolve the DEM tile paths: --dem wins, else every GeoTIFF in ./dem."""
    if dem_args:
        paths = [Path(p) for p in dem_args]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            sys.exit("DEM not found: " + ", ".join(str(p) for p in missing))
        return paths
    dem_dir = script_dir / "data" / "02_filled"
    paths = sorted(dem_dir.glob("*.tif")) + sorted(dem_dir.glob("*.tiff"))
    if not paths:
        sys.exit(f"No GeoTIFF found in {dem_dir} (and no --dem given). "
                 f"Run 02_fill_dem.py first.")
    return paths


def validate_tiles(paths):
    """Require one shared CRS, cell size and grid lattice across the tiles.

    The tiles are merged and routed as one surface, so a tile on a shifted
    grid or in another CRS would be silently resampled - fail instead.
    Overlapping tiles are fine for a mosaic (last one wins) but worth a note.
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


def build_mosaic(paths, work_dir, nodata=-9999.0):
    """Merge the tiles into one GeoTIFF and return its path.

    Always materialized, even for a single tile: WhiteboxTools cannot read
    VRTs and rejects the floating-point TIFF predictor the conditioning
    pipeline writes, so a plain (predictor-free) copy is the common ground
    that pysheds, WhiteboxTools and rasterio can all read. Written in
    float64 so the conditioned tiles' sub-millimetre flat-fix gradients
    survive the merge (float32 would collapse them back into ties).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    sources = [rasterio.open(p) for p in paths]
    try:
        if all(s.nodata is not None for s in sources):
            nodata = sources[0].nodata
        mosaic, transform = rio_merge(sources, nodata=nodata)
        profile = dict(
            driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
            count=1, dtype="float64", crs=sources[0].crs, transform=transform,
            nodata=nodata, tiled=True, blockxsize=256, blockysize=256,
            compress="deflate", bigtiff="IF_SAFER",
        )
    finally:
        for s in sources:
            s.close()
    band = mosaic[0].astype("float64")
    band[~np.isfinite(band)] = nodata
    path = work_dir / "mosaic.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(band, 1)
    return path


def metres_per_map_unit(crs, bounds):
    """Metres per CRS unit along (x, y); approximate for lat/lon grids."""
    if crs.is_projected:
        try:
            to_metre = crs.linear_units_factor[1]
        except Exception:
            to_metre = 1.0
        return to_metre, to_metre
    # Geographic CRS: metres per degree at the DEM's mid latitude.
    mid_lat = math.radians((bounds.top + bounds.bottom) / 2.0)
    m_per_deg_lat = 111_320.0
    return m_per_deg_lat * math.cos(mid_lat), m_per_deg_lat


def line_length_m(coords, sx, sy):
    """Length of a coordinate chain in metres (sx, sy = metres per map unit)."""
    xs = np.array([p[0] for p in coords], dtype="float64")
    ys = np.array([p[1] for p in coords], dtype="float64")
    return float(np.hypot(np.diff(xs) * sx, np.diff(ys) * sy).sum())


def get_wbt(work_dir):
    """A quiet WhiteboxTools instance working in work_dir."""
    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_working_dir(str(work_dir))
    return wbt


def check_drainage(mosaic_path):
    """Verify the mosaic drains; conditioning is the filling step's job.

    Counts interior cells with no strictly lower neighbour - a pit or a
    flat tie, where every flow-routing algorithm stops. Nodata counts as
    an outlet and edge cells drain off-grid, so a properly conditioned
    DEM leaves only a scattered handful (outlets), well under 0.1 %.
    Orders of magnitude more means the conditioning did not survive into
    the files - classically a filled DEM saved as float32, which
    collapses the sub-mm fix_flats gradients back into exact ties - and
    the run stops with a pointer at the filling step rather than routing
    on and returning beautiful, empty networks.
    """
    with rasterio.open(mosaic_path) as src:
        z = src.read(1).astype("float64")
        nodata = src.nodata
    valid = np.isfinite(z)
    if nodata is not None:
        valid &= z != nodata
    z[~valid] = -np.inf  # nodata is an outlet: neighbours drain into it
    centre = z[1:-1, 1:-1]
    has_lower = np.zeros(centre.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                has_lower |= (
                    z[1 + dr:z.shape[0] - 1 + dr,
                      1 + dc:z.shape[1] - 1 + dc] < centre
                )
    stuck = int((valid[1:-1, 1:-1] & ~has_lower).sum())
    n_valid = int(valid.sum())
    if stuck > max(1, n_valid // 1000):
        sys.exit(
            f"The DEM does not drain: {stuck} of {n_valid} cells "
            f"({100.0 * stuck / n_valid:.1f} %) are pits or flat ties with no "
            f"lower neighbour, so flow accumulation dies before any stream "
            f"reaches the threshold. Re-run the filling step (02_fill_dem.py: "
            f"FillDepressions with fix_flats=True, float64 output - float32 "
            f"collapses the flat-fix gradients) and route its data/02_filled "
            f"output."
        )
    print(f"         drains: {n_valid - stuck} of {n_valid} valid cells "
          f"({stuck} outlet/tie cells)")


def mdinf_accumulation(conditioned_tif, work_dir):
    """MDinf contributing area (in cells) via WhiteboxTools.

    pysheds has no MDinf, so the conditioned surface is handed to
    WhiteboxTools' MDInfFlowAccumulation (Seibert and McGlynn, 2007) and the
    accumulation grid read back; both tools then see the same elevations.
    Nodata cells come back as 0 so they can never reach the threshold.
    """
    wbt = get_wbt(work_dir)
    out = work_dir / "mdinf_accumulation.tif"
    ret = wbt.md_inf_flow_accumulation(
        str(conditioned_tif), str(out),
        out_type="cells", exponent=MDINF_EXPONENT,
    )
    if ret != 0 or not out.exists():
        raise RuntimeError(f"WhiteboxTools MDInfFlowAccumulation failed (exit {ret})")
    with rasterio.open(out) as src:
        acc = src.read(1).astype("float64")
        if src.nodata is not None:
            acc[acc == src.nodata] = 0.0
    acc[~np.isfinite(acc)] = 0.0
    return acc


def as_grid_raster(values, template):
    """Wrap a bare array in a pysheds Raster on the template's grid."""
    raster = template.astype("float64")  # copies; keeps the viewfinder
    raster[:] = values
    return raster


def build_network(grid, fdir_d8, accumulation, threshold_cells, sx, sy):
    """Threshold accumulation into stream cells, vectorize along D8 paths.

    The mask is taken exactly as the algorithm's own accumulation gives it,
    with no cleaning or pruning - the differences between the masks are the
    whole point. Connecting the cells into LINES follows D8 steepest-descent
    links, and a D8 tree can only converge: braided reaches and mask cells
    lying off the D8 tree cannot be drawn as lines. The GeoJSON is therefore
    a D8-path approximation of the dispersive networks; the returned mask is
    the algorithm's true footprint and is what the summary CSV measures.

    Returns (segments, stream_mask): segments are dicts with native-CRS
    coordinates and length in metres; stream_mask is a boolean grid.
    """
    stream_mask = accumulation >= threshold_cells
    if not np.any(stream_mask):
        return [], np.asarray(stream_mask, dtype=bool)
    network = grid.extract_river_network(fdir_d8, stream_mask)
    segments = []
    for feature in network["features"]:
        coords = feature["geometry"]["coordinates"]
        if len(coords) < 2:  # an isolated cell cannot form a line
            continue
        segments.append(
            {
                "coords": [(float(x), float(y)) for x, y in coords],
                "length_m": line_length_m(coords, sx, sy),
            }
        )
    return segments, np.asarray(stream_mask, dtype=bool)


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


def write_fdir_raster(path, key, spec, fdir, crs, transform,
                      provenance=None):
    """Write one algorithm's flow-direction grid as a GeoTIFF.

    Each algorithm's notion of "direction" is its own, so the files differ:
    d8 is one int32 band of direction codes, dinf one float32 band of flow
    angle in radians, mfd and mdinf eight float32 bands of flow fractions
    in pysheds neighbour order (N, NE, E, SE, S, SW, W, NW). The encoding
    is written into the band descriptions and dataset tags so each file
    explains itself. Returns the path actually written.
    """
    directions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    data = np.asarray(fdir, dtype="float64")
    if key == "d8":
        bands = data.astype("int32")[np.newaxis, :, :]
        nodata = 0
        descriptions = ["D8 direction code"]
        encoding = ("codes 64=N 128=NE 1=E 2=SE 4=S 8=SW 16=W 32=NW; "
                    "-1 = flat, -2 = pit, 0 = nodata")
    elif key == "dinf":
        bands = data.astype("float32")[np.newaxis, :, :]
        nodata = float("nan")
        descriptions = ["Dinf flow angle (radians)"]
        encoding = ("flow angle in [0, 2*pi) radians counter-clockwise "
                    "from east; -1 = flat, -2 = pit, NaN = nodata")
    else:  # mfd and mdinf: eight bands of flow fractions
        bands = data.astype("float32")
        nodata = float("nan")
        descriptions = [f"{spec['title']} flow fraction to {d}"
                        for d in directions]
        encoding = ("band b = fraction of the cell's flow sent to its "
                    f"{'/'.join(directions)} neighbour; NaN = nodata")
        if key == "mdinf":
            encoding += (f"; dispersed over downslope triangular facets with "
                         f"slope exponent {MDINF_EXPONENT:g}, computed by the "
                         f"companion module mdinf.py")
    profile = dict(
        driver="GTiff", height=bands.shape[1], width=bands.shape[2],
        count=bands.shape[0], dtype=bands.dtype.name, crs=crs,
        transform=transform, nodata=nodata, tiled=True, blockxsize=256,
        blockysize=256, compress="deflate", bigtiff="IF_SAFER",
    )
    tags = dict(algorithm=spec["title"], founder=spec["founder"],
                encoding=encoding, flow_routing_algorithm=key,
                **(provenance or {}))
    if key == "mdinf":
        tags["flow_routing_exponent"] = f"{MDINF_EXPONENT:g}"
    tmp = path.with_name(path.stem + ".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(bands)
        dst.update_tags(**tags)
        for b, desc in enumerate(descriptions, start=1):
            dst.set_band_description(b, desc)
    return swap_in(tmp, path)


def write_geojson(path, segments, spec, transformer, dem_name, min_area_km2):
    """Write one network as an RFC 7946 GeoJSON file (WGS84), longest first.

    Returns the path actually written.
    """
    features = []
    for i, seg in enumerate(sorted(segments, key=lambda s: -s["length_m"])):
        lons, lats = transformer.transform(
            [p[0] for p in seg["coords"]], [p[1] for p in seg["coords"]]
        )
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "segment_id": i,
                    "length_m": round(seg["length_m"], 1),
                    "algorithm": spec["title"],
                    "algorithm_founder": spec["founder"],
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(lon, 7), round(lat, 7)] for lon, lat in zip(lons, lats)
                    ],
                },
            }
        )
    collection = {
        "type": "FeatureCollection",
        "name": f"flow_network_{spec['title'].lower()}",
        "description": (
            f"{spec['title']} flow network ({spec['one_liner']}; credit "
            f"{spec['founder']}), extracted from {dem_name}. "
            f"Streams start where the contributing area reaches "
            f"{min_area_km2:g} km2. Line geometry follows D8 steepest-descent "
            f"paths (a line network cannot braid); see "
            f"flow_networks_summary.csv for the algorithm's true, possibly "
            f"anastomosing footprint in numbers. "
            f"Coordinates in WGS84 per RFC 7946."
        ),
        "features": features,
    }
    tmp = path.with_name(path.stem + ".part.geojson")
    tmp.write_text(json.dumps(collection, separators=(",", ": "), indent=None),
                   encoding="utf-8")
    return swap_in(tmp, path)


def summary_rows(masks, networks, stats, area_m2, valid_km2, min_area_km2,
                 threshold_cells, dem_name):
    """One comparison row per algorithm for the summary CSV.

    unique_cells are stream cells no other algorithm claims - for the
    dispersive methods that is their braided, anastomosing footprint.
    overlap_jaccard_* is intersection/union of the stream-cell masks, so
    1.0 means two networks are identical and small values mean they took
    different courses.
    """
    rows = []
    for key in masks:  # the algorithms that actually ran
        spec = ALGORITHMS[key]
        others = np.zeros_like(masks[key])
        for other in masks:
            if other != key:
                others |= masks[other]
        n_cells = int(masks[key].sum())
        unique = int(np.count_nonzero(masks[key] & ~others))
        lengths_m = [seg["length_m"] for seg in networks[key]]
        total_km = sum(lengths_m) / 1000.0
        row = {
            "algorithm": spec["title"],
            "founder": spec["founder"],
            "method": spec["one_liner"],
            "stream_cells": n_cells,
            "stream_area_km2": round(n_cells * area_m2 / 1e6, 3),
            "unique_cells": unique,
            "unique_cells_pct": round(100.0 * unique / max(1, n_cells), 1),
            "segments": len(networks[key]),
            "total_length_km": round(total_km, 1),
            "longest_segment_km": round(max(lengths_m, default=0.0) / 1000.0, 2),
            "drainage_density_km_per_km2": round(total_km / valid_km2, 3),
            "max_catchment_km2": stats[key]["max_catchment_km2"],
        }
        for other in masks:
            inter = int(np.count_nonzero(masks[key] & masks[other]))
            union = int(np.count_nonzero(masks[key] | masks[other]))
            row[f"overlap_jaccard_{other}"] = (
                round(inter / union, 3) if union else 0.0
            )
        row.update(
            runtime_s=stats[key]["runtime_s"],
            min_area_km2=min_area_km2,
            threshold_cells=threshold_cells,
            dem=dem_name,
        )
        rows.append(row)
    return rows


def write_csv(path, rows):
    """Write the comparison table; returns the path actually written."""
    tmp = path.with_name(path.stem + ".part.csv")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return swap_in(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Route flow over a DEM (D8 by default; MFD, Dinf and "
                    "MDinf via --fdir); write each algorithm's stream "
                    "network (GeoJSON), flow-direction raster and a "
                    "comparison table (CSV).",
    )
    parser.add_argument("--dem", nargs="+", default=None, metavar="TIF",
                        help="DEM GeoTIFF tile(s) (default: every .tif in "
                             "./data/02_filled)")
    parser.add_argument("--out", default=None,
                        help="output folder (default: ./data/03_flows next "
                             "to this script)")
    parser.add_argument("--area", type=float, default=DEFAULT_MIN_AREA_KM2,
                        metavar="KM2",
                        help=f"minimum catchment area defining a stream, in km2 "
                             f"(default {DEFAULT_MIN_AREA_KM2:g})")
    parser.add_argument("--fdir", nargs="+", default=None,
                        choices=list(ALGORITHMS) + ["all"], metavar="ALG",
                        help="which flow-routing algorithms to run "
                             "(d8 mfd dinf mdinf, or all); default: "
                             + " ".join(DEFAULT_METHODS))
    parser.add_argument("--keep-work", action="store_true",
                        help="keep the _work folder (mosaic + intermediate rasters)")
    parser.add_argument("--describe", action="store_true",
                        help="print the precise algorithm definitions and exit")
    args = parser.parse_args(argv)

    if args.describe:
        print(__doc__)
        return 0

    started = time.perf_counter()
    script_dir = Path(__file__).resolve().parent

    # ------------------------------------------------- 1. stream definition
    min_area_km2 = args.area
    if min_area_km2 <= 0:
        sys.exit("The catchment area must be positive.")
    if args.fdir and "all" in args.fdir:
        selected = list(ALGORITHMS)
    elif args.fdir:
        selected = [k for k in ALGORITHMS if k in args.fdir]
    else:
        selected = [k for k in ALGORITHMS if k in DEFAULT_METHODS]

    # -------------------------------------------- 2. DEM tiles in, mosaicked
    dem_paths = find_dems(args.dem, script_dir)
    validate_tiles(dem_paths)
    provenance = collect_provenance(dem_paths)
    out_dir = Path(args.out) if args.out else script_dir / "data" / "03_flows"
    work_dir = out_dir / "_work"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"DEM      {len(dem_paths)} tile(s): "
          + ", ".join(p.name for p in dem_paths))
    mosaic_path = build_mosaic(dem_paths, work_dir)
    with rasterio.open(mosaic_path) as src:
        crs, res, bounds, nodata = src.crs, src.res, src.bounds, src.nodata
        transform = src.transform
        n_rows, n_cols = src.height, src.width
    sx, sy = metres_per_map_unit(crs, bounds)
    area_m2 = abs(res[0] * sx * res[1] * sy)
    epsg = crs.to_epsg()
    crs_label = f"EPSG:{epsg}" if epsg else crs.to_string()
    dem_name = (dem_paths[0].name if len(dem_paths) == 1
                else f"mosaic of {len(dem_paths)} tiles")
    print(f"         mosaic {n_cols} x {n_rows} cells, "
          f"{res[0]:g} x {res[1]:g} map units, {crs_label}, "
          f"cell = {area_m2:g} m2")

    threshold_cells = max(1, int(round(min_area_km2 * 1e6 / area_m2)))
    print(f"Streams  contributing area >= {min_area_km2:g} km2 "
          f"= {threshold_cells} cells\n")

    # --------------------------------------- 3. drainage check, D8 tree
    check_drainage(mosaic_path)
    grid = Grid.from_raster(str(mosaic_path))
    conditioned = grid.read_raster(str(mosaic_path))
    fdir_d8 = grid.flowdir(conditioned, routing="d8")

    # ----------------------------------- 4. accumulate, threshold, vectorize
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    networks, masks, stats, written = {}, {}, {}, []
    acc_template = None
    for key in selected:
        spec = ALGORITHMS[key]
        step = time.perf_counter()
        print(f"{spec['title']:<6}{spec['one_liner']} - {spec['founder']}")
        if key == "mdinf":
            if acc_template is None:
                # WBT returns a bare array; a pysheds Raster is needed as
                # the template to re-wrap it (only when MDinf runs alone).
                acc_template = grid.accumulation(fdir_d8, routing="d8")
            accumulation = as_grid_raster(
                mdinf_accumulation(mosaic_path, work_dir), acc_template
            )
        else:
            fdir = (fdir_d8 if key == "d8"
                    else grid.flowdir(conditioned, routing=spec["routing"]))
            accumulation = grid.accumulation(fdir, routing=spec["routing"])
            if acc_template is None:
                acc_template = accumulation
        fdir_grid = fdir if key != "mdinf" else None
        if key == "mdinf":
            # Imported only here, so the script runs without the module.
            try:
                from mdinf import mdinf_flowdir
            except ImportError:
                print("      no MDinf direction raster: companion module "
                      "mdinf.py not found next to 03_flow_router.py - "
                      "skipped.")
            else:
                fdir_grid = mdinf_flowdir(np.asarray(conditioned), nodata,
                                          res[0], res[1],
                                          exponent=MDINF_EXPONENT)
        if fdir_grid is not None:
            written.append(
                write_fdir_raster(out_dir / f"flow_direction_{key}.tif",
                                  key, spec, fdir_grid, crs, transform,
                                  provenance)
            )
        segments, mask = build_network(grid, fdir_d8, accumulation,
                                       threshold_cells, sx, sy)
        networks[key] = segments
        masks[key] = mask
        n_cells = int(mask.sum())
        acc_max = float(np.nanmax(np.asarray(accumulation, dtype="float64")))
        stats[key] = {
            "max_catchment_km2": round(acc_max * area_m2 / 1e6, 2),
            "runtime_s": round(time.perf_counter() - step, 1),
        }

        written.append(
            write_geojson(out_dir / f"flow_network_{key}.geojson", segments,
                          spec, transformer, dem_name, min_area_km2)
        )

        total_km = sum(s["length_m"] for s in segments) / 1000.0
        print(f"      {n_cells} stream cells -> {len(segments)} segments, "
              f"{total_km:.1f} km [{time.perf_counter() - step:.1f} s]\n")

    # --------------------------------------------- 5. the comparison table
    print("Writing the comparison table ...")
    valid_km2 = (np.count_nonzero(np.asarray(conditioned) != nodata)
                 * area_m2 / 1e6)
    rows = summary_rows(masks, networks, stats, area_m2, valid_km2,
                        min_area_km2, threshold_cells, dem_name)
    written.append(write_csv(out_dir / "flow_networks_summary.csv", rows))

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nDone in {time.perf_counter() - started:.1f} s. Written to {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
