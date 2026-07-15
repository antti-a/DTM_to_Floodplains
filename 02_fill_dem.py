#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrologically condition (fill) DEMs with pysheds.

Created on Fri Jul 3 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 2 (see README.md):
    reads   data/01_carved/*.tif        (01_carve_dem.py output)
    writes  data/02_filled/filled_<name>.tif
which is the default input of the next stage (03_flow_router.py).

For every GeoTIFF DEM in the input folder, removes the artefacts that
break downstream flow routing:

  1. fill pits and depressions    (pysheds ``fill_depressions``, the
                                   priority-flood of Barnes, Lehman and
                                   Mulla (2014a); a single-pixel pit is just
                                   a one-pixel depression, so one pass
                                   removes both)
  2. resolve flats                (pysheds ``resolve_flats``, the flat-
                                   resolution method of Barnes, Lehman and
                                   Mulla (2014b): a tiny gradient is baked
                                   into the elevations of filled flats, so
                                   the output DEM itself drains -- not just
                                   some side-channel flow-direction raster)

The inputs are mosaicked *virtually* with ``gdalbuildvrt`` and filled as one
surface, so depressions spanning tile edges fill correctly (filling is a
global operation and cannot be done lazily inside the VRT itself). The filled
mosaic is then cropped back onto each input's exact grid and written as
``data/02_filled/filled_<name>.tif`` -- the same DEMs, but filled.

The outputs are float64 on purpose. Source elevations are quantized to
the centimetre, so flat terrain is full of exactly tied pixels; the
flat-resolution gradient that makes those ties drain is far smaller than
float32 can represent at these elevations, and a float32 output silently
collapses the flats right back -- downstream flow routing then dies on a
DEM that merely looks conditioned. float64 keeps the gradients, and the
fill step verifies that the written mosaic actually drains.

Notes:
  * Depression removal is fill-only (no breaching), by design.
  * Depressions draining across the mosaic's outer edge fill to the edge
    elevation ("outlets at edge"). Only the outer rim of the data acts as
    an outlet: the priority flood is seeded from the outermost valid pixel
    of each row and column, so a depression that drains only into an
    interior nodata hole fills up to the hole's surrounding rim.
  * Pixels that are nodata in an input stay nodata in its output.
  * All inputs must share CRS, pixel size and grid alignment (validated).

References:
  * Barnes, R., Lehman, C. and Mulla, D. (2014a) 'Priority-flood: an
    optimal depression-filling and watershed-labeling algorithm for
    digital elevation models', Computers & Geosciences, 62, pp. 117-127.
  * Barnes, R., Lehman, C. and Mulla, D. (2014b) 'An efficient assignment
    of drainage direction over flat surfaces in raster digital elevation
    models', Computers & Geosciences, 62, pp. 128-135.
  * Bartos, M. (2020) pysheds: simple and fast watershed delineation in
    python. doi:10.5281/zenodo.3822494

Run inside the ``water`` conda environment:

    conda activate water
    python 02_fill_dem.py
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.windows import from_bounds

logger = logging.getLogger("fill_dem")

# --------------------------------------------------------------------------- #
# Defaults / constants
# --------------------------------------------------------------------------- #
NODATA = -9999.0             # KM2 convention; also stamped on the outputs
KEEP_INTERMEDIATE = False    # keep outputs/_work/ after a successful run

# Flat-resolution parameters (pysheds resolve_flats). Each flat pixel is
# raised by FLAT_EPS times its Barnes drainage-gradient count, which grows
# up to ~3 * FLAT_MAX_ITER steps. The values are chosen so even the widest
# resolvable flat inflates by at most 3 mm -- well under the source DEM's
# 1 cm quantization -- while each single step (1e-8 m) stays orders of
# magnitude above float64 resolution at Finnish elevations (~1e-13 m at
# 100 m). FLAT_MAX_ITER must exceed the widest flat in pixels; 100 000
# pixels = 200 km at 2 m, far beyond any filled lake here. (The pysheds
# defaults, eps=1e-5 and max_iter=1000, would inflate big flats by metres
# and leave flats wider than 2 km unresolved.)
FLAT_EPS = 1e-8
FLAT_MAX_ITER = 100_000

# Used when the input carries stage-1 provenance tags (dem_source_tiles).
SOURCE_DATA_CREDIT = (
    "National Land Survey of Finland 2 m elevation model (KM2), CC BY 4.0, "
    "carved with the SYKE 'Tierumpujen uomakorjaus' culvert correction "
    "(CC BY 4.0); source tiles in the dem_source_tiles tag."
)
# Fallback for pre-convention inputs without tags.
SOURCE_DATA_CREDIT_PRESUMED = (
    "2 m DEM in EPSG:3067 (ETRS89 / TM35FIN), vertical datum N2000; "
    "presumed source: National Land Survey of Finland 2 m elevation model "
    "(KM2), CC BY 4.0."
)

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
INPUT_DIR = DATA_DIR / "01_carved"    # carved DEMs (01_carve_dem.py output)
OUT_DIR = DATA_DIR / "02_filled"      # filled DEMs -> 03_flow_router.py input
WORK_DIR = OUT_DIR / "_work"          # mosaic + fill intermediates


# --------------------------------------------------------------------------- #
# GDAL VRT helpers (mosaic the input DEMs virtually)
# --------------------------------------------------------------------------- #
def _gdalbuildvrt_exe() -> str:
    """Locate the ``gdalbuildvrt`` executable shipped with the conda env."""
    exe = shutil.which("gdalbuildvrt")
    if exe:
        return exe
    # Windows-only fallback: on conda/Windows the GDAL CLI tools live in
    # <env>/Library/bin. On Linux/macOS shutil.which() above finds them.
    cand = Path(sys.executable).parent / "Library" / "bin" / "gdalbuildvrt.exe"
    if cand.exists():
        return str(cand)
    raise RuntimeError(
        "gdalbuildvrt not found -- run inside the 'water' conda env."
    )


def build_vrt(
    tiles: Sequence[Path], vrt_path: Path, nodata: float = NODATA
) -> Path:
    """Mosaic ``tiles`` virtually into ``vrt_path`` (no physical merge)."""
    vrt_path = Path(vrt_path)
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _gdalbuildvrt_exe(), "-overwrite", str(vrt_path),
        *[str(t) for t in tiles],
    ]
    logger.info("Building VRT from %d tile(s) -> %s", len(tiles), vrt_path)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    with rasterio.open(vrt_path) as src:
        if src.nodata is None:
            logger.warning(
                "VRT has no nodata; downstream reads assume %s", nodata
            )
        elif src.nodata != nodata:
            logger.warning("VRT nodata %s != expected %s", src.nodata, nodata)
    return vrt_path


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_inputs(dems: Sequence[Path]) -> None:
    """Require one shared CRS, pixel size and grid lattice across ``dems``.

    Filling runs on a mosaic of all inputs, so a tile on a shifted grid or in
    another CRS would be silently resampled by the VRT -- fail instead. Also
    warns on overlapping tiles (gdalbuildvrt stacks them last-wins).
    """
    infos = []
    for path in dems:
        with rasterio.open(path) as src:
            infos.append((path.name, src.crs, src.transform, src.bounds))

    name0, crs0, t0, _ = infos[0]
    res = (abs(t0.a), abs(t0.e))
    for name, crs, t, _ in infos[1:]:
        if crs != crs0:
            raise ValueError(f"{name}: CRS {crs} != {crs0} ({name0})")
        if (abs(t.a), abs(t.e)) != res:
            raise ValueError(
                f"{name}: pixel size {(abs(t.a), abs(t.e))} != {res} ({name0})"
            )
        dx = (t.c - t0.c) / t.a
        dy = (t.f - t0.f) / t.e
        if abs(dx - round(dx)) > 1e-6 or abs(dy - round(dy)) > 1e-6:
            raise ValueError(
                f"{name}: grid origin misaligned with {name0} by "
                f"({dx % 1:.6f}, {dy % 1:.6f}) pixels"
            )

    for i, (name_a, _, _, ba) in enumerate(infos):
        for name_b, _, _, bb in infos[i + 1:]:
            if (ba.left < bb.right and bb.left < ba.right
                    and ba.bottom < bb.top and bb.bottom < ba.top):
                logger.warning(
                    "Tiles %s and %s overlap; VRT keeps the later one "
                    "in the overlap", name_a, name_b,
                )


# --------------------------------------------------------------------------- #
# Mosaic materialisation
# --------------------------------------------------------------------------- #
def materialize_mosaic(vrt_path: Path, out_tif: Path,
                       nodata: float = NODATA) -> Path:
    """Copy the virtual mosaic into a real GeoTIFF for the fill.

    Filling is a global operation over one in-memory surface anyway, so
    reading the mosaic once here costs nothing extra and normalizes the
    nodata value and any non-finite pixels before the fill sees them.
    """
    with rasterio.open(vrt_path) as src:
        dem = src.read(1).astype("float64")
        if src.nodata is not None and src.nodata != nodata:
            dem[dem == src.nodata] = nodata
        dem[~np.isfinite(dem)] = nodata
        logger.info(
            "Mosaic grid: %d x %d pixels @ %g m, bounds %s",
            src.width, src.height, src.transform.a, tuple(src.bounds),
        )
        return write_dem(
            out_tif, dem, src.transform, src.crs, nodata, dtype="float64",
        )


# --------------------------------------------------------------------------- #
# The fill (pysheds)
# --------------------------------------------------------------------------- #
def count_undrainable(dem_tif: Path, nodata: float = NODATA) -> tuple[int, int]:
    """Count interior pixels with no strictly lower neighbour.

    Such a pixel (a pit or a flat tie) stops every flow-routing algorithm.
    Nodata pixels count as outlets (like the fill treats them) and edge
    pixels drain off-grid, so a healthy conditioned DEM leaves only a
    scattered handful. Returns (undrainable_pixels, valid_pixels).
    """
    with rasterio.open(dem_tif) as src:
        z = src.read(1).astype("float64")
        if src.nodata is not None:
            nodata = src.nodata
    valid = np.isfinite(z) & (z != nodata)
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
    stuck = valid[1:-1, 1:-1] & ~has_lower
    return int(stuck.sum()), int(valid.sum())


def fill(mosaic_tif: Path, work_dir: Path = WORK_DIR) -> Path:
    """Fill pits + depressions and resolve flats; return the filled mosaic.

    ``fill_depressions`` is a priority-flood fill (Barnes, Lehman and
    Mulla, 2014a) that removes single-pixel pits along with larger
    depressions; ``resolve_flats`` (Barnes, Lehman and Mulla, 2014b)
    applies a tiny gradient (``FLAT_EPS`` per step, at most
    ``FLAT_MAX_ITER`` steps) across flats so the result drains
    everywhere. That gradient only survives the file round-trip in
    float64 (see the module docstring), so the written mosaic is checked
    here: this function fails loudly rather than hand a non-draining
    "filled" DEM downstream.

    The nodata mask is taken before filling (``fill_depressions`` mutates
    its input in place) and re-applied afterwards: ``resolve_flats`` does
    not exclude nodata regions, whose constant elevation reads as one big
    flat, so they come back slightly inflated and must not be mistaken
    for valid interior minima by the drainage check.
    """
    from pysheds.grid import Grid

    filled_tif = work_dir / "mosaic_filled.tif"

    grid = Grid.from_raster(str(mosaic_tif))
    dem = grid.read_raster(str(mosaic_tif))
    nodata_mask = (np.asarray(dem) == NODATA) | ~np.isfinite(np.asarray(dem))

    logger.info("pysheds fill_depressions (priority-flood) ...")
    flooded = grid.fill_depressions(dem)

    logger.info("pysheds resolve_flats (eps=%g, max_iter=%d) ...",
                FLAT_EPS, FLAT_MAX_ITER)
    inflated = grid.resolve_flats(flooded, eps=FLAT_EPS,
                                  max_iter=FLAT_MAX_ITER)

    band = np.asarray(inflated, dtype="float64")
    band[nodata_mask | ~np.isfinite(band)] = NODATA
    with rasterio.open(mosaic_tif) as src:
        write_dem(filled_tif, band, src.transform, src.crs, NODATA,
                  dtype="float64")

    stuck, valid = count_undrainable(filled_tif)
    logger.info("Drainage check: %d of %d valid pixels undrainable", stuck, valid)
    if stuck > max(1, valid // 1000):
        raise RuntimeError(
            f"Filled mosaic does not drain: {stuck} of {valid} pixels have no "
            f"lower neighbour. The flat-resolution gradients were lost - "
            f"check that every step keeps the DEM in float64, and that "
            f"FLAT_MAX_ITER exceeds the widest flat in pixels."
        )
    return filled_tif


# --------------------------------------------------------------------------- #
# Output helper
# --------------------------------------------------------------------------- #
def write_dem(
    path: Path,
    array: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    nodata: float = NODATA,
    dtype: str = "float32",
    predictor: int | None = None,
    tags: dict | None = None,
) -> Path:
    """Write a single-band GeoTIFF (tiled + compressed) in ``dtype``.

    ``predictor`` overrides the dtype-based default (3 = floating-point,
    2 = integer horizontal differencing).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if predictor is None:
        # predictor 3 = floating-point, 2 = integer horizontal differencing
        predictor = 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2
    profile = dict(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=predictor,
        bigtiff="IF_SAFER",   # large rasters can exceed 4 GB
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        if tags:
            dst.update_tags(**tags)
    logger.info("Wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# Crop the filled mosaic back onto each input grid
# --------------------------------------------------------------------------- #
def crop_back(
    filled_tif: Path,
    dem_path: Path,
    out_dir: Path = OUT_DIR,
    nodata: float = NODATA,
) -> Path:
    """Write ``out_dir/filled_<name>.tif`` on ``dem_path``'s exact grid.

    Reads the input's window out of the filled mosaic (same lattice by
    construction, so this is a pure crop), re-applies the input's nodata
    mask, and logs how much was filled as a QC summary.
    """
    dem_path = Path(dem_path)
    with rasterio.open(dem_path) as src:
        orig = src.read(1).astype("float64")
        if src.nodata is not None and src.nodata != nodata:
            orig[orig == src.nodata] = nodata
        transform, crs, bounds = src.transform, src.crs, src.bounds
        src_tags = src.tags()

    with rasterio.open(filled_tif) as src:
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        filled = src.read(
            1, window=window, boundless=True, fill_value=nodata
        ).astype("float64")
    if filled.shape != orig.shape:
        raise RuntimeError(
            f"{dem_path.name}: crop shape {filled.shape} != {orig.shape}"
        )

    valid = (orig != nodata) & np.isfinite(orig)
    filled[~valid] = nodata

    diff = filled[valid] - orig[valid]
    if diff.size and diff.min() < -1e-3:
        raise RuntimeError(
            f"{dem_path.name}: fill lowered pixels by up to {-diff.min():.3f} m"
        )
    raised = int(np.count_nonzero(diff > 0))
    logger.info(
        "%s: %d of %d valid pixels raised (%.2f %%), max fill depth %.3f m",
        dem_path.name, raised, int(valid.sum()),
        100.0 * raised / max(1, valid.sum()), float(diff.max()) if diff.size else 0.0,
    )

    forwarded = {k: src_tags[k] for k in ("dem_source_tiles", "dem_carve")
                 if k in src_tags}
    if "dem_source_tiles" not in forwarded:
        logger.warning(
            "%s: input carries no provenance tags (pre-convention carved "
            "tile); recording presumed source", dem_path.name,
        )
    return write_dem(
        Path(out_dir) / f"filled_{dem_path.stem}.tif",
        filled, transform, crs, nodata, dtype="float64",
        tags=dict(
            title="Hydrologically conditioned (depression-filled) DEM",
            fill_method="pysheds fill_depressions (priority-flood, Barnes "
                        "et al. 2014) + resolve_flats (Barnes et al. 2014, "
                        f"eps={FLAT_EPS:g}, max_iter={FLAT_MAX_ITER}), run "
                        "on the virtual mosaic of all input tiles and "
                        "cropped back to this tile's grid; float64 preserves "
                        "the flat-resolution gradients",
            dem_fill="pysheds_fill_depressions_resolve_flats",
            source_data_credit=(SOURCE_DATA_CREDIT
                                if "dem_source_tiles" in forwarded
                                else SOURCE_DATA_CREDIT_PRESUMED),
            generated_by="02_fill_dem.py",
            **forwarded,
        ),
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def fill_all(
    input_dir: Path = INPUT_DIR,
    out_dir: Path = OUT_DIR,
    work_dir: Path = WORK_DIR,
    keep_intermediate: bool = KEEP_INTERMEDIATE,
) -> list[Path]:
    """Fill every ``*.tif`` in ``input_dir``; return the written paths."""
    dems = sorted(Path(input_dir).glob("*.tif"))
    if not dems:
        raise FileNotFoundError(f"No .tif DEMs found in {input_dir}")
    logger.info("Found %d DEM(s) in %s", len(dems), input_dir)

    validate_inputs(dems)
    work_dir.mkdir(parents=True, exist_ok=True)

    vrt = build_vrt(dems, work_dir / "inputs_mosaic.vrt")
    mosaic_tif = materialize_mosaic(vrt, work_dir / "mosaic.tif")
    filled_tif = fill(mosaic_tif, work_dir)

    written = [crop_back(filled_tif, d, out_dir) for d in dems]

    if not keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("Removed intermediates in %s", work_dir)
    return written


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    written = fill_all()
    print("\nFilled DEM(s):")
    for p in written:
        print(f"  {p}")
